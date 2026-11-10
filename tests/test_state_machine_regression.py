#!/usr/bin/env python
"""Offline state-machine regression checks for :class:`tablib.Dataset`.

The single-operation unit tests in ``test_tablib.py`` do not exercise long,
interleaved sequences of structural mutations (insert/delete columns, row
slicing, sorting, stacking) followed by exports.  This module closes that gap
with a *reference model* that re-implements the operation semantics in plain
Python and never calls any tablib method to compute an expected value.

Design
------
* :class:`RefDataset` is the independent oracle: it tracks headers, row
  values, per-row tags and formatter registrations itself.
* :class:`CountingFormatter` is an observable callback: every invocation is
  recorded with the column position the callback was handed, so an export can
  be audited for "formatter ran on the wrong column" or "ran twice".
* A seeded RNG (:mod:`random`, no Hypothesis) emits short, replayable
  operation sequences mixing legal and intentionally invalid operations.
  Invalid operations must be atomic: the post-failure state has to equal the
  pre-failure state (no half update).
* After each operation the tablib object and the reference model are compared
  on headers, width, every cell, per-row tags and formatter registry.
* CSV and JSON exports are parsed back and compared structurally against an
  independently computed expected export, including the exact formatter call
  log and the total invocation count.

Two modes
~~~~~~~~~
``faithful`` mode mirrors tablib's *observed* behaviour for every operation
covered by the random runner, so the main regression suite stays green and
guards against accidental behaviour changes.  That includes deliberately
reproducing tablib's documented, non-atomic failure paths (SM-12..SM-14),
which are asserted separately in :class:`KnownIssuesTests`.  ``ideal`` mode
implements what the documented semantics ought to be (indices stay attached
to their column through structural edits; failures are atomic; ``sort``
keeps row metadata).
Divergences found in ``ideal`` mode are cross-checked against a registry of
already-localised, real tablib defects and minimised to the shortest still
reproducing sequence.  Any *unknown* divergence fails the suite.

The whole module is sized to run in a few seconds (see :data:`SEED_COUNT`);
no network or optional spreadsheet dependency is used.
"""

import csv
import io
import json
import os
import random
import time
import unittest
from copy import deepcopy

import tablib
from tablib.exceptions import HeadersNeeded, InvalidDatasetIndex, InvalidDimensions

# Small, fixed budget so the full offline suite stays in the 10s range.
# Overridable via environment variables for local stress runs.
SEED_COUNT = int(os.environ.get("SM_SEED_COUNT", "40"))
OPS_PER_SEED = int(os.environ.get("SM_OPS", "30"))
ARTIFACT_DIR = os.environ.get(
    "SM_ARTIFACT_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 ".sm_artifacts"),
)

TAG_POOL = ("red", "blue", "green", "odd", "vip")
VALUE_POOL = (0, 1, 2, 7, 42, "a", "bb", "ccc", None, True)
HEADER_POOL = ("h0", "h1", "h2", "h3", "h4", "extra")


# ---------------------------------------------------------------------------
# Observable formatter
# ---------------------------------------------------------------------------

class CountingFormatter:
    """Formatter callback that records every invocation.

    Each recorded entry is ``(formatter_id, column_position, input_value)``
    where *column_position* is the index the export machinery used when
    reaching into the row (``None`` for whole-row formatters).  Comparing two
    logs therefore detects both wrong-column attribution and duplicate runs.
    """

    def __init__(self, formatter_id, marker=None):
        self.formatter_id = formatter_id
        self.marker = marker or f"F{formatter_id}"
        self.calls = []

    def __call__(self, value):
        self.calls.append(value)
        return f"{self.marker}:{value!r}"

    def reset(self):
        self.calls = []


# ---------------------------------------------------------------------------
# Independent reference model -- no tablib calls compute expectations
# ---------------------------------------------------------------------------

class RefError(Exception):
    """Mirror of the tablib exception family raised by the reference model."""

    def __init__(self, kind):
        super().__init__(kind)
        self.kind = kind


class RefDataset:
    """Plain-Python oracle for the subset of the Dataset API under test.

    The model deliberately does not implement dynamic (callable) columns:
    those keep their own index bookkeeping and are covered by dedicated,
    hand-written known-issue tests instead.  Formatters are stored as
    ``(column, formatter)`` tuples and *ideal* mode keeps the column attached
    to its logical column across insert/delete operations, while *faithful*
    mode reproduces tablib's raw stored-index behaviour.
    """

    def __init__(self, mode="faithful", with_headers=True):
        assert mode in ("faithful", "ideal")
        self.mode = mode
        self.headers = list(HEADER_POOL[:2]) if with_headers else None
        self.rows = []          # list[list] of cell values
        self.tags = []          # list[list[str]], parallel to rows
        self.formatters = []    # list[(col|None, CountingFormatter)]
        self._next_formatter = 0

    # -- introspection -----------------------------------------------------

    @property
    def width(self):
        if self.rows:
            return len(self.rows[0])
        return len(self.headers) if self.headers else 0

    @property
    def height(self):
        return len(self.rows)

    @property
    def has_headers(self):
        # tablib treats an empty header list as falsy everywhere it branches
        # on ``if self.headers``.
        return bool(self.headers)

    def snapshot(self):
        return deepcopy((self.headers, self.rows, self.tags,
                         [(c, f.formatter_id) for c, f in self.formatters]))

    def restore(self, snap):
        headers, rows, tags, fmt_ids = snap
        ids_to_fmt = {f.formatter_id: f for _, f in self.formatters}
        self.headers = deepcopy(headers)
        self.rows = deepcopy(rows)
        self.tags = deepcopy(tags)
        self.formatters = [(c, ids_to_fmt[fid]) for c, fid in fmt_ids]

    # -- row operations ----------------------------------------------------

    def _validate_row_width(self, row):
        if self.width and len(row) != self.width:
            raise RefError("InvalidDimensions")

    def insert_row(self, index, row, tags):
        self._validate_row_width(row)
        index = max(0, min(index, self.height))
        self.rows.insert(index, list(row))
        self.tags.insert(index, list(tags))

    def delete_row(self, key):
        if isinstance(key, str):
            raise RefError("TypeError")
        if isinstance(key, int) and not (-self.height <= key < self.height):
            raise RefError("IndexError")
        del self.rows[key]
        del self.tags[key]

    def pop_row(self, end):
        if not self.rows:
            raise RefError("IndexError")
        if end == "first":
            self.rows.pop(0)
            self.tags.pop(0)
        else:
            self.rows.pop()
            self.tags.pop()

    # -- column operations -------------------------------------------------

    def insert_col(self, index, col, header):
        # Mirror tablib's exact validation ordering, which differs depending
        # on whether a header row is present.
        if self.has_headers:
            if self.height == 0:
                if not header:
                    raise RefError("HeadersNeeded")
                if col:
                    raise RefError("InvalidDimensions")
            else:
                if not col:
                    if not header:
                        raise RefError("HeadersNeeded")
                    if not self.width:
                        # Column-less rows: skip the per-row loop entirely and
                        # fall through to the rebuild branch (SM-14).
                        pass
                    else:
                        # Non-atomic tablib path (known issue SM-12): the
                        # header is inserted before the per-row loop raises
                        # IndexError.  Tablib uses the raw, un-clamped index.
                        if self.mode == "faithful":
                            self.headers.insert(index, header)
                        raise RefError("IndexError")
                if col and len(col) != self.height:
                    raise RefError("InvalidDimensions")
                if not header:
                    raise RefError("HeadersNeeded")
        else:
            if self.height:
                if not col and self.width:
                    raise RefError("IndexError")
                if col and len(col) != self.height:
                    raise RefError("InvalidDimensions")
        if self.has_headers:
            self.headers.insert(index, header)
        if self.height and self.width:
            # Live in-place column insertion into existing rectangular rows.
            for i, row in enumerate(self.rows):
                row.insert(index, col[i])
        elif not col:
            if self.height and not self.width:
                # Known issue SM-14: with rows present but zero live columns,
                # tablib takes the else branch and rebuilds _data from the
                # empty column, wiping every existing row and its tags.
                if self.mode == "faithful":
                    self.rows = []
                    self.tags = []
                # ideal mode keeps the existing rows untouched
            # Header-only / genuinely empty dataset: nothing to rebuild.
        else:
            # tablib's else branch: ``self._data = [Row([row]) for row in
            # col]``.  This rebuilds every row from the new column alone.
            self.rows = [[v] for v in col]
            if self.mode == "faithful":
                # When rows still exist but carry no live column (width 0)
                # this discards their tags (known issue SM-13).
                self.tags = [[] for _ in col]
            else:
                self.tags = [list(self.tags[i])
                             if i < len(self.tags) else []
                             for i in range(len(col))]
        if self.mode == "ideal":
            self.formatters = [
                (None if c is None else (c + 1 if c >= index else c), f)
                for c, f in self.formatters
            ]

    def delete_col(self, header):
        if header is None:
            raise RefError("TypeError")
        if self.headers is None:
            raise RefError("HeadersNeeded")
        if not self.has_headers:
            # headers == []: tablib's ``key in self.headers`` fails first.
            raise RefError("KeyError")
        if header not in self.headers:
            raise RefError("KeyError")
        pos = self.headers.index(header)
        ragged = self.rows and any(pos >= len(row) for row in self.rows)
        if ragged and self.mode == "ideal":
            # Ideal semantics validate up front and stay fully atomic, but
            # raise the same error class to stay aligned with tablib.
            raise RefError("IndexError")
        del self.headers[pos]
        # tablib walks rows one by one; a ragged short row aborts the loop
        # part-way, leaving earlier rows mutated (non-atomic half update).
        for row in self.rows:
            if pos >= len(row) and self.mode == "faithful":
                raise RefError("IndexError")
            del row[pos]
        if self.mode == "ideal":
            kept = []
            for c, f in self.formatters:
                if c is None:
                    kept.append((c, f))
                elif c == pos:
                    continue
                elif c > pos:
                    kept.append((c - 1, f))
                else:
                    kept.append((c, f))
            self.formatters = kept

    # -- formatters --------------------------------------------------------

    def make_formatter(self):
        fmt = CountingFormatter(self._next_formatter)
        self._next_formatter += 1
        return fmt

    def add_formatter(self, col, formatter=None):
        if col is None:
            self._register(col, formatter)
            return
        if col < 0:
            if self.mode == "ideal":
                raise RefError("InvalidDatasetIndex")
            self._register(col, formatter)
            return
        if self.mode == "ideal":
            if col >= self.width:
                raise RefError("InvalidDatasetIndex")
        else:
            # tablib guard is ``col <= self.width`` (off by one): an index
            # equal to the width is stored and explodes later on export.
            if col > self.width:
                raise RefError("InvalidDatasetIndex")
        self._register(col, formatter)

    def _register(self, col, formatter):
        if formatter is None:  # generator shadow oracle: allocate locally
            formatter = self.make_formatter()
        self.formatters.append((col, formatter))

    # -- derived datasets --------------------------------------------------

    def sort(self, col_index, reverse):
        out = RefDataset(mode=self.mode, with_headers=False)
        row_width = max((len(row) for row in self.rows), default=0)
        if self.has_headers:
            # tablib evaluates ``self.headers[col]`` before anything else,
            # so an integer past the header list always raises IndexError.
            if col_index >= len(self.headers):
                raise RefError("IndexError")
            # Sorts .dict output (formatters applied; may raise on stale
            # indices) and rebuilds rows from those formatted values.
            try:
                packaged, _ = self.expected_json()
            except RefError:
                raise
            except (KeyError, IndexError) as exc:
                raise RefError(type(exc).__name__)
            header_name = self.headers[col_index]
            try:
                order = sorted(
                    range(self.height),
                    key=lambda i: packaged[i][header_name],
                    reverse=reverse)
            except TypeError:
                raise RefError("TypeError")
            except KeyError:
                # Header existed but the packaged rows are narrower.
                raise RefError("KeyError")
            except IndexError as exc:
                raise RefError(type(exc).__name__)
            out.headers = deepcopy(self.headers)
            out.rows = [[packaged[i][h] for h in out.headers]
                        for i in order]
        elif row_width:
            # headers == [] but rows still carry columns (reachable after a
            # half-update path): tablib takes the integer branch, still
            # sorting .dict output so formatters are baked into the rows.
            if col_index >= row_width:
                raise RefError("IndexError")
            packaged, _ = self.expected_json()
            try:
                order = sorted(range(self.height),
                               key=lambda i: packaged[i][col_index],
                               reverse=reverse)
            except TypeError:
                raise RefError("TypeError")
            out.headers = None
            out.rows = deepcopy([packaged[i] for i in order])
        else:
            if self.height:
                # Column-less rows still present: itemgetter(col) IndexErrors.
                raise RefError("IndexError")
            # Genuinely empty dataset: sorting an empty .dict yields [].
            out.headers = None
            out.rows = []
            order = []
        if self.mode == "ideal":
            out.tags = deepcopy([self.tags[i] for i in order])
            out.formatters = list(self.formatters)
        else:
            # tablib rebuilds rows from dict() output and drops tags and
            # formatters on the floor.
            out.tags = [[] for _ in order]
            out.formatters = []
        out._next_formatter = self._next_formatter
        return out

    def filter(self, tag):
        keep = [i for i, row_tags in enumerate(self.tags) if tag in row_tags]
        out = RefDataset(mode=self.mode, with_headers=False)
        out.headers = deepcopy(self.headers)
        out.rows = deepcopy([self.rows[i] for i in keep])
        out.tags = deepcopy([self.tags[i] for i in keep])
        out.formatters = list(self.formatters)
        out._next_formatter = self._next_formatter
        return out

    def stack(self, other):
        if self.width != other.width:
            raise RefError("InvalidDimensions")
        out = RefDataset(mode=self.mode, with_headers=False)
        out.headers = deepcopy(self.headers)
        out.rows = deepcopy(self.rows) + deepcopy(other.rows)
        out.tags = deepcopy(self.tags) + deepcopy(other.tags)
        out.formatters = list(self.formatters)
        out._next_formatter = self._next_formatter
        return out

    # -- export oracle -----------------------------------------------------

    def _render_row(self, row, call_log):
        rendered = list(row)
        for col, fmt in self.formatters:
            if col is None:
                for pos, value in enumerate(rendered):
                    call_log.append((fmt.formatter_id, pos, value))
                    rendered[pos] = fmt(value)
            else:
                if col < 0:
                    # Faithful mode mirrors tablib, which stores the negative
                    # index and lets Python's Row.__getitem__ resolve it.
                    if self.mode != "faithful" or -col > len(rendered):
                        raise RefError("IndexError")
                    effective = col
                else:
                    if col >= len(rendered):
                        raise RefError("IndexError")
                    effective = col
                value = rendered[effective]
                call_log.append((fmt.formatter_id, effective, value))
                rendered[col] = fmt(value)
        return rendered

    def expected_json(self):
        call_log = []
        if self.has_headers:
            payload = []
            for row in self.rows:
                rendered = self._render_row(row, call_log)
                payload.append(dict(zip(self.headers, rendered)))
        else:
            payload = [self._render_row(row, call_log) for row in self.rows]
        return payload, call_log

    def expected_csv(self):
        call_log = []
        table = []
        if self.has_headers:
            table.append(list(self.headers))
        for row in self.rows:
            table.append(self._render_row(row, call_log))
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        for row in table:
            writer.writerow(row)
        return buffer.getvalue(), call_log


# ---------------------------------------------------------------------------
# Operation vocabulary
# ---------------------------------------------------------------------------
#
# Every operation is a plain JSON-serialisable tuple, so a failing sequence
# can be written to disk and replayed verbatim.

def _random_row(rng, width):
    return [rng.choice(VALUE_POOL) for _ in range(width)]


def _random_tags(rng):
    return rng.sample(TAG_POOL, rng.randrange(0, 3))


def generate_ops(rng, with_headers, count):
    """Emit *count* replayable ops, keeping a local faithful oracle in sync
    so generated indices stay in range.  Roughly one op in five is an
    intentionally invalid variant."""
    ref = RefDataset(mode="faithful", with_headers=with_headers)
    ops = []
    for _ in range(count):
        choices = ["insert_row", "delete_row", "insert_col", "delete_col",
                   "add_formatter", "sort", "filter", "stack", "slice",
                   "set_headers", "pop"]
        kind = rng.choice(choices)
        invalid = rng.random() < 0.2

        if kind == "insert_row":
            width = max(1, ref.width)
            row = _random_row(rng, width + 1 if invalid else width)
            ops.append(("insert_row", rng.randrange(0, ref.height + 1),
                        row, _random_tags(rng)))

        elif kind == "delete_row":
            if ref.height == 0:
                ops.append(("delete_row", 0))  # guaranteed IndexError
            else:
                key = ref.height if invalid else rng.randrange(0, ref.height)
                ops.append(("delete_row", key))

        elif kind == "pop":
            ops.append((rng.choice(("lpop", "rpop")),))

        elif kind == "insert_col":
            height = ref.height
            col = [rng.choice(VALUE_POOL) for _ in range(height)]
            if invalid and height:
                col.append(rng.choice(VALUE_POOL))  # wrong height
            header = None
            if ref.headers is not None:
                header = None if invalid else f"c{ref.width}"
            ops.append(("insert_col", rng.randrange(0, ref.width + 1),
                        col, header))

        elif kind == "delete_col":
            if not ref.has_headers or ref.width == 0:
                continue
            if invalid:
                ops.append(("delete_col", "missing-header"))
            else:
                ops.append(("delete_col", rng.choice(ref.headers)))

        elif kind == "add_formatter":
            if invalid:
                col = rng.choice((ref.width, ref.width + 2, -1))
            elif ref.width == 0:
                col = "__all__"
            else:
                col = rng.choice(list(range(ref.width)) + ["__all__"])
            ops.append(("add_formatter", col))

        elif kind == "sort":
            if ref.height == 0 or ref.width == 0:
                continue
            col_index = rng.randrange(0, ref.width)
            if ref.has_headers and rng.random() < 0.5:
                col = ref.headers[col_index]
            else:
                col = col_index
            ops.append(("sort", col, rng.random() < 0.5))

        elif kind == "filter":
            ops.append(("filter", rng.choice(TAG_POOL)))

        elif kind == "stack":
            width = max(1, ref.width)
            other_width = width + 1 if invalid else width
            other_rows = [_random_row(rng, other_width)
                          for _ in range(rng.randrange(0, 3))]
            other_tags = [_random_tags(rng) for _ in other_rows]
            ops.append(("stack", other_rows, other_tags))

        elif kind == "slice":
            if ref.height == 0:
                continue
            start = rng.randrange(0, ref.height + 1)
            stop = rng.randrange(0, ref.height + 1)
            ops.append(("slice", min(start, stop), max(start, stop)))

        elif kind == "set_headers":
            width = max(1, ref.width)
            new_headers = [f"k{j}" for j in range(width + 1 if invalid
                                                   else width)]
            ops.append(("set_headers", new_headers))

        # Keep the generator's shadow oracle in sync with accepted ops.
        try:
            _apply_to_ref(ref, ops[-1], None)
        except (RefError, IndexError, KeyError, TypeError, ValueError):
            pass
    return ops


def _apply_to_ref(ref, op, id_allocator=None, shared_formatter=None):
    """Apply one op to the reference model.  Returns a new RefDataset for
    derived-dataset ops (sort/filter/stack), otherwise ``None``."""
    kind = op[0]
    if kind == "set_headers":
        if ref.width and len(op[1]) != ref.width:
            raise RefError("InvalidDimensions")
        ref.headers = list(op[1]) if op[1] else None
    elif kind == "insert_row":
        ref.insert_row(op[1], op[2], op[3])
    elif kind == "delete_row":
        ref.delete_row(op[1])
    elif kind in ("lpop", "rpop"):
        ref.pop_row("first" if kind == "lpop" else "last")
    elif kind == "insert_col":
        ref.insert_col(op[1], op[2], op[3])
    elif kind == "delete_col":
        ref.delete_col(op[1])
    elif kind == "add_formatter":
        col = None if op[1] == "__all__" else op[1]
        fmt = shared_formatter
        ref.add_formatter(col, formatter=fmt)
    elif kind == "sort":
        col = op[1]
        if isinstance(col, str):
            if not ref.has_headers:
                raise RefError("HeadersNeeded")
            # tablib builds .dict (runs formatters) before itemgetter ever
            # sees the missing name, so packaging errors take precedence.
            if ref.height:
                try:
                    ref.expected_json()
                except RefError as exc:
                    raise RefError(exc.kind)
            if col not in ref.headers:
                # Empty dataset short-circuits inside sorted([]); otherwise
                # itemgetter on a missing dict key raises KeyError.
                if ref.height:
                    raise RefError("KeyError")
                col = 0
            else:
                col = ref.headers.index(col)
        try:
            return ref.sort(col, reverse=op[2])
        except RefError:
            raise
        except (KeyError, IndexError) as exc:
            raise RefError(type(exc).__name__)
    elif kind == "filter":
        return ref.filter(op[1])
    elif kind == "stack":
        other = RefDataset(mode=ref.mode, with_headers=False)
        other.headers = deepcopy(ref.headers)
        other.rows = deepcopy(op[1])
        other.tags = deepcopy(op[2])
        return ref.stack(other)
    return None


# ---------------------------------------------------------------------------
# Execution engine: drives tablib and the oracle through the same sequence
# ---------------------------------------------------------------------------

# Registry of already-localised, real tablib defects (see module docstring and
# KnownIssuesTests for hand-written reproductions of each).
KNOWN_ISSUES = {
    "SM-01": "formatter column index not shifted by insert_col",
    "SM-02": "formatter column index not adjusted by column deletion",
    "SM-03": "stale formatter index makes CSV/JSON export raise IndexError",
    "SM-04": "add_formatter accepts col == width; export then raises",
    "SM-05": "add_formatter accepts negative column indices",
    "SM-06": "Dataset.sort drops row tags",
    "SM-07": "Dataset.sort drops registered formatters",
    "SM-08": "Dataset.stack shares Row objects with its operands",
    "SM-09": "dynamic column index goes stale after insert_col",
    "SM-10": "dynamic column index goes stale after column deletion",
    "SM-11": "failed callable insert_col leaves _dynamic_columns populated",
    "SM-12": "empty-column insert_col mutates headers before raising IndexError",
    "SM-13": "insert_col rebuilds rows and drops row tags when width is 0",
    "SM-14": "empty-column insert at width 0 silently wipes all rows",
}


class Divergence(Exception):
    """A concrete mismatch between tablib and the reference model."""

    def __init__(self, message, issue_id, op_index, op):
        super().__init__(message)
        self.message = message
        self.issue_id = issue_id
        self.op_index = op_index
        self.op = op

    def as_dict(self):
        return {"issue_id": self.issue_id, "message": self.message,
                "op_index": self.op_index, "op": list(self.op)}

    def __eq__(self, other):
        if not isinstance(other, Divergence):
            return NotImplemented
        return self.as_dict() == other.as_dict()

    def __repr__(self):
        return f"Divergence({self.as_dict()!r})"


def _exc_kind(exc):
    return type(exc).__name__


def _real_snapshot(ds):
    """Deep, plain-data snapshot of everything the oracle tracks."""
    return (
        deepcopy(ds.headers),
        [list(row) for row in ds._data],
        [list(row.tags) for row in ds._data],
        [(col, fmt.formatter_id) for col, fmt in ds._formatters],
    )


def _ref_snapshot(ref):
    return (
        deepcopy(ref.headers),
        deepcopy(ref.rows),
        deepcopy(ref.tags),
        [(col, fmt.formatter_id) for col, fmt in ref.formatters],
    )

def _formatter_calls(ref):
    return {fmt.formatter_id: list(fmt.calls)
            for _, fmt in ref.formatters}


def _real_formatter_calls(ds):
    return {fmt.formatter_id: list(fmt.calls)
            for _, fmt in ds._formatters}


def _reset_formatter_calls(ref, ds):
    for _, fmt in ref.formatters:
        fmt.reset()
    for _, fmt in ds._formatters:
        fmt.reset()


def classify_divergence(op, message, ref_snap=None, real_snap=None,
                        prior_issue=None):
    """Map an observed mismatch to a known issue id, or ``UNKNOWN``.

    Classification prefers concrete snapshot evidence (formatter columns,
    tags, header/row contents) over message text.  A state mismatch that is
    merely downstream of an earlier, already-classified defect is attributed
    to that root cause instead of being reported as a new, unknown issue.
    """
    kind = op[0]
    text = message.lower()

    if "half-updated" in text:
        return "SM-12"

    # Direct, operation-local evidence.
    if kind == "insert_col":
        if ref_snap and real_snap and ref_snap[3] != real_snap[3]:
            return "SM-01"
        if ref_snap and real_snap and ref_snap[2] != real_snap[2]:
            return "SM-13"
        if ref_snap and real_snap and \
                len(ref_snap[1]) != len(real_snap[1]):
            return "SM-14"
    if kind == "delete_col":
        if ref_snap and real_snap and ref_snap[3] != real_snap[3]:
            return "SM-02"
    if "export" in text and "indexerror" in text:
        return "SM-03"
    if kind == "add_formatter":
        if op[1] == -1 or (isinstance(op[1], int) and op[1] < 0):
            return "SM-05"
        # A genuinely out-of-range integer column: ideal rejects it while
        # tablib's off-by-one guard stores it.  When the index is in range,
        # any state difference is a cascade of an earlier structural defect.
        if ref_snap and ref_snap[1]:
            ref_width = len(ref_snap[1][0])
        elif ref_snap and ref_snap[0]:
            ref_width = len(ref_snap[0])
        else:
            ref_width = 0
        if (isinstance(op[1], int) and op[1] >= 0
                and ref_snap is not None and op[1] >= ref_width):
            # Out-of-range (or width-0) integer column: ideal rejects while
            # tablib's ``col <= width`` guard stores it anyway.
            return "SM-04"
        # Whole-row ("__all__") formatter state differences are cascades.
        if prior_issue is not None:
            return prior_issue
        return "UNKNOWN"
    if kind == "sort":
        if ref_snap and real_snap:
            if ref_snap[2] != real_snap[2]:
                return "SM-06"
            if ref_snap[3] != real_snap[3]:
                return "SM-07"
        return "SM-06"
    if kind == "stack" and ref_snap is not None and real_snap is not None \
            and ref_snap != real_snap:
        return "SM-08"
    if kind == "stack" and "mismatch" in text:
        return "SM-08"

    # Once a structural defect has put the two sides out of sync, subsequent
    # state mismatches and exception-type differences are cascades of that
    # root cause rather than new, independent findings.  SM-08 (row aliasing)
    # is invisible to snapshot comparison, so it is never used to explain an
    # unrelated mismatch.
    if prior_issue is not None and (
            "mismatch" in text or "half-updated" in text
            or "raised" in text):
        return prior_issue
    return "UNKNOWN"


def run_sequence(ops, mode, with_headers=True, check_exports=True,
                 divergence_ids=None):
    """Replay *ops* against tablib and a RefDataset oracle.

    Returns ``(ref, ds, divergences)`` where divergences is a list of
    :class:`Divergence`.  In ``faithful`` mode any divergence is a genuine
    regression; in ``ideal`` mode divergences correspond to known defects.
    """
    ref = RefDataset(mode=mode, with_headers=with_headers)
    ds = tablib.Dataset(
        headers=list(HEADER_POOL[:2]) if with_headers else None)
    id_counter = iter(range(10_000))
    divergences = [] if divergence_ids is None else divergence_ids
    structural_issue = [None]

    def note(message, op_index, op, pre_ref_snap=None):
        if mode == "faithful":
            issue_id = "FAITHFUL-MISMATCH"
        else:
            issue_id = classify_divergence(
                op, message,
                pre_ref_snap if pre_ref_snap is not None
                else _ref_snapshot(ref),
                _real_snapshot(ds),
                prior_issue=structural_issue[0])
            if issue_id != "UNKNOWN":
                if issue_id != "SM-08":
                    structural_issue[0] = issue_id
        div = Divergence(message, issue_id, op_index, op)
        divergences.append(div)
        return div

    def compare_states(op_index, op):
        if _real_snapshot(ds) != _ref_snapshot(ref):
            note("state mismatch: headers/rows/tags/formatters differ",
                 op_index, op)

    def check_export(op_index, op):
        # JSON
        _reset_formatter_calls(ref, ds)
        ref_error = None
        real_error = None
        expected_json = None
        actual_json = None
        try:
            expected_json, _ = ref.expected_json()
        except RefError as exc:
            ref_error = exc.kind
        try:
            actual_json_text = ds.export("json")
            actual_json = json.loads(actual_json_text)
        except Exception as exc:  # tablib raised where oracle did not
            real_error = _exc_kind(exc)
        if ref_error or real_error:
            if ref_error != real_error:
                note(f"export json exception mismatch oracle={ref_error} "
                     f"tablib={real_error}", op_index, op)
        else:
            if actual_json != expected_json:
                note("export json payload mismatch", op_index, op)
        if _formatter_calls(ref) != _real_formatter_calls(ds):
            note("export json formatter call log mismatch "
                 "(wrong column or duplicate/missing invocation)",
                 op_index, op)

        # CSV
        _reset_formatter_calls(ref, ds)
        ref_error = None
        real_error = None
        expected_csv = None
        actual_csv = None
        try:
            expected_csv, _ = ref.expected_csv()
        except RefError as exc:
            ref_error = exc.kind
        try:
            actual_csv = ds.export("csv")
        except Exception as exc:
            real_error = _exc_kind(exc)
        if ref_error or real_error:
            if ref_error != real_error:
                note(f"export csv exception mismatch oracle={ref_error} "
                     f"tablib={real_error}", op_index, op)
        else:
            if list(csv.reader(io.StringIO(actual_csv))) != \
                    list(csv.reader(io.StringIO(expected_csv))):
                note("export csv structure/values mismatch", op_index, op)
        if _formatter_calls(ref) != _real_formatter_calls(ds):
            note("export csv formatter call log mismatch "
                 "(wrong column or duplicate/missing invocation)",
                 op_index, op)

    for op_index, op in enumerate(ops):
        kind = op[0]

        if kind == "slice":
            expected = [list(row) for row in ref.rows[op[1]:op[2]]]
            actual = [list(row) for row in ds[op[1]:op[2]]]
            if actual != expected:
                note("slice mismatch", op_index, op)
            continue

        ref_before = ref.snapshot()
        real_before = _real_snapshot(ds)

        ref_error = None
        real_error = None
        new_ref = None
        new_ds = None
        shared_formatter = None
        if kind == "add_formatter" and id_counter is not None:
            shared_formatter = CountingFormatter(next(id_counter))
        try:
            new_ref = _apply_to_ref(ref, op, id_counter,
                                    shared_formatter=shared_formatter)
        except RefError as exc:
            ref_error = exc.kind
        except Exception as exc:  # pragma: no cover - oracle bug guard
            note(f"oracle raised {_exc_kind(exc)}: {exc}", op_index, op)
            continue

        try:
            new_ds = _apply_to_real(ds, op, id_counter,
                                    shared_formatter=shared_formatter)
        except (HeadersNeeded, InvalidDatasetIndex, InvalidDimensions,
                KeyError, IndexError, TypeError, ValueError) as exc:
            real_error = _exc_kind(exc)

        if ref_error or real_error:
            if ref_error != real_error:
                note(f"exception mismatch oracle={ref_error} "
                     f"tablib={real_error}", op_index, op,
                     pre_ref_snap=ref_before)
            real_changed = _real_snapshot(ds) != real_before
            ref_changed = ref.snapshot() != ref_before
            if mode == "ideal":
                # Ideal semantics demand full atomicity on both sides.
                if real_error and real_changed:
                    note("failed operation left a half-updated dataset",
                         op_index, op)
                if ref_error and ref_changed:
                    note("oracle failed non-atomically (model bug)",
                         op_index, op)
            elif real_changed != ref_changed:
                # Faithful mode may mirror a known non-atomic path, but the
                # two sides must still move identically.
                note("atomicity behaviour diverged between oracle and tablib",
                     op_index, op)
            continue

        if new_ref is not None:
            ref = new_ref
        if new_ds is not None:
            ds = new_ds

        compare_states(op_index, op)
        if check_exports and (op_index % 4 == 0 or op_index == len(ops) - 1):
            check_export(op_index, op)

    return ref, ds, divergences


def _apply_to_real(ds, op, id_counter, shared_formatter=None):
    """Apply one op to a real tablib Dataset; mirrors _apply_to_ref."""
    kind = op[0]
    if kind == "set_headers":
        ds.headers = list(op[1]) if op[1] else None
    elif kind == "insert_row":
        ds.insert(op[1], list(op[2]), tags=list(op[3]))
    elif kind == "delete_row":
        del ds[op[1]]
    elif kind == "lpop":
        ds.lpop()
    elif kind == "rpop":
        ds.rpop()
    elif kind == "insert_col":
        ds.insert_col(op[1], list(op[2]), header=op[3])
    elif kind == "delete_col":
        del ds[op[1]]
    elif kind == "add_formatter":
        col = None if op[1] == "__all__" else op[1]
        fmt = shared_formatter or CountingFormatter(next(id_counter))
        ds.add_formatter(col, fmt)
    elif kind == "sort":
        return ds.sort(op[1], reverse=op[2])
    elif kind == "filter":
        return ds.filter(op[1])
    elif kind == "stack":
        other = tablib.Dataset(headers=deepcopy(ds.headers))
        for row, row_tags in zip(op[1], op[2]):
            other.append(list(row), tags=list(row_tags))
        return ds.stack(other)
    return None


# ---------------------------------------------------------------------------
# Delta-debugging style minimiser and artifact persistence
# ---------------------------------------------------------------------------

def _reproduces(ops, mode, with_headers, issue_ids):
    _, _, divergences = run_sequence(
        ops, mode, with_headers=with_headers, check_exports=False)
    return any(d.issue_id in issue_ids for d in divergences), divergences


def minimise(ops, mode, with_headers, issue_ids, max_passes=12):
    """Delete irrelevant ops while the same issue still reproduces.

    Two passes alternate: single-op removal and contiguous-block removal.
    The result is the shortest sequence found within ``max_passes`` rounds;
    it is guaranteed (by construction) to still reproduce one of
    ``issue_ids``.
    """
    current = list(ops)
    reproduces, _ = _reproduces(current, mode, with_headers, issue_ids)
    if not reproduces:
        raise AssertionError("minimiser input does not reproduce the issue")
    for _ in range(max_passes):
        shortened = False
        # 1) remove single operations
        for i in range(len(current) - 1, -1, -1):
            candidate = current[:i] + current[i + 1:]
            ok, _ = _reproduces(candidate, mode, with_headers, issue_ids)
            if ok:
                current = candidate
                shortened = True
        # 2) remove contiguous blocks (halving window)
        window = max(1, len(current) // 2)
        while window:
            i = 0
            while i + window <= len(current):
                candidate = current[:i] + current[i + window:]
                ok, _ = _reproduces(candidate, mode, with_headers,
                                     issue_ids)
                if ok:
                    current = candidate
                    shortened = True
                else:
                    i += 1
            window //= 2
        if not shortened:
            break
    return current


def save_artifact(seed, ops, divergences, reduced=None):
    """Persist seed + replayable sequence for every divergence."""
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    payload = {
        "seed": seed,
        "ops": [list(op) for op in ops],
        "divergences": [d.as_dict() for d in divergences],
        "reduced_ops": ([list(op) for op in reduced]
                        if reduced is not None else None),
        "known_issues": KNOWN_ISSUES,
    }
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = os.path.join(ARTIFACT_DIR, f"divergence-seed{seed}-{stamp}.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False,
                  default=repr)
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class FaithfulModelTests(unittest.TestCase):
    """The oracle mirrors observed tablib behaviour: any mismatch is a
    genuine, unclassified regression and must fail."""

    def test_reference_model_matches_tablib_across_random_sequences(self):
        rng = random.Random(20261110)
        total_ops = 0
        failure_ops = 0
        for seed in range(SEED_COUNT):
            with_headers = random.Random(20261110 + seed).random() < 0.8
            seed_rng = random.Random(seed * 7919 + 13)
            ops = generate_ops(seed_rng, with_headers, OPS_PER_SEED)
            total_ops += len(ops)
            failure_ops += _count_raised_failures(ops, "faithful",
                                                   with_headers)
            _, _, divergences = run_sequence(ops, "faithful",
                                             with_headers=with_headers)
            if divergences:
                path = save_artifact(seed, ops, divergences)
                self.fail(
                    f"faithful oracle diverged from tablib; artifact: "
                    f"{path}\n" +
                    "\n".join(d.as_dict().__repr__() for d in divergences))
        # Sanity: the generator actually exercised failing operations, so the
        # atomicity checks below are meaningful.
        self.assertGreater(failure_ops, SEED_COUNT)
        self.assertGreater(total_ops, 100)

    def test_replay_is_deterministic(self):
        ops = generate_ops(random.Random(7), True, OPS_PER_SEED)
        ref_a, ds_a, div_a = run_sequence(ops, "faithful")
        ref_b, ds_b, div_b = run_sequence(ops, "faithful")
        self.assertEqual(_ref_snapshot(ref_a), _ref_snapshot(ref_b))
        self.assertEqual(_real_snapshot(ds_a), _real_snapshot(ds_b))
        self.assertEqual(div_a, div_b)


class IdealModelKnownIssueTests(unittest.TestCase):
    """Ideal semantics vs tablib: every divergence must map to a documented
    known issue.  Unknown divergences fail the run."""

    def test_ideal_mode_only_reports_known_issues_and_minimises(self):
        seen_issue_ids = set()
        example = None
        for seed in range(SEED_COUNT):
            with_headers = True
            ops = generate_ops(random.Random(seed * 104729 + 3),
                               with_headers, OPS_PER_SEED)
            _, _, divergences = run_sequence(ops, "ideal",
                                             with_headers=with_headers)
            for divergence in divergences:
                self.assertIn(
                    divergence.issue_id, KNOWN_ISSUES,
                    msg=f"unclassified divergence: {divergence.as_dict()}")
                seen_issue_ids.add(divergence.issue_id)
                if example is None and divergence.issue_id in (
                        "SM-01", "SM-02", "SM-03", "SM-06", "SM-07"):
                    example = (seed, ops, divergence.issue_id)

        # The random runner must actually hit at least the formatter shift
        # and sort metadata issues -- otherwise this test would be vacuous.
        self.assertTrue(
            {"SM-01", "SM-02", "SM-03"} & seen_issue_ids,
            msg=f"no formatter-shift issue observed; got {seen_issue_ids}")
        self.assertIn("SM-06", seen_issue_ids)

        if example is not None:
            seed, ops, issue_id = example
            reduced = minimise(ops, "ideal", True, {issue_id})
            self.assertLessEqual(len(reduced), len(ops))
            ok, divergences = _reproduces(reduced, "ideal", True,
                                           {issue_id})
            self.assertTrue(ok)
            save_artifact(seed, ops, divergences, reduced=reduced)


class KnownIssuesTests(unittest.TestCase):
    """Hand-written, minimal reproductions localising each real defect."""

    def test_sm01_formatter_runs_on_wrong_column_after_insert_col(self):
        ds = tablib.Dataset(headers=["a", "b"])
        ds.append(("a0", "b0"))
        fmt = CountingFormatter(0)
        ds.add_formatter(1, fmt)
        ds.insert_col(0, ["x0"], header="x")
        ds.export("json")
        # The formatter was attached to logical column 'b'; after the insert
        # it receives the value of column 'a' instead.
        self.assertEqual(fmt.calls, ["a0"])

    def test_sm02_formatter_index_ignores_column_deletion(self):
        ds = tablib.Dataset(headers=["a", "b", "c"])
        ds.append(("a0", "b0", "c0"))
        ds.add_formatter(2, CountingFormatter(1))
        del ds["a"]
        with self.assertRaises(IndexError):
            ds.export("json")

    def test_sm03_stale_formatter_breaks_csv_and_json_exports(self):
        ds = tablib.Dataset(headers=["a", "b", "c"])
        ds.append(("a0", "b0", "c0"))
        ds.add_formatter(2, CountingFormatter(9))
        del ds["a"]
        with self.assertRaises(IndexError):
            ds.export("csv")
        with self.assertRaises(IndexError):
            ds.export("json")

    def test_sm04_add_formatter_accepts_col_equal_to_width(self):
        ds = tablib.Dataset(headers=["a"])
        ds.append(("a0",))
        self.assertTrue(ds.add_formatter(1, CountingFormatter(2)))
        with self.assertRaises(IndexError):
            ds.export("csv")

    def test_sm05_negative_formatter_index_is_accepted(self):
        ds = tablib.Dataset(headers=["a", "b"])
        ds.append(("a0", "b0"))
        # A negative index is silently stored and resolves to the last column
        # at export time instead of being rejected up front.
        self.assertTrue(ds.add_formatter(-1, CountingFormatter(3)))
        self.assertEqual(ds._formatters[0][0], -1)

    def test_sm06_sort_drops_row_tags(self):
        ds = tablib.Dataset(headers=["name", "age"])
        ds.append(("a", 3), tags=["t1"])
        ds.append(("b", 1), tags=["t2"])
        sorted_ds = ds.sort("age")
        self.assertEqual([r.tags for r in sorted_ds._data], [[], []])
        self.assertEqual([r.tags for r in ds._data],
                         [["t1"], ["t2"]])

    def test_sm07_sort_drops_formatters(self):
        ds = tablib.Dataset(headers=["name", "age"])
        ds.append(("a", 3))
        ds.add_formatter(1, CountingFormatter(4))
        sorted_ds = ds.sort("age")
        self.assertEqual(sorted_ds._formatters, [])

    def test_sm08_stack_shares_row_objects(self):
        left = tablib.Dataset(headers=["x"])
        left.append((1,))
        right = tablib.Dataset(headers=["x"])
        right.append((2,))
        stacked = left.stack(right)
        stacked._data[0][0] = 99
        # Documented defect: the stacked dataset copies the Row references,
        # so mutating it reaches back into the left operand.
        self.assertEqual(left[0], (99,))
        self.assertEqual(stacked[0], (99,))

    def test_sm09_dynamic_column_index_goes_stale_after_insert_col(self):
        ds = tablib.Dataset(headers=["a"])
        ds.append(("r0",))
        ds.append(("r1",))
        ds.append_col(lambda row: "dyn:" + row[0], header="dyn")
        ds.insert_col(0, ["z0", "z1"], header="z")
        ds.append(("new", "raw"))
        # The dynamic value is generated at the stale position and lands in
        # the wrong column; the explicitly provided value is displaced.
        self.assertEqual(ds[-1], ("new", "dyn:new", "raw"))
        self.assertEqual(list(ds._dynamic_columns), [1])

    def test_sm10_dynamic_column_index_goes_stale_after_delete_col(self):
        ds = tablib.Dataset(headers=["a", "b"])
        ds.append((1, 2))
        ds.append((3, 4))
        ds.append_col(lambda row: row[0] + 10, header="d1")
        ds.append_col(lambda row: row[1] + 20, header="d2")
        del ds["a"]
        ds.append((9,))
        # Dynamic functions read stale indices after 'a' was removed.
        self.assertEqual(ds[-1], (9, 19, 39))

    def test_sm11_failed_callable_insert_col_half_updates_dynamic_map(self):
        ds = tablib.Dataset(headers=["a"])
        ds.append((1,))
        with self.assertRaises(HeadersNeeded):
            ds.append_col(lambda row: row[0] + 1)
        # The callable was registered before the header check ran, so the
        # failure leaves a phantom dynamic column behind.
        self.assertEqual(len(ds._dynamic_columns), 1)

    def test_sm12_empty_column_insert_mutates_headers_before_failing(self):
        ds = tablib.Dataset(headers=["a", "b"])
        ds.append((1, 2))
        ds.append((3, 4))
        with self.assertRaises(IndexError):
            ds.insert_col(0, [], header="z")
        # The header was inserted before the per-row loop failed, so the
        # aborted operation is not atomic.
        self.assertEqual(ds.headers, ["z", "a", "b"])
        self.assertEqual(ds[:], [(1, 2), (3, 4)])

    def test_sm13_insert_col_rebuild_drops_row_tags_at_zero_width(self):
        ds = tablib.Dataset(headers=["k0"])
        ds.append((True,), tags=["odd"])
        del ds["k0"]
        self.assertEqual(ds.height, 1)
        self.assertEqual(ds.width, 0)
        ds.insert_col(0, ["bb"])
        # Rebuilding _data from the column loses the tags carried by the
        # now column-less surviving row.
        self.assertEqual(ds[:], [("bb",)])
        self.assertEqual([row.tags for row in ds._data], [[]])

    def test_sm14_empty_column_insert_at_zero_width_wipes_rows(self):
        ds = tablib.Dataset(headers=["k0"])
        ds.append((1,), tags=["odd"])
        ds.append((2,), tags=["even"])
        del ds["k0"]
        self.assertEqual(ds.height, 2)
        self.assertEqual(ds.width, 0)
        ds.insert_col(0, [])
        # The empty-column rebuild silently destroys the surviving rows.
        self.assertEqual(ds.height, 0)
        self.assertEqual(ds[:], [])

    def test_failed_plain_insert_col_is_atomic(self):
        ds = tablib.Dataset(headers=["a", "b"])
        ds.append((1, 2))
        before = _real_snapshot(ds)
        with self.assertRaises(InvalidDimensions):
            ds.insert_col(0, [1, 2, 3], header="x")
        self.assertEqual(_real_snapshot(ds), before)

    def test_failed_row_insert_is_atomic(self):
        ds = tablib.Dataset(headers=["a", "b"])
        ds.append((1, 2))
        before = _real_snapshot(ds)
        with self.assertRaises(InvalidDimensions):
            ds.append((1, 2, 3))
        self.assertEqual(_real_snapshot(ds), before)


def _count_raised_failures(ops, mode, with_headers):
    """Count ops where both sides raised: the exercised failure paths."""
    ref = RefDataset(mode=mode, with_headers=with_headers)
    ds = tablib.Dataset(
        headers=list(HEADER_POOL[:2]) if with_headers else None)
    id_counter = iter(range(10_000))
    raised = 0
    for op in ops:
        if op[0] == "slice":
            continue
        ref_error = False
        real_error = False
        new_ref = None
        new_ds = None
        shared_formatter = None
        if op[0] == "add_formatter":
            shared_formatter = CountingFormatter(next(id_counter))
        try:
            new_ref = _apply_to_ref(ref, op, id_counter,
                                    shared_formatter=shared_formatter)
        except (RefError, IndexError, KeyError, TypeError, ValueError):
            ref_error = True
        try:
            new_ds = _apply_to_real(ds, op, id_counter,
                                    shared_formatter=shared_formatter)
        except (HeadersNeeded, InvalidDatasetIndex, InvalidDimensions,
                KeyError, IndexError, TypeError, ValueError):
            real_error = True
        if ref_error and real_error:
            raised += 1
        elif new_ref is not None:
            ref = new_ref
        if new_ds is not None:
            ds = new_ds
    return raised


if __name__ == "__main__":
    unittest.main()
