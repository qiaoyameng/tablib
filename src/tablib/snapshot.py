"""
    tablib.snapshot
    ~~~~~~~~~~~~~~~

    Read-only snapshots, block-shared views and copy-on-write branches.

    A :class:`Snapshot` captures the structural state of a :class:`Dataset`
    (headers, title, formatter bindings, separators, dynamic columns and row
    tags) together with its cell values.  Immutable scalar cells are stored in
    shared :class:`StorageBlock` objects so that overlapping views and branches
    reference the same underlying storage instead of deep-copying whole tables.

    Only values of known immutable scalar types are shared.  Mutable values are
    either deep-copied (``on_mutable='copy'``) or cause the snapshot to be
    rejected (``on_mutable='reject'``, the default); they are never aliased
    behind a claimed shallow-copy boundary.
"""

import datetime
import decimal
from copy import deepcopy

from .exceptions import (
    HeadersNeeded,
    SnapshotMutableCellError,
    SnapshotReadOnlyError,
    UnsupportedFormat,
)
from .core import Dataset, Row
from .formats import registry

__all__ = [
    'Snapshot',
    'DatasetView',
    'SnapshotDataset',
    'StorageBlock',
    'StorageStats',
    'is_immutable_scalar',
]

#: Number of rows grouped into a single storage block.
DEFAULT_BLOCK_SIZE = 1024

#: Built-in value types whose instances cannot be mutated in place.
_IMMUTABLE_SCALAR_TYPES = (
    str,
    bytes,
    int,
    float,
    bool,
    complex,
    tuple,
    frozenset,
    decimal.Decimal,
    datetime.datetime,
    datetime.date,
    datetime.time,
    datetime.timedelta,
    datetime.tzinfo,
)


def is_immutable_scalar(value):
    """Return ``True`` when *value* is a built-in immutable scalar.

    ``None`` is treated as immutable as well.  Mutable containers such as
    :class:`list` or :class:`dict` and arbitrary custom objects return
    ``False`` and are therefore copied or rejected when a snapshot is taken.
    """
    if value is None:
        return True
    return isinstance(value, _IMMUTABLE_SCALAR_TYPES)


class StorageStats:
    """Monotonic counters describing physical storage-block operations.

    The counters let tests assert on block identity and copy counts rather
    than trusting claimed sharing.
    """

    blocks_created = 0
    blocks_copied = 0

    @classmethod
    def reset(cls):
        cls.blocks_created = 0
        cls.blocks_copied = 0

    @classmethod
    def as_dict(cls):
        return {
            'blocks_created': cls.blocks_created,
            'blocks_copied': cls.blocks_copied,
        }


class StorageBlock:
    """An immutable container for a group of rows.

    Each stored row is a :class:`tuple` of cell values.  Blocks are shared by
    reference between snapshots, views and unmodified branch rows.
    """

    __slots__ = ('block_id', 'rows', '_frozen')

    def __init__(self, rows):
        StorageStats.blocks_created += 1
        self.block_id = StorageStats.blocks_created
        # Defensive copy at construction time; afterwards the block is frozen.
        self.rows = tuple(tuple(row) for row in rows)
        object.__setattr__(self, '_frozen', True)

    def __setattr__(self, name, value):
        if getattr(self, '_frozen', False):
            raise AttributeError(
                f'StorageBlock is immutable; cannot set {name!r}.'
            )
        super().__setattr__(name, value)

    def __delattr__(self, name):
        if getattr(self, '_frozen', False):
            raise AttributeError(
                f'StorageBlock is immutable; cannot delete {name!r}.'
            )
        super().__delattr__(name)

    def __len__(self):
        return len(self.rows)

    def __iter__(self):
        return iter(self.rows)

    def __repr__(self):
        return f'<StorageBlock id={self.block_id} rows={len(self.rows)}>'


def _prepare_cell(value, on_mutable):
    """Return a value safe to place in shared storage."""
    if is_immutable_scalar(value):
        return value
    if on_mutable == 'reject':
        raise SnapshotMutableCellError(
            f"Mutable cell of type {type(value).__name__!r} cannot be shared; "
            "pass on_mutable='copy' to deep-copy it instead."
        )
    if on_mutable != 'copy':
        raise ValueError(
            f"on_mutable must be 'reject' or 'copy', got {on_mutable!r}."
        )
    return deepcopy(value)


class _ReadOnlyBase:
    """Shared read-only surface for snapshots and views."""

    is_read_only = True

    # ---- export machinery -------------------------------------------------

    def export(self, format, **kwargs):
        """Export the snapshot structure using a registered format."""
        fmt = registry.get_format(format)
        if not hasattr(fmt, 'export_set'):
            raise UnsupportedFormat(f'Format {format} cannot be exported.')
        return fmt.export_set(self, **kwargs)

    def __getattr__(self, name):
        # Format shorthand, e.g. ``snapshot.csv`` / ``view.json``.
        try:
            fmt = registry.get_format(name)
        except UnsupportedFormat as err:
            raise AttributeError(name) from err
        if not hasattr(fmt, 'export_set'):
            raise AttributeError(name)
        return fmt.export_set(self)

    # ---- frozen attribute handling ---------------------------------------

    def _freeze(self):
        object.__setattr__(self, '_frozen', True)

    def __setattr__(self, name, value):
        if getattr(self, '_frozen', False):
            raise SnapshotReadOnlyError(
                f'{type(self).__name__} is read-only; create an editable branch instead.'
            )
        object.__setattr__(self, name, value)

    def __delattr__(self, name):
        if getattr(self, '_frozen', False):
            raise SnapshotReadOnlyError(
                f'{type(self).__name__} is read-only; create an editable branch instead.'
            )
        object.__delattr__(self, name)

    # ---- rejected mutating Dataset API -----------------------------------

    def _reject(self, *args, **kwargs):
        raise SnapshotReadOnlyError(
            f'{type(self).__name__} is read-only; create an editable branch instead.'
        )

    insert = rpush = lpush = append = extend = _reject
    lpop = rpop = pop = _reject
    insert_col = rpush_col = lpush_col = append_col = _reject
    insert_separator = append_separator = _reject
    add_formatter = remove_duplicates = wipe = load = _reject
    sort = transpose = stack = stack_cols = filter = subset = _reject

    def __setitem__(self, key, value):
        self._reject()

    def __delitem__(self, key):
        self._reject()

    # ---- shared read helpers ---------------------------------------------

    def _package(self, dicts=True):
        """Package projected rows, applying bound formatters."""
        rows = list(self._iter_rows())

        def format_row(row):
            if self._formatters:
                row = list(row)
                for col, callback in self._formatters:
                    if col is None:
                        row = [callback(cell) for cell in row]
                    else:
                        row[col] = callback(row[col])
            return list(row)

        if self.headers:
            if dicts:
                return [dict(zip(self.headers, format_row(row))) for row in rows]
            return [list(self.headers)] + [format_row(row) for row in rows]
        return [format_row(row) for row in rows]

    @property
    def dict(self):
        """Dictionary (or tuple-row) representation, matching :class:`Dataset`."""
        return self._package()

    def __len__(self):
        return self.height

    def __iter__(self):
        return self._iter_rows()

    def __contains__(self, item):
        return any(tuple(row) == tuple(item) for row in self._iter_rows())

    def get(self, index):
        """Return the projected row tuple at *index*."""
        if not isinstance(index, int):
            raise TypeError('Row indices must be integers.')
        return self[index]

    def get_col(self, index):
        """Return a column by positional index or header name."""
        if isinstance(index, str):
            if self.headers is None:
                raise HeadersNeeded()
            if index not in self.headers:
                raise KeyError(index)
            index = list(self.headers).index(index)
        return [row[index] for row in self._iter_rows()]


class Snapshot(_ReadOnlyBase):
    """Immutable, read-only capture of a :class:`~tablib.Dataset`.

    Use :meth:`from_dataset` to create one.  The snapshot owns a set of
    :class:`StorageBlock` instances; rows and immutable cells are referenced,
    not copied, by views and unchanged branch rows.
    """

    def __init__(self):
        self._blocks = {}
        self._row_locations = []
        self._row_tags = []
        self._headers = None
        self._title = None
        self._width = 0
        self._formatters = ()
        self._separators = ()
        self._dynamic_columns = {}
        self._frozen = False

    @classmethod
    def from_dataset(cls, dataset, on_mutable='reject', block_size=DEFAULT_BLOCK_SIZE):
        """Capture *dataset* without altering it.

        :param on_mutable: ``'reject'`` (default) raises
            :class:`SnapshotMutableCellError` on mutable cells; ``'copy'``
            deep-copies them so the snapshot stays isolated.
        :param block_size: number of rows packed per storage block.
        """
        if block_size < 1:
            raise ValueError('block_size must be a positive integer.')

        snapshot = cls()
        snapshot._title = dataset.title
        snapshot._headers = tuple(dataset.headers) if dataset.headers else None
        snapshot._formatters = tuple(dataset._formatters)
        snapshot._separators = tuple(dataset._separators)
        snapshot._dynamic_columns = dict(dataset._dynamic_columns)
        snapshot._width = dataset.width

        pending = []

        def flush():
            if not pending:
                return
            block = StorageBlock(pending)
            snapshot._blocks[block.block_id] = block
            for offset in range(len(block.rows)):
                snapshot._row_locations.append((block.block_id, offset))
            pending.clear()

        for row in dataset._data:
            prepared = tuple(_prepare_cell(cell, on_mutable) for cell in row)
            snapshot._row_tags.append(tuple(row.tags))
            pending.append(prepared)
            if len(pending) >= block_size:
                flush()
        flush()

        snapshot._freeze()
        return snapshot

    def __repr__(self):
        if not self._title:
            return '<snapshot object>'
        return f'<snapshot {self._title.lower()}>'

    # ---- structure --------------------------------------------------------

    @property
    def headers(self):
        return list(self._headers) if self._headers is not None else None

    @property
    def title(self):
        return self._title

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return len(self._row_locations)

    @property
    def storage_stats(self):
        return StorageStats.as_dict()

    @property
    def blocks(self):
        """Tuple of storage-block ids backing this snapshot."""
        return tuple(self._blocks)

    def block(self, block_id):
        return self._blocks[block_id]

    def block_for_row(self, index):
        """Return the storage-block id backing snapshot row *index*."""
        return self._row_locations[index][0]

    def row_tags(self, index):
        """Return the tags captured for snapshot row *index*."""
        return list(self._row_tags[index])

    # ---- row access -------------------------------------------------------

    def _row_at(self, index):
        block_id, offset = self._row_locations[index]
        return self._blocks[block_id].rows[offset]

    def _iter_rows(self):
        for block_id, offset in self._row_locations:
            yield self._blocks[block_id].rows[offset]

    @property
    def _data(self):
        """Row objects for format compatibility (e.g. the SQL exporter)."""
        from .core import Row
        return [Row(row, tags=self._row_tags[i]) for i, row in enumerate(self._iter_rows())]

    def __getitem__(self, key):
        if isinstance(key, str):
            if self.headers is None:
                raise HeadersNeeded()
            if key not in self._headers:
                raise KeyError(key)
            pos = self._headers.index(key)
            return [row[pos] for row in self._iter_rows()]
        if isinstance(key, slice):
            return [self._row_at(i) for i in range(*key.indices(self.height))]
        return self._row_at(key)

    # ---- views and branches ----------------------------------------------

    def view(self, rows=None, cols=None, tags=None):
        """Return a block-sharing :class:`DatasetView` over this snapshot."""
        return DatasetView(self, rows=rows, cols=cols, tags=tags)

    def to_branch(self, on_mutable='copy'):
        """Materialize an editable, copy-on-write :class:`SnapshotDataset`."""
        return SnapshotDataset.from_snapshot(self, on_mutable=on_mutable)


class DatasetView(_ReadOnlyBase):
    """A read-only row/column projection over a :class:`Snapshot`.

    Views hold no cell values of their own: every projected row reads through
    to the snapshot's storage blocks, so overlapping views share storage.
    """

    def __init__(self, snapshot, rows=None, cols=None, tags=None):
        self._snapshot = snapshot

        if rows is None:
            row_indices = list(range(snapshot.height))
        else:
            row_indices = []
            for row in rows:
                if not isinstance(row, int) or not (-snapshot.height <= row < snapshot.height):
                    raise IndexError(f'Row index out of snapshot bounds: {row!r}')
                row_indices.append(row % snapshot.height)

        if tags is not None:
            wanted = {tags} if isinstance(tags, str) else set(tags)
            row_indices = [
                row for row in row_indices
                if wanted & set(snapshot._row_tags[row])
            ]
        self._row_indices = tuple(row_indices)

        if cols is None:
            col_positions = list(range(snapshot.width))
        else:
            col_positions = []
            for col in cols:
                if isinstance(col, str):
                    if snapshot._headers is None:
                        raise HeadersNeeded()
                    if col not in snapshot._headers:
                        raise KeyError(col)
                    col = snapshot._headers.index(col)
                if not isinstance(col, int) or not (-snapshot.width <= col < snapshot.width):
                    raise IndexError(f'Column index out of snapshot bounds: {col!r}')
                col_positions.append(col % snapshot.width)
        self._col_positions = tuple(col_positions)

        if snapshot._headers is not None:
            self._headers = tuple(snapshot._headers[pos] for pos in self._col_positions)
        else:
            self._headers = None

        # Re-map formatter column indices onto the projected columns.
        self._formatters = tuple(
            (None if col is None else self._col_positions.index(col), callback)
            for col, callback in snapshot._formatters
            if col is None or col in self._col_positions
        )
        self._separators = self._remap_separators(snapshot._separators)
        self._frozen = False
        self._freeze()

    def _remap_separators(self, separators):
        remapped = []
        for original_index, text in separators:
            new_index = sum(
                1 for source_row in self._row_indices if source_row < original_index
            )
            remapped.append((new_index, text))
        return tuple(remapped)

    def __repr__(self):
        return f'<DatasetView rows={self.height} cols={self.width}>'

    # ---- structure --------------------------------------------------------

    @property
    def snapshot(self):
        return self._snapshot

    @property
    def headers(self):
        return list(self._headers) if self._headers is not None else None

    @property
    def title(self):
        return self._snapshot.title

    @property
    def width(self):
        return len(self._col_positions)

    @property
    def height(self):
        return len(self._row_indices)

    @property
    def storage_stats(self):
        return StorageStats.as_dict()

    @property
    def blocks(self):
        """Storage-block ids referenced by this view, in row order."""
        return tuple(
            self._snapshot._row_locations[source_row][0]
            for source_row in self._row_indices
        )

    def block_for_row(self, index):
        source_row = self._row_indices[index]
        return self._snapshot._row_locations[source_row][0]

    def row_tags(self, index):
        return self._snapshot.row_tags(self._row_indices[index])

    # ---- row access -------------------------------------------------------

    def _projected_row(self, source_row):
        block_id, offset = self._snapshot._row_locations[source_row]
        source = self._snapshot._blocks[block_id].rows[offset]
        return tuple(source[pos] for pos in self._col_positions)

    def _iter_rows(self):
        for source_row in self._row_indices:
            yield self._projected_row(source_row)

    @property
    def _data(self):
        from .core import Row
        return [
            Row(self._projected_row(source_row), tags=self._snapshot._row_tags[source_row])
            for source_row in self._row_indices
        ]

    def __getitem__(self, key):
        if isinstance(key, str):
            if self._headers is None:
                raise HeadersNeeded()
            if key not in self._headers:
                raise KeyError(key)
            pos = self._headers.index(key)
            return [row[pos] for row in self._iter_rows()]
        if isinstance(key, slice):
            return [
                self._projected_row(source_row)
                for source_row in self._row_indices[key]
            ]
        return self._projected_row(self._row_indices[key])

    # ---- narrowing and branching -----------------------------------------

    def view(self, rows=None, cols=None, tags=None):
        """Return a narrower view; indices are relative to this view."""
        if rows is None:
            source_rows = self._row_indices
        else:
            source_rows = tuple(self._row_indices[row] for row in rows)

        if cols is None:
            source_cols = self._col_positions
        else:
            source_cols = tuple(self._col_positions[col] for col in cols)

        nested = DatasetView.__new__(DatasetView)
        nested._snapshot = self._snapshot
        nested._row_indices = tuple(source_rows)
        nested._col_positions = tuple(source_cols)
        if self._snapshot._headers is not None:
            nested._headers = tuple(
                self._snapshot._headers[pos] for pos in source_cols
            )
        else:
            nested._headers = None
        nested._formatters = tuple(
            (None if col is None else nested._col_positions.index(col), callback)
            for col, callback in self._snapshot._formatters
            if col is None or col in nested._col_positions
        )
        if tags is not None:
            wanted = {tags} if isinstance(tags, str) else set(tags)
            keep = tuple(
                row for row in nested._row_indices
                if wanted & set(self._snapshot._row_tags[row])
            )
            nested._row_indices = keep
        nested._separators = nested._remap_separators(self._snapshot._separators)
        nested._frozen = False
        nested._freeze()
        return nested

    def to_branch(self, on_mutable='copy'):
        """Materialize this view as an editable :class:`SnapshotDataset`."""
        return SnapshotDataset.from_snapshot(self._snapshot, view=self, on_mutable=on_mutable)


class SnapshotDataset(Dataset):
    """Editable copy-on-write branch of a :class:`Snapshot` or :class:`DatasetView`.

    It is a real :class:`~tablib.Dataset` subclass, so the ordinary Dataset
    API behaves exactly as usual.  Rows initially reference the snapshot's
    immutable storage blocks; a block is copied at most once, on the first
    mutation that touches one of its rows.  Mutating the branch never touches
    the source snapshot or dataset.
    """

    is_read_only = False

    def __init__(self):
        super().__init__()
        # Maps each local row index to the id of its backing storage block,
        # or ``None`` when the row is private (inserted or detached).
        self._row_blocks = []
        self._snapshot = None
        self._copied_blocks = set()

    @classmethod
    def from_snapshot(cls, source, view=None, on_mutable='copy'):
        if view is None:
            view = source.view() if isinstance(source, Snapshot) else source
        snapshot = view._snapshot

        branch = cls()
        branch._snapshot = snapshot
        branch.title = snapshot.title
        branch.headers = view.headers
        branch._formatters = list(view._formatters)
        branch._separators = list(view._separators)
        branch._dynamic_columns = dict(snapshot._dynamic_columns)

        for local_index, source_row in enumerate(view._row_indices):
            block_id, offset = snapshot._row_locations[source_row]
            block = snapshot._blocks[block_id]
            values = tuple(
                _prepare_cell(block.rows[offset][pos], on_mutable)
                for pos in view._col_positions
            )
            branch._data.append(Row(values, tags=snapshot._row_tags[source_row]))
            branch._row_blocks.append(block_id)

        return branch

    # ---- storage introspection -------------------------------------------

    @property
    def storage_stats(self):
        return StorageStats.as_dict()

    @property
    def blocks(self):
        """Backing block id per local row (``None`` for private rows)."""
        return tuple(self._row_blocks)

    def block_for_row(self, index):
        return self._row_blocks[index]

    def shares_block_with(self, snapshot, index=None):
        """Return True when branch rows still point at snapshot blocks."""
        snapshot_blocks = set(snapshot._blocks)
        if index is not None:
            return self._row_blocks[index] in snapshot_blocks
        return any(block_id in snapshot_blocks for block_id in self._row_blocks)

    @property
    def shared_row_count(self):
        snapshot_blocks = set(self._snapshot._blocks) if self._snapshot else set()
        return sum(
            1 for block_id in self._row_blocks if block_id in snapshot_blocks
        )

    # ---- copy-on-write primitives ----------------------------------------

    def _detach_row_block(self, index):
        """Ensure the block backing row *index* is private before mutation."""
        block_id = self._row_blocks[index]
        if block_id is None or block_id in self._copied_blocks:
            return

        StorageStats.blocks_copied += 1
        self._copied_blocks.add(block_id)
        for local_index, current_block in enumerate(self._row_blocks):
            if current_block == block_id:
                # A shallow row copy is sufficient: immutable cells are shared
                # while the Row container becomes private.
                self._data[local_index] = self._data[local_index].copy()
                self._row_blocks[local_index] = None

    def _detach_all(self):
        for index in range(len(self._row_blocks)):
            self._detach_row_block(index)

    # ---- mutating Dataset API overridden for COW -------------------------

    def __setitem__(self, key, value):
        if isinstance(key, tuple) and len(key) == 2:
            row_index, col_index = key
            if row_index < 0 or row_index >= len(self._data):
                raise IndexError(row_index)
            self._detach_row_block(row_index)
            self._data[row_index][col_index] = value
            return
        if isinstance(key, int):
            self._detach_row_block(key)
        super(SnapshotDataset, self).__setitem__(key, value)
        self._rebuild_row_blocks_after_replace(key)

    def _rebuild_row_blocks_after_replace(self, key):
        if isinstance(key, int):
            self._row_blocks[key] = None
        else:
            for index in range(*key.indices(len(self._data))):
                self._row_blocks[index] = None

    def __delitem__(self, key):
        if isinstance(key, str):
            self._detach_all()
            super(SnapshotDataset, self).__delitem__(key)
            return
        if isinstance(key, int):
            self._detach_row_block(key)
        super(SnapshotDataset, self).__delitem__(key)
        del self._row_blocks[key]

    def insert(self, index, row, tags=()):
        # A middle insertion would shift every later row out of its block
        # mapping; detach first so block identity tracking stays accurate.
        # Appending at the end keeps existing rows aligned with their blocks.
        normalized = index + len(self._data) + 1 if index < 0 else index
        if normalized < len(self._data):
            self._detach_all()
        super().insert(index, row, tags=tags)
        self._row_blocks.insert(index, None)

    def insert_col(self, index, col=None, header=None):
        # Adding a column changes every row, so every shared block must split.
        self._detach_all()
        super().insert_col(index, col=col, header=header)

    def remove_duplicates(self):
        kept = []
        seen = set()
        for index, row in enumerate(self._data):
            if tuple(row) in seen:
                continue
            seen.add(tuple(row))
            kept.append(index)
        if len(kept) == len(self._data):
            return
        self._data[:] = [self._data[index] for index in kept]
        self._row_blocks[:] = [self._row_blocks[index] for index in kept]

    def wipe(self):
        super().wipe()
        self._row_blocks = []
        self._copied_blocks = set()
        self._snapshot = None

    def set_row_tags(self, index, tags):
        """Replace the tags of row *index* (tags are branch-local COW state)."""
        self._detach_row_block(index)
        self._data[index].tags = list(tags)
        self._row_blocks[index] = None
