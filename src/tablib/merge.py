"""
    tablib.merge
    ~~~~~~~~~~~~

    Column-identity reconciliation for :class:`~tablib.core.Dataset` objects.

    :meth:`Dataset.stack` joins rows by *position*: it only checks that the
    widths agree. Departments frequently hand in the same columns in a
    different order, or with an extra column, so positional stacking is not
    enough. The functions here join rows by *column identity* (the header),
    support intersecting or unioning the column sets, and always produce a
    reviewable :class:`MergePlan` before any result is built.

    Typical use::

        plan = tablib.plan_merge(hq, branch_a, branch_b, how='union')
        if plan.rejected:
            ...inspect plan.rejected and adjust the policy...
        merged = plan.apply()

    :copyright: (c) 2026 Jazzband.
    :license: MIT, see LICENSE for more details.
"""

__lazy_modules__ = {
    "copy",
    "datetime",
    "decimal",
    "tablib.exceptions",
}

import copy
import datetime as _dt
from decimal import Decimal

from .exceptions import (
    AmbiguousColumns,
    HeadersRequired,
    InvalidIdentityMap,
    MergeRejected,
    PlanStale,
)

__all__ = [
    "MergePlan",
    "ReconcilePolicy",
    "merge_sets",
    "plan_merge",
]


BOOL = "bool"
INT = "int"
FLOAT = "float"
DECIMAL = "decimal"
DATE = "date"
DATETIME = "datetime"
TIME = "time"
TEXT = "text"
EMPTY = "empty"
CUSTOM = "custom"


def value_kind(value):
    """Classify a non-``None`` cell value into a supported type family."""

    if isinstance(value, bool):  # bool before int: bool subclasses int
        return BOOL
    if isinstance(value, int):
        return INT
    if isinstance(value, Decimal):
        return DECIMAL
    if isinstance(value, float):
        return FLOAT
    if isinstance(value, _dt.datetime):
        return DATETIME
    if isinstance(value, _dt.date):
        return DATE
    if isinstance(value, _dt.time):
        return TIME
    if isinstance(value, str):
        return TEXT
    return type(value).__name__


def _to_float(value):
    if isinstance(value, Decimal):
        return float(value)
    return float(value)


def _to_datetime(value):
    return _dt.datetime(value.year, value.month, value.day)


def _to_text(value):
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value)


_CONVERTERS = {
    BOOL: bool,
    INT: int,
    FLOAT: _to_float,
    DECIMAL: Decimal,
    DATE: lambda value: value.date(),
    DATETIME: _to_datetime,
    TIME: lambda value: value.replace(year=1900, month=1, day=1).time(),
    TEXT: _to_text,
}


def reconcile_types(kinds, allow_lossy=False):
    """Reconcile a set of observed type families into one target family.

    Returns ``(target_kind, lossy, reason)``. On success ``reason`` is
    ``None``; ``target_kind`` is ``None`` when the families cannot be
    reconciled, in which case ``reason`` explains why.

    With ``allow_lossy`` at its default of ``False`` only lossless
    combinations succeed. ``bool``/``int`` merge losslessly to ``int``;
    mixing :class:`~decimal.Decimal` with :class:`float` (and other numeric
    promotions), :class:`~datetime.date` with
    :class:`~datetime.datetime`, or anything with text, is lossy and is left
    for the caller to decide.
    """

    kinds = {kind for kind in kinds if kind != EMPTY}

    if not kinds:
        return EMPTY, False, None

    if len(kinds) == 1:
        return next(iter(kinds)), False, None

    numeric = {BOOL, INT, FLOAT, DECIMAL}
    if kinds <= numeric:
        if FLOAT not in kinds:
            # bool -> int -> Decimal are exact, lossless promotions.
            if DECIMAL in kinds:
                return DECIMAL, False, None
            return INT, False, None
        # Any float mixed with bool/int/Decimal rounds the others to binary
        # floating point, which is lossy (Decimal + float in particular).
        if allow_lossy:
            return FLOAT, True, None
        return None, False, (
            "lossy numeric combination {}; set allow_lossy or provide a "
            "column converter".format(sorted(kinds))
        )

    if kinds == {DATE, DATETIME}:
        if allow_lossy:
            return DATETIME, True, None
        return None, False, (
            "lossy date/datetime combination; set allow_lossy or provide a "
            "column converter"
        )

    if TEXT in kinds:
        if allow_lossy:
            return TEXT, True, None
        return None, False, (
            "lossy combination with text {}; set allow_lossy or provide a "
            "column converter".format(sorted(kinds))
        )

    return None, False, "incompatible type families {}".format(sorted(kinds))


class ReconcilePolicy:
    """Rules describing how several :class:`~tablib.core.Dataset` instances
    should be joined by column identity.

    :param how: ``'intersection'`` keeps columns shared by every dataset;
        ``'union'`` keeps every column seen anywhere.
    :param fill: value used for rows coming from a dataset that lacks the
        column. May be a plain value or a zero-argument callable. Defaults to
        ``None``. This fill is distinct from a cell that already contained
        ``None`` (which is preserved as ``None``).
    :param allow_lossy: permit lossy type combinations such as
        :class:`~decimal.Decimal` with :class:`float` (promoted to
        :class:`float`), :class:`~datetime.date` with
        :class:`~datetime.datetime`, or anything with text. Defaults to
        ``False``: only lossless conversions are performed automatically.
    :param converters: mapping of ``{column_identity: callable}``. A provided
        converter overrides automatic reconciliation for that column and is
        applied to every non-``None`` cell.
    """

    def __init__(self, how="intersection", fill=None, allow_lossy=False,
                 converters=None):
        if how not in ("intersection", "union"):
            raise ValueError("how must be 'intersection' or 'union'")
        self.how = how
        self.fill = fill
        self.allow_lossy = allow_lossy
        self.converters = dict(converters or {})


class ColumnPlan:
    """Reconciliation details for a single result column.

    Attributes:
        identity: the column identity (usually the header string).
        result_index: position of the column in the result.
        sources: ordered list of ``(dataset_index, source_position)``.
        target_type: reconciled type family (or ``'custom'``/``'empty'``),
            ``None`` when rejected.
        observed_types: mapping of ``dataset_index`` to a frozenset of the
            type families observed in that source column.
        lossy: ``True`` when automatic reconciliation is lossy.
        fill: value used for datasets missing the column.
        fill_is_callable: whether ``fill`` is invoked per cell.
        converter: explicit converter overriding automatic reconciliation.
        reason: rejection reason, or ``None``.
    """

    __slots__ = (
        "identity", "result_index", "sources", "target_type",
        "observed_types", "lossy", "fill", "fill_is_callable",
        "converter", "reason",
    )

    def __init__(self, identity, result_index, sources, target_type,
                 observed_types, lossy, fill, converter, reason):
        self.identity = identity
        self.result_index = result_index
        self.sources = sources
        self.target_type = target_type
        self.observed_types = observed_types
        self.lossy = lossy
        self.fill = fill
        self.fill_is_callable = callable(fill)
        self.converter = converter
        self.reason = reason

    @property
    def rejected(self):
        """``True`` when this column could not be reconciled."""
        return self.reason is not None

    def describe(self):
        """Return a human-readable description of this column plan."""

        present = [str(index) for index, _ in self.sources]
        if self.rejected:
            status = "REJECTED ({})".format(self.reason)
        else:
            types = sorted({
                kind
                for group in self.observed_types.values()
                for kind in group
                if kind != EMPTY
            })
            status = "-> {}".format(self.target_type)
            if types and types != [self.target_type]:
                status += " from {}".format(types)
            if self.lossy:
                status += " [lossy]"
        fill_note = ("fill=<callable>" if self.fill_is_callable
                     else "fill={!r}".format(self.fill))
        return (
            "  [{index}] {identity}: datasets [{sources}] {status}; {fill}"
            .format(
                index=self.result_index,
                identity=self.identity,
                sources=", ".join(present),
                status=status,
                fill=fill_note,
            )
        )


class MergePlan:
    """A reviewable reconciliation plan produced by :func:`plan_merge`.

    Inspect :attr:`headers` (result column order), :attr:`columns`
    (per-column conversions and fills) and :attr:`rejected` before calling
    :meth:`apply`. Applying a plan with rejected columns raises
    :class:`~tablib.exceptions.MergeRejected`.
    """

    def __init__(self, datasets, identities, how, column_plans,
                 result_headers, fingerprints, notes=None):
        self._datasets = list(datasets)
        self.identities = [list(ids) for ids in identities]
        self.how = how
        self.columns = list(column_plans)
        self.headers = list(result_headers)
        self._fingerprints = list(fingerprints)
        self.notes = list(notes or [])

    @property
    def rejected(self):
        """List of ``(identity, reason)`` pairs that block application."""

        return [
            (column.identity, column.reason)
            for column in self.columns
            if column.rejected
        ]

    @property
    def lossy(self):
        """Identities reconciled with a lossy conversion."""

        return [column.identity for column in self.columns if column.lossy]

    def is_ready(self):
        """``True`` when the plan can be applied without rejection."""

        return not self.rejected

    def describe(self):
        """Return a readable summary of the full plan."""

        lines = [
            "merge plan ({})".format(self.how),
            "  result columns: {}".format(self.headers),
        ]
        lines.extend(column.describe() for column in self.columns)
        for note in self.notes:
            lines.append("  note: {}".format(note))
        if self.rejected:
            lines.append("  BLOCKED: {}".format(self.rejected))
        else:
            lines.append("  ready to apply")
        return "\n".join(lines)

    def __repr__(self):
        state = "ready" if self.is_ready() else (
            "{} rejected".format(len(self.rejected))
        )
        return "<MergePlan columns={} {}>".format(len(self.columns), state)

    def _verify_structure(self):
        current = [_fingerprint(dataset) for dataset in self._datasets]
        if current != self._fingerprints:
            raise PlanStale(
                "input Dataset structure changed after the plan was built"
            )

    def apply(self):
        """Build and return the merged :class:`~tablib.core.Dataset`.

        Raises :class:`~tablib.exceptions.PlanStale` if any input dataset's
        structure changed after planning, and
        :class:`~tablib.exceptions.MergeRejected` if the plan has rejected
        columns. Row tags are preserved, and cell values are deep-copied so
        later edits cannot leak between the inputs and the result.
        """

        from .core import Dataset

        self._verify_structure()
        if self.rejected:
            raise MergeRejected(
                "cannot apply plan with rejected columns: {}".format(
                    self.rejected
                )
            )

        result = Dataset(headers=list(self.headers))

        for dataset_index, dataset in enumerate(self._datasets):
            positions_by_column = []
            for column in self.columns:
                position = None
                for source_index, source_position in column.sources:
                    if source_index == dataset_index:
                        position = source_position
                        break
                positions_by_column.append((column, position))

            for row in dataset._data:
                out_row = [
                    _build_cell(column, row[position])
                    if position is not None
                    else _fill_cell(column)
                    for column, position in positions_by_column
                ]
                result.append(out_row, tags=copy.copy(row.tags))

        return result


def _build_cell(column, value):
    if value is None:
        # An existing None is preserved as None; it is not a missing column.
        return None
    value = copy.deepcopy(value)
    if column.converter is not None:
        return column.converter(value)
    target = column.target_type
    if target in (EMPTY, CUSTOM):
        return value
    kind = value_kind(value)
    if kind == target:
        return value
    return _CONVERTERS[target](value)


def _fill_cell(column):
    fill = column.fill
    value = fill() if column.fill_is_callable else fill
    # Copy per cell so a mutable fill value is never shared across rows.
    return copy.deepcopy(value)


def _fingerprint(dataset):
    """Structural signature captured at planning time.

    Captures width, row count and the current headers in order. Cell values
    and types are deliberately not part of the signature: only structural
    changes invalidate a plan.
    """

    headers = dataset.headers
    return (
        dataset.width,
        dataset.height,
        tuple(headers) if headers is not None else None,
    )


def _resolve_identities(datasets, identity_maps):
    """Return one identity tuple per dataset.

    Headers are used directly when they are unique. Duplicate headers are
    ambiguous and require an explicit position-based identity map.
    """

    resolved = []
    for index, dataset in enumerate(datasets):
        explicit = None
        if identity_maps is not None:
            explicit = identity_maps.get(index, identity_maps.get(dataset))

        if explicit is None:
            headers = dataset.headers
            if headers is None:
                raise HeadersRequired(
                    "dataset #{} has no headers; pass an identity_maps entry "
                    "to name its columns by position".format(index)
                )
            if len(set(headers)) != len(headers):
                raise AmbiguousColumns(
                    "dataset #{} has duplicate headers {}; pass an explicit "
                    "identity_maps entry mapping each position to a unique "
                    "identity".format(index, list(headers))
                )
            resolved.append(tuple(headers))
            continue

        if isinstance(explicit, dict):
            if set(explicit) != set(range(dataset.width)):
                raise InvalidIdentityMap(
                    "identity map for dataset #{} must name every position "
                    "0..{}".format(index, dataset.width - 1)
                )
            identities = tuple(explicit[pos] for pos in range(dataset.width))
        else:
            identities = tuple(explicit)
            if len(identities) != dataset.width:
                raise InvalidIdentityMap(
                    "identity map for dataset #{} has {} entries but the "
                    "dataset is {} columns wide".format(
                        index, len(identities), dataset.width
                    )
                )
        if len(set(identities)) != len(identities):
            raise AmbiguousColumns(
                "identity map for dataset #{} contains duplicate identities"
                .format(index)
            )
        resolved.append(identities)

    return resolved


def _observed_kinds(dataset):
    """Return the observed type family of each column (ignoring ``None``)."""

    width = dataset.width
    kinds = [set() for _ in range(width)]
    for row in dataset._data:
        for position in range(width):
            value = row[position]
            if value is not None:
                kinds[position].add(value_kind(value))
    return [seen or {EMPTY} for seen in kinds]


def _column_order(how, identities):
    first = list(identities[0])
    if not identities[1:]:
        return first

    if how == "intersection":
        common = set(first)
        for rest in identities[1:]:
            common &= set(rest)
        return [identity for identity in first if identity in common]

    # union: first-seen order across datasets
    order = list(first)
    seen = set(first)
    for rest in identities[1:]:
        for identity in rest:
            if identity not in seen:
                seen.add(identity)
                order.append(identity)
    return order


def plan_merge(datasets, policy=None, identity_maps=None, **policy_kwargs):
    """Build a reviewable :class:`MergePlan` for ``datasets``.

    :param datasets: non-empty sequence of
        :class:`~tablib.core.Dataset` instances.
    :param policy: a :class:`ReconcilePolicy`; keyword arguments
        (``how``, ``fill``, ``allow_lossy``, ``converters``) build one if
        omitted.
    :param identity_maps: optional mapping of ``dataset_index -> identities``.
        Each value is either a sequence of identities (one per column
        position) or a ``{position: identity}`` dict. Required for datasets
        with duplicate headers or no headers.
    """

    from .core import Dataset

    datasets = list(datasets)
    if not datasets:
        raise ValueError("at least one Dataset is required")
    for index, dataset in enumerate(datasets):
        if not isinstance(dataset, Dataset):
            raise TypeError(
                "datasets[{}] is not a Dataset instance".format(index)
            )

    if policy is None:
        policy = ReconcilePolicy(**policy_kwargs)
    elif policy_kwargs:
        raise ValueError(
            "pass either a ReconcilePolicy or policy keyword arguments, "
            "not both"
        )

    identities = _resolve_identities(datasets, identity_maps)
    fingerprints = [_fingerprint(dataset) for dataset in datasets]
    observed = [_observed_kinds(dataset) for dataset in datasets]

    order = _column_order(policy.how, identities)
    notes = []
    if policy.how == "intersection" and not order:
        notes.append("intersection is empty: no columns are shared")

    column_plans = []
    for result_index, identity in enumerate(order):
        sources = []
        kinds = set()
        observed_types = {}
        for dataset_index, ids in enumerate(identities):
            if identity in ids:
                position = ids.index(identity)
                sources.append((dataset_index, position))
                column_kinds = observed[dataset_index][position]
                observed_types[dataset_index] = frozenset(column_kinds)
                kinds.update(column_kinds)

        converter = policy.converters.get(identity)
        if converter is not None:
            plan = ColumnPlan(
                identity, result_index, sources, CUSTOM, observed_types,
                False, policy.fill, converter, None,
            )
        else:
            target, lossy, reason = reconcile_types(
                kinds, allow_lossy=policy.allow_lossy
            )
            plan = ColumnPlan(
                identity, result_index, sources, target, observed_types,
                lossy, policy.fill, None, reason,
            )
        column_plans.append(plan)

    return MergePlan(
        datasets, identities, policy.how, column_plans, order,
        fingerprints, notes=notes,
    )

def merge_sets(datasets, policy=None, identity_maps=None, **policy_kwargs):
    """Plan and immediately apply a column-identity merge.

    Equivalent to ``plan_merge(...).apply()``. Use :func:`plan_merge` directly
    when the plan must be reviewed first.
    """

    return plan_merge(
        datasets, policy=policy, identity_maps=identity_maps, **policy_kwargs
    ).apply()
