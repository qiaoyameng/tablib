"""
    tablib.merging
    ~~~~~~~~~~~~~~

    Merge :class:`~tablib.Dataset` instances by column identity rather than
    by positional width.

    The workflow is deliberately split in two steps:

    1. :func:`plan_merge` produces a reviewable :class:`MergePlan` describing
       the resulting column order, the (lossless) value conversions and every
       rejection reason.
    2. :meth:`MergePlan.apply` constructs a new :class:`~tablib.Dataset`,
       re-checking that the inputs still match the structure the plan was
       built from.

    Only lossless type reconciliations are applied by default. Lossy
    combinations (e.g. ``Decimal`` with ``float``) must be resolved explicitly
    by the caller through the ``coercions`` argument.
"""

import datetime as _dt
from copy import deepcopy
from decimal import Decimal

from .core import Dataset, Row
from .exceptions import HeadersNeeded, TablibException

UNION = 'union'
INTERSECTION = 'intersection'

# Type family labels exposed in plans and used as coercion keys.
BOOL = 'bool'
INT = 'int'
FLOAT = 'float'
DECIMAL = 'decimal'
DATE = 'date'
DATETIME = 'datetime'
TIME = 'time'
TEXT = 'text'
UNKNOWN = 'unknown'

# Order of the well-known type families, used both to name pairwise
# combination keys deterministically and to prefer the wider family.
_KIND_ORDER = [BOOL, INT, FLOAT, DECIMAL, DATE, DATETIME, TIME, TEXT]


class MissingColumn:
    """Sentinel marking a cell whose column is absent from a source dataset.

    It is intentionally distinct from ``None``: ``None`` remains a regular
    value a cell may already hold, while a :class:`MissingColumn` is replaced
    at application time with the fill value configured in the plan.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self):
        return 'MISSING'

    def __deepcopy__(self, memo):
        return self


MISSING = MissingColumn()


class MergePlanError(TablibException, ValueError):
    """A merge plan cannot be created from the given inputs."""


class MergePlanStale(MergePlanError):
    """The inputs changed after the plan was created."""


def _kind(value):
    """Classify a non-missing cell value into a supported type family."""
    # bool is a subclass of int and datetime a subclass of date, so both
    # checks must happen before their wider relatives.
    if isinstance(value, bool):
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
    return UNKNOWN


def _float_to_text(value):
    return repr(value)


def _decimal_to_text(value):
    return format(value, 'f')


def _float_to_decimal(value):
    # Decimal(str(...)) preserves the value as the caller wrote it, instead
    # of exposing binary float artefacts.
    return Decimal(str(value))


# (source_kind, target_kind) -> converter. Every row is a lossless default.
_LOSSLESS_CONVERTERS = {
    (BOOL, INT): int,
    (BOOL, FLOAT): lambda value: float(int(value)),
    (BOOL, DECIMAL): lambda value: Decimal(int(value)),
    (BOOL, TEXT): str,
    (INT, FLOAT): float,
    (INT, DECIMAL): Decimal,
    (INT, TEXT): str,
    (FLOAT, TEXT): _float_to_text,
    (DECIMAL, TEXT): _decimal_to_text,
    (DATETIME, TEXT): lambda value: value.isoformat(sep=' '),
    (DATE, TEXT): lambda value: value.isoformat(),
    (TIME, TEXT): lambda value: value.isoformat(),
}

# Conversions the caller may opt into; these are lossy or directional.
_OPT_IN_CONVERTERS = {
    (FLOAT, DECIMAL): _float_to_decimal,
    (DECIMAL, FLOAT): float,
    (DATETIME, DATE): lambda value: value.date(),
    (DATE, DATETIME): lambda value: _dt.datetime.combine(value, _dt.time()),
}

# Pairs without a lossless edge. ``'lossy'`` pairs are resolvable through
# ``coercions``; ``'incompatible'`` pairs never are.
_PAIR_NATURE = {
    (BOOL, DATE): 'incompatible',
    (BOOL, DATETIME): 'incompatible',
    (BOOL, TIME): 'incompatible',
    (INT, DATE): 'incompatible',
    (INT, DATETIME): 'incompatible',
    (INT, TIME): 'incompatible',
    (FLOAT, DATE): 'incompatible',
    (FLOAT, DATETIME): 'incompatible',
    (FLOAT, TIME): 'incompatible',
    (FLOAT, DECIMAL): 'lossy',
    (DATE, DATETIME): 'lossy',
    (DECIMAL, DATE): 'incompatible',
    (DECIMAL, DATETIME): 'incompatible',
    (DECIMAL, TIME): 'incompatible',
    (DATE, TIME): 'incompatible',
    (DATETIME, TIME): 'incompatible',
}


def _canonical_pair(kind_a, kind_b):
    index_a = _KIND_ORDER.index(kind_a)
    index_b = _KIND_ORDER.index(kind_b)
    if index_a <= index_b:
        return kind_a, kind_b
    return kind_b, kind_a


def _combine_kinds(target, kind, coercions):
    """Return the reconciled target kind for a running ``target`` plus a new
    ``kind``. Raises :class:`MergePlanError` with ``'lossy'`` or
    ``'incompatible'`` as the message when no default reconciliation exists.
    """
    if kind == UNKNOWN:
        return target
    if target == UNKNOWN or kind == target:
        return kind if target == UNKNOWN else target

    pair = _canonical_pair(target, kind)
    low, high = pair

    chosen = coercions.get(pair)
    if chosen is None:
        if pair in _LOSSLESS_CONVERTERS:
            chosen = high
        else:
            raise MergePlanError(_PAIR_NATURE.get(pair, 'incompatible'))

    if chosen not in (low, high):
        # Lossy pair steered toward a shared wider type (e.g. -> text).
        if not (
            (low, chosen) in _LOSSLESS_CONVERTERS
            and (high, chosen) in _LOSSLESS_CONVERTERS
        ):
            raise MergePlanError(
                f'coercion target {chosen!r} cannot reconcile '
                f'{low!r}/{high!r}'
            )

    return chosen


class ColumnPlan:
    """Reviewable plan for a single result column.

    Attributes:
        name: Result column header.
        order: Zero-based position in the result.
        sources: ``{dataset_index: source_position_or_None}``; ``None`` means
            the column is missing in that dataset.
        kinds: Observed type family per dataset (``None`` for missing columns
            or empty datasets).
        target_kind: Final type family all values are converted to, or
            ``None`` when the column was rejected.
        converters: ``{dataset_index: {source_kind: callable_or_None}}``
            used to promote each non-missing value at application time.
        fill: Value inserted for structurally missing columns.
        rejected: Whether the column cannot be produced as configured.
        reasons: List of human-readable rejection/mapping reasons.
    """

    __slots__ = (
        'name', 'order', 'sources', 'kinds', 'target_kind', 'converters',
        'fill', 'rejected', 'reasons',
    )

    def __init__(self, name, order):
        self.name = name
        self.order = order
        self.sources = {}
        self.kinds = {}
        self.target_kind = None
        self.converters = {}
        self.fill = None
        self.rejected = False
        self.reasons = []

    def __repr__(self):
        status = 'rejected' if self.rejected else 'ok'
        return f'<ColumnPlan {self.name!r} {status} -> {self.target_kind!r}>'

    def to_dict(self):
        return {
            'name': self.name,
            'order': self.order,
            'sources': dict(self.sources),
            'kinds': {str(key): value for key, value in self.kinds.items()},
            'converters': {str(key): value for key, value in self.converters.items()},
            'target_kind': self.target_kind,
            'fill': self.fill,
            'rejected': self.rejected,
            'reasons': list(self.reasons),
        }


class MergePlan:
    """A reviewable, independently applicable column-identity merge plan.

    Construct plans with :func:`plan_merge` (or
    :meth:`Dataset.plan_merge`) rather than instantiating this directly.
    Inspect :attr:`columns` before calling :meth:`apply`; a plan whose
    :attr:`ok` is false raises if applied unchanged.
    """

    def __init__(self, how, columns, structure, fill, titles=()):
        self.how = how
        self.columns = columns
        self._structure = structure
        self._fill = fill
        self.titles = list(titles)

    def __repr__(self):
        status = 'rejected' if not self.ok else 'ready'
        return f'<MergePlan {status} columns={[c.name for c in self.columns]}>'

    @property
    def ok(self):
        """True when every result column can be produced."""
        return all(not column.rejected for column in self.columns)

    @property
    def result_headers(self):
        """Column headers of the dataset produced by :meth:`apply`, in order."""
        return [column.name for column in self.columns]

    @property
    def rejections(self):
        """``{column_name: [reasons]}`` for every rejected column."""
        return {
            column.name: list(column.reasons)
            for column in self.columns
            if column.rejected
        }

    def to_dict(self):
        """Serializable summary suitable for review or logging."""
        return {
            'how': self.how,
            'ok': self.ok,
            'headers': self.result_headers,
            'rejections': self.rejections,
            'columns': [column.to_dict() for column in self.columns],
        }

    def apply(self, datasets=None, fill=None):
        """Build the merged :class:`~tablib.Dataset`.

        The source datasets are matched against the structure captured at
        planning time. Pass the same objects (or structurally identical ones)
        explicitly when the plan was built in another scope.

        :param fill: override the default fill value; may also be a mapping
            of header to value or a callable taking the :class:`ColumnPlan`.
        """
        if not self.ok:
            details = '; '.join(
                name + ': ' + ', '.join(reasons)
                for name, reasons in self.rejections.items()
            )
            raise MergePlanError(f'plan has unresolved rejections: {details}')

        if datasets is None:
            datasets = self._datasets
        datasets = list(datasets)

        current = [_snapshot(dataset) for dataset in datasets]
        if current != self._structure:
            raise MergePlanStale(
                'input structure changed after the plan was created'
            )

        fills = _resolve_fills(self.columns, fill if fill is not None else self._fill)

        result = Dataset(headers=self.result_headers)
        for dataset_index, dataset in enumerate(datasets):
            for row in dataset._data:
                values = []
                for column in self.columns:
                    position = column.sources.get(dataset_index)
                    if position is None:
                        values.append(deepcopy(fills[column.name]))
                        continue
                    value = row[position]
                    if value is None:
                        values.append(None)
                    elif isinstance(value, MissingColumn):
                        values.append(deepcopy(fills[column.name]))
                    else:
                        value_kind = _kind(value)
                        if value_kind == UNKNOWN:
                            raise MergePlanError(
                                f'unsupported value type in column '
                                f'{column.name!r} of dataset {dataset_index}'
                            )
                        converter = (
                            column.converters.get(dataset_index, {})
                            .get(value_kind)
                        )
                        if converter is not None:
                            value = converter(value)
                        values.append(deepcopy(value))
                result._data.append(Row(values, tags=list(row.tags or ())))

        return result

    # Populated by plan_merge() so plan.apply() works without passing the
    # datasets again. Kept private and out of to_dict() (not serializable).
    _datasets = ()


def _snapshot(dataset):
    """Structural fingerprint used to detect post-planning edits."""
    return {
        'headers': list(dataset.headers) if dataset.headers is not None else None,
        'width': dataset.width,
        'height': dataset.height,
    }


def _resolve_sources(dataset, index, mappings):
    """Return an ordered {result_header: position} mapping for one dataset.

    Datasets with unique headers resolve by header name automatically.
    Duplicate headers force an explicit, position-based mapping entry; the
    resulting column order follows the source positions.
    """
    headers = dataset.headers
    mapping = mappings.get(index, {}) if mappings else {}
    if not isinstance(mapping, dict):
        raise MergePlanError(f'mappings[{index}] must be a dict')

    seen = set()
    duplicated = {header for header in headers if header in seen or seen.add(header)}

    if not duplicated:
        if mapping:
            raise MergePlanError(
                f'mappings[{index}] is not allowed: headers are unique'
            )
        return {header: position for position, header in enumerate(headers)}

    ambiguous_positions = {
        position
        for position, header in enumerate(headers)
        if header in duplicated
    }
    if set(mapping.values()) != ambiguous_positions:
        raise MergePlanError(
            f'mappings[{index}] must explicitly name every position with a '
            f'duplicate header ({sorted(ambiguous_positions)})'
        )

    resolved = {}
    for position, header in enumerate(headers):
        if header in duplicated:
            result_name = next(
                name for name, mapped_position in mapping.items()
                if mapped_position == position
            )
            if result_name in resolved:
                raise MergePlanError(
                    f'mappings[{index}] assigns {result_name!r} twice'
                )
            resolved[result_name] = position
        else:
            if header in resolved:
                raise MergePlanError(
                    f'mappings[{index}]: mapped name collides with header '
                    f'{header!r}'
                )
            resolved[header] = position
    return resolved


def _observed_kinds(dataset, position):
    """Return (kinds_in_first_seen_order, unknown_python_type_names)."""
    kinds = []
    unknown_types = set()
    for row in dataset._data:
        value = row[position]
        if value is None or isinstance(value, MissingColumn):
            continue
        kind = _kind(value)
        if kind == UNKNOWN:
            unknown_types.add(type(value).__name__)
        if kind not in kinds:
            kinds.append(kind)
    return kinds, unknown_types


def _path_converter(source, target, opt_edges):
    """BFS for source -> target over lossless edges plus chosen opt-in
    edges. ``opt_edges`` maps (source_kind, target_kind) -> converter.
    Raises MergePlanError when no path exists.
    """
    if source == target:
        return None

    # Adjacency: lossless graph first, selected opt-in directions second.
    graph = {}
    for (src, dst), converter in _LOSSLESS_CONVERTERS.items():
        graph.setdefault(src, {})[dst] = converter
    for (src, dst), converter in opt_edges.items():
        graph.setdefault(src, {})[dst] = converter

    queue = [source]
    routes = {source: []}
    while queue:
        current = queue.pop(0)
        for nxt, converter in graph.get(current, {}).items():
            if nxt in routes:
                continue
            route = routes[current] + [converter]
            routes[nxt] = route
            if nxt == target:
                def composed(value, funcs=route):
                    for func in funcs:
                        value = func(value)
                    return value
                return composed
            queue.append(nxt)
    raise MergePlanError(
        f'no supported conversion from {source!r} to {target!r}'
    )


def _coercion_is_valid(low, high, chosen):
    """Validate one caller coercion for a lossy/conflicting pair.

    ``chosen`` may be either member of the pair (using the lossy edge in the
    requested direction) or a wider type reachable from *both* members
    without another lossy step (e.g. decimal/float -> text).
    """
    if chosen == high:
        return (
            (low, high) in _LOSSLESS_CONVERTERS
            or (low, high) in _OPT_IN_CONVERTERS
        )
    if chosen == low:
        return (high, low) in _OPT_IN_CONVERTERS
    # Third-party widening target: both members must reach it through
    # lossless edges only (a second lossy hop would need its own decision).
    return (
        chosen != low
        and (low, chosen) in _LOSSLESS_CONVERTERS
        and (high, chosen) in _LOSSLESS_CONVERTERS
    )


def _resolve_fills(columns, fill):
    fills = {}
    for column in columns:
        if callable(fill) and not isinstance(fill, str):
            value = fill(column)
        elif isinstance(fill, dict):
            value = fill.get(column.name, None)
        else:
            value = fill
        fills[column.name] = value
    return fills


def plan_merge(datasets, how=UNION, mappings=None, coercions=None, fill=None):
    """Create a reviewable :class:`MergePlan` for column-identity merging.

    :param datasets: iterable of :class:`~tablib.Dataset` instances; every
        dataset must have headers.
    :param how: :data:`UNION` (default) keeps every column and fills missing
        ones; :data:`INTERSECTION` keeps only columns present in every input.
    :param mappings: ``{dataset_index: {result_header: position}}`` required
        for any dataset whose headers contain duplicates.
    :param coercions: ``{(kind_a, kind_b): target_kind}`` decisions for
        lossy type combinations such as ``{('decimal', 'float'): 'decimal'}``.
    :param fill: fill value for structurally missing columns; may be a fixed
        value, a ``{header: value}`` mapping, or a callable receiving the
        :class:`ColumnPlan`. Defaults to ``None`` but is recorded separately
        from cells that already contain ``None``.
    """
    datasets = list(datasets)
    if len(datasets) < 2:
        raise MergePlanError('at least two datasets are required to merge')
    for index, dataset in enumerate(datasets):
        if not isinstance(dataset, Dataset):
            raise TypeError(f'datasets[{index}] must be a Dataset instance')
        if dataset.headers is None:
            raise HeadersNeeded(
                f'datasets[{index}] has no headers; identity merge needs headers'
            )

    if how not in (UNION, INTERSECTION):
        raise MergePlanError(f"how must be 'union' or 'intersection', got {how!r}")

    mappings = mappings or {}
    coercions = dict(coercions or {})
    normalized_coercions = {}
    for pair, chosen in coercions.items():
        try:
            low, high = _canonical_pair(*pair)
        except ValueError:
            raise MergePlanError(
                f'coercion key {pair!r} must contain two type family names'
            )
        if not _coercion_is_valid(low, high, chosen):
            raise MergePlanError(
                f'coercion target {chosen!r} cannot reconcile {low!r}/{high!r}'
            )
        normalized_coercions[(low, high)] = chosen

    resolved = [
        _resolve_sources(dataset, index, mappings)
        for index, dataset in enumerate(datasets)
    ]

    # Result column order: first-seen order of result header names.
    ordered_names = []
    for sources in resolved:
        for name in sources:
            if name not in ordered_names:
                ordered_names.append(name)

    presence = [set(sources) for sources in resolved]
    if how == INTERSECTION:
        common = set.intersection(*presence)
        ordered_names = [name for name in ordered_names if name in common]

    columns = [ColumnPlan(name, order) for order, name in enumerate(ordered_names)]
    columns_by_name = {column.name: column for column in columns}

    for dataset_index, sources in enumerate(resolved):
        for name, position in sources.items():
            column = columns_by_name.get(name)
            if column is None:
                continue  # dropped under INTERSECTION
            column.sources[dataset_index] = position

    for column in columns:
        for dataset_index in range(len(datasets)):
            if dataset_index not in column.sources:
                column.sources[dataset_index] = None
                column.kinds[dataset_index] = None

    # Type reconciliation per result column.
    for column in columns:
        # kinds observed in each dataset (first-seen order within a dataset)
        observed = {}
        target = UNKNOWN
        for dataset_index, dataset in enumerate(datasets):
            position = column.sources.get(dataset_index)
            if position is None:
                column.kinds[dataset_index] = None
                continue
            kinds, unknown_types = _observed_kinds(dataset, position)
            column.kinds[dataset_index] = list(kinds) if kinds else None
            if unknown_types:
                column.rejected = True
                column.reasons.append(
                    f'dataset {dataset_index} column {column.name!r} contains '
                    f'unsupported type(s): {sorted(unknown_types)}'
                )
                continue
            for kind in kinds:
                try:
                    target = _combine_kinds(target, kind, normalized_coercions)
                except MergePlanError as reason:
                    column.rejected = True
                    pair = _canonical_pair(target, kind)
                    if str(reason) == 'lossy':
                        column.reasons.append(
                            f'lossy combination {pair[0]!r}/{pair[1]!r} in '
                            f'{column.name!r}; pass a coercions decision '
                            f'(e.g. {{{pair!r}: {pair[0]!r} or {pair[1]!r}}})'
                        )
                    else:
                        column.reasons.append(
                            f'incompatible types {pair[0]!r}/{pair[1]!r} in '
                            f'{column.name!r}'
                        )
            observed[dataset_index] = kinds

        if column.rejected:
            continue

        column.target_kind = None if target == UNKNOWN else target

        # Build a per-dataset converter map keyed by each observed source
        # kind, so individual cells inside one source column are promoted
        # correctly (e.g. bool cells inside an int/float column).
        opt_edges = {}
        for pair, chosen in normalized_coercions.items():
            low, high = pair
            if chosen == high:
                converter = _OPT_IN_CONVERTERS.get((low, high))
                if converter is not None:
                    opt_edges[(low, high)] = converter
            elif chosen == low:
                converter = _OPT_IN_CONVERTERS.get((high, low))
                if converter is not None:
                    opt_edges[(high, low)] = converter

        for dataset_index, kinds in observed.items():
            by_kind = {}
            for source_kind in kinds:
                if source_kind == target:
                    by_kind[source_kind] = None
                else:
                    by_kind[source_kind] = _path_converter(
                        source_kind, target, opt_edges
                    )
            column.converters[dataset_index] = by_kind

    structure = [_snapshot(dataset) for dataset in datasets]
    fills = _resolve_fills(columns, fill)
    for column in columns:
        column.fill = fills[column.name]

    plan = MergePlan(
        how=how,
        columns=columns,
        structure=structure,
        fill=fill,
        titles=[dataset.title for dataset in datasets],
    )
    plan._datasets = tuple(datasets)
    return plan


def merge(datasets, how=UNION, mappings=None, coercions=None, fill=None):
    """Plan and immediately apply a column-identity merge.

    Raises :class:`MergePlanError` when the resulting plan has unresolved
    rejections. Prefer :func:`plan_merge` when the plan needs review.
    """
    plan = plan_merge(
        datasets,
        how=how,
        mappings=mappings,
        coercions=coercions,
        fill=fill,
    )
    return plan.apply(datasets)
