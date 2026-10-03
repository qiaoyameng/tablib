class TablibException(Exception):
    """Tablib common exception."""


class InvalidDatasetType(TablibException, TypeError):
    """Only Datasets can be added to a Databook."""


class InvalidDimensions(TablibException, ValueError):
    """The size of the column or row doesn't fit the table dimensions."""


class InvalidDatasetIndex(TablibException, IndexError):
    """Outside of Dataset size."""


class HeadersNeeded(TablibException, AttributeError):
    """Header parameter must be given when appending a column to this Dataset."""


class UnsupportedFormat(TablibException, NotImplementedError):
    """Format not supported."""


class HeadersRequired(TablibException, ValueError):
    """Column-identity merge needs headers on every Dataset."""


class AmbiguousColumns(TablibException, ValueError):
    """Duplicate headers require an explicit position-based identity map."""


class InvalidIdentityMap(TablibException, ValueError):
    """The explicit column-identity map does not match the dataset width."""


class PlanStale(TablibException, RuntimeError):
    """The input structure changed after the reconciliation plan was built."""


class MergeRejected(TablibException, ValueError):
    """The plan contains columns that could not be reconciled."""
