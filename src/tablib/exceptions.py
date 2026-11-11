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


class SnapshotReadOnlyError(TablibException, TypeError):
    """A mutating operation was attempted on a read-only snapshot or view."""


class SnapshotMutableCellError(TablibException, ValueError):
    """A mutable cell was encountered while taking a snapshot in ``reject`` mode."""
