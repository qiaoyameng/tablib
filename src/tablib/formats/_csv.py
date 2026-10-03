""" Tablib - *SV Support.
"""

__lazy_modules__ = {"codecs", "csv", "io"}

import codecs
import csv
from io import StringIO


class CSVFormat:
    title = 'csv'
    extensions = ('csv',)

    DEFAULT_DELIMITER = ','

    # Default size (in characters) of the text chunks yielded by
    # ``export_set_chunks``.
    STREAM_CHUNK_SIZE = 8192

    @classmethod
    def _snapshot(cls, dataset):
        """Take a fixed structural snapshot of ``dataset``.

        Rows are shallow-copied and the formatter list is captured, so rows
        written during an export keep a uniform width even if the caller
        mutates the :class:`~tablib.core.Dataset` (adding or removing rows or
        columns) while iteration is in progress. Cell values themselves are
        not copied.
        """
        return (
            list(dataset.headers) if dataset.headers else None,
            [row.copy() for row in dataset._data],
            list(dataset._formatters),
        )

    @staticmethod
    def _format_row(row, formatters):
        """Apply ``formatters`` to a row, formatting each cell at most once."""
        if formatters:
            row = row.copy()
            for col, callback in formatters:
                if col is None:
                    row = [callback(cell) for cell in row]
                else:
                    row[col] = callback(row[col])
        return list(row)

    @classmethod
    def _iter_rows(cls, dataset):
        """Yield packaged rows (optional header, then data rows) lazily.

        The structural snapshot is taken when iteration starts, and data row
        formatters are only invoked as each row is requested.
        """
        headers, rows, formatters = cls._snapshot(dataset)
        if headers is not None:
            yield list(headers)
        for row in rows:
            yield cls._format_row(row, formatters)

    @classmethod
    def _iter_text_chunks(cls, dataset, chunk_size, writer_kwargs):
        """Yield CSV text chunks, running at most one full row of formatters
        ahead of the caller."""
        writer_buffer = StringIO()
        writer = csv.writer(writer_buffer, **writer_kwargs)
        pending = ''
        for row in cls._iter_rows(dataset):
            writer.writerow(row)
            pending += writer_buffer.getvalue()
            writer_buffer.seek(0)
            writer_buffer.truncate(0)
            while len(pending) >= chunk_size:
                yield pending[:chunk_size]
                pending = pending[chunk_size:]
        if pending:
            yield pending

    @classmethod
    def export_set_chunks(cls, dataset, *, chunk_size=STREAM_CHUNK_SIZE,
                          encoding=None, errors='strict', **kwargs):
        """Yield a CSV representation of ``dataset`` as lazily produced chunks.

        Rows are formatted on demand: iteration stops as soon as the caller
        stops consuming, and a formatter never runs for a row that was not
        reached. Chunks already yielded before an error are not retracted.

        A fixed structural snapshot is taken when iteration begins, so
        modifying the :class:`~tablib.core.Dataset` during iteration cannot
        produce rows of mismatched width; such a mutation simply does not
        affect the in-progress export.

        :param chunk_size: approximate size of each yielded chunk, measured
            in characters (or in bytes after encoding). The final chunk may
            be smaller; individual rows longer than ``chunk_size`` are still
            emitted as a single chunk.
        :param encoding: if given, chunks are yielded as ``bytes`` encoded
            incrementally with :func:`codecs.getincrementalencoder`.
            Signature-bearing codecs (e.g. ``utf-8-sig`` or ``utf-16``) emit
            their BOM exactly once, in the first chunk.
        :param errors: codec error handling scheme, used with ``encoding``.
        :param \\*\\*kwargs: extra parameters for :func:`csv.writer`.
        """
        if chunk_size < 1:
            raise ValueError('chunk_size must be a positive integer')

        kwargs.setdefault('delimiter', cls.DEFAULT_DELIMITER)

        text_chunks = cls._iter_text_chunks(dataset, chunk_size, kwargs)

        if encoding is None:
            yield from text_chunks
            return

        encoder = codecs.getincrementalencoder(encoding)(errors=errors)
        for text_chunk in text_chunks:
            encoded = encoder.encode(text_chunk, False)
            if encoded:
                yield encoded
        encoded = encoder.encode('', True)
        if encoded:
            yield encoded

    @classmethod
    def export_stream_set(cls, dataset, **kwargs):
        """Returns CSV representation of Dataset as file-like."""
        return StringIO(''.join(cls.export_set_chunks(dataset, **kwargs)))

    @classmethod
    def export_set(cls, dataset, **kwargs):
        """Returns CSV representation of Dataset."""
        return ''.join(cls.export_set_chunks(dataset, **kwargs))

    @classmethod
    def import_set(cls, dset, in_stream, headers=True, skip_lines=0, **kwargs):
        """Returns dataset from CSV stream."""

        dset.wipe()

        kwargs.setdefault('delimiter', cls.DEFAULT_DELIMITER)

        rows = csv.reader(in_stream, **kwargs)
        for i, row in enumerate(rows):
            if i < skip_lines:
                continue
            if i == skip_lines and headers:
                dset.headers = row
            elif row:
                if i > 0 and len(row) < dset.width:
                    row += [''] * (dset.width - len(row))
                dset.append(row)

    @classmethod
    def detect(cls, stream, delimiter=None):
        """Returns True if given stream is valid CSV."""
        try:
            csv.Sniffer().sniff(stream.read(2048), delimiters=delimiter or cls.DEFAULT_DELIMITER)
            return True
        except Exception:
            return False
