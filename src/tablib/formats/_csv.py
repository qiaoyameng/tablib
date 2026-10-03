""" Tablib - *SV Support.
"""

__lazy_modules__ = {"codecs", "csv", "io"}

import codecs
import csv
from io import StringIO


def _iter_sized_chunks(parts, chunk_size, empty):
    pending = empty
    for part in parts:
        offset = 0
        if pending:
            needed = chunk_size - len(pending)
            if len(part) < needed:
                pending += part
                continue
            yield pending + part[:needed]
            offset = needed
            pending = empty

        while len(part) - offset >= chunk_size:
            yield part[offset:offset + chunk_size]
            offset += chunk_size
        pending = part[offset:]

    if pending:
        yield pending


class CSVFormat:
    title = 'csv'
    extensions = ('csv',)

    DEFAULT_DELIMITER = ','

    @classmethod
    def export_stream_set(cls, dataset, encoding=None, chunk_size=8192, as_bytes=False, **kwargs):
        """Yield the CSV representation of a Dataset in chunks.

        Text chunks are returned by default. When ``as_bytes`` is true or
        ``encoding`` is provided, encoded byte chunks are returned. Byte output
        defaults to UTF-8. The Dataset headers and rows are snapshotted when
        the iterator is created.
        """
        if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size < 1:
            raise ValueError('chunk_size must be a positive integer')

        kwargs.setdefault('delimiter', cls.DEFAULT_DELIMITER)

        stream = StringIO()
        _csv = csv.writer(stream, **kwargs)
        if as_bytes and encoding is None:
            encoding = 'utf-8'
        encoder = codecs.getincrementalencoder(encoding)() if encoding is not None else None

        headers = list(dataset.headers) if dataset.headers is not None else None
        rows = [list(row) for row in dataset._data]
        formatters = tuple(dataset._formatters)

        def chunks():
            def row_texts():
                if headers is not None:
                    stream.seek(0)
                    stream.truncate(0)
                    _csv.writerow(headers)
                    yield stream.getvalue()

                for row in rows:
                    for col, callback in formatters:
                        if col is None:
                            row[:] = [callback(cell) for cell in row]
                        else:
                            row[col] = callback(row[col])

                    stream.seek(0)
                    stream.truncate(0)
                    _csv.writerow(row)
                    yield stream.getvalue()

            if encoder is None:
                yield from _iter_sized_chunks(row_texts(), chunk_size, '')
            else:
                def encoded_parts():
                    for row_text in row_texts():
                        yield encoder.encode(row_text)
                    yield encoder.encode('', final=True)

                yield from _iter_sized_chunks(encoded_parts(), chunk_size, b'')

        return chunks()

    @classmethod
    def export_set(cls, dataset, **kwargs):
        """Returns CSV representation of Dataset."""
        encoding = kwargs.pop('encoding', None)
        as_bytes = kwargs.pop('as_bytes', False)
        if encoding is not None or as_bytes:
            raise TypeError('export_set() returns text; use export_stream_set() for bytes')

        return ''.join(cls.export_stream_set(dataset, **kwargs))

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
