""" Tablib - ODF Support.
"""

__lazy_modules__ = {"datetime", "io", "numbers"}

import datetime as dt
import numbers
from io import BytesIO

from odf import number, opendocument, style, table, text

import tablib

bold = style.Style(name="bold", family="paragraph")
bold.addElement(style.TextProperties(
    fontweight="bold",
    fontweightasian="bold",
    fontweightcomplex="bold",
))


def set_date_style(style):
    style.addElement(number.Year(style="long"))
    style.addElement(number.Text(text="-"))
    style.addElement(number.Month(style="long"))
    style.addElement(number.Text(text="-"))
    style.addElement(number.Day(style="long"))


def set_time_style(style):
    style.addElement(number.Hours(style="long"))
    style.addElement(number.Text(text=":"))
    style.addElement(number.Minutes(style="long"))
    style.addElement(number.Text(text=":"))
    style.addElement(number.Seconds(style="long", decimalplaces="0"))


date_style = number.DateStyle(name="date-style1")
set_date_style(date_style)
ds = style.Style(
    name="ds1",
    datastylename="date-style1",
    parentstylename="Default",
    family="table-cell",
)

time_style = number.DateStyle(name="time-style1")
set_time_style(time_style)
ts = style.Style(
    name="ts1",
    datastylename="time-style1",
    parentstylename="Default",
    family="table-cell",
)

datetime_style = number.DateStyle(name="datetime-style1")
set_date_style(datetime_style)
datetime_style.addElement(number.Text(text=" "))
set_time_style(datetime_style)
dts = style.Style(
    name="dts1",
    datastylename="datetime-style1",
    parentstylename="Default",
    family="table-cell",
)


class ODSFormat:
    title = 'ods'
    extensions = ('ods',)

    @staticmethod
    def _repeat_count(node, attribute):
        """Return the 1-based repetition count declared on an ODF node."""
        repeated = node.getAttribute(attribute)
        if repeated is None:
            return 1
        try:
            return max(int(repeated), 1)
        except (TypeError, ValueError):
            return 1

    @staticmethod
    def _normalize_window(window):
        """Normalize a zero-based, half-open ``(start, stop)`` window.

        ``None`` means an unbounded edge, matching the document extent.
        """
        if window is None:
            return 0, None
        try:
            start, stop = window
        except (TypeError, ValueError) as exc:
            raise ValueError("window bounds must be (start, stop) pairs") from exc
        if len(window) != 2:
            raise ValueError("window bounds must be (start, stop) pairs")
        start = max(int(start), 0)
        if stop is None:
            return start, None
        stop = max(int(stop), start)
        return start, stop

    @classmethod
    def _sheet_extent(cls, sheet):
        """Return ``(row_count, column_count)`` from repeated node markers."""
        row_count = 0
        column_count = 0
        for node in sheet.childNodes:
            if node.tagName != "table:table-row":
                continue
            row_count += cls._repeat_count(node, 'numberrowsrepeated')
            row_width = 0
            for cell in node.childNodes:
                if getattr(cell, 'tagName', None) not in (
                    "table:table-cell",
                    "table:covered-table-cell",
                ):
                    continue
                row_width += cls._repeat_count(cell, 'numbercolumnsrepeated')
            column_count = max(column_count, row_width)
        return row_count, column_count

    @classmethod
    def _iter_logical_rows(cls, sheet, row_start, row_stop):
        """Yield ``(logical_index, row_node)`` pairs inside the row window.

        Repeated rows are advanced by their ``table:number-rows-repeated``
        marker; repetitions outside the window are skipped in a single step
        without instantiating the skipped row objects.
        """
        logical_row = 0
        for node in sheet.childNodes:
            if node.tagName != "table:table-row":
                continue
            repeat = cls._repeat_count(node, 'numberrowsrepeated')
            block_end = logical_row + repeat
            if block_end <= row_start:
                logical_row = block_end
                continue
            overlap_start = max(logical_row, row_start)
            overlap_end = block_end if row_stop is None else min(block_end, row_stop)
            for row_index in range(overlap_start, overlap_end):
                yield row_index, node
            logical_row = block_end

    @classmethod
    def _read_row_window(cls, row_node, col_start, col_stop):
        """Read one row through a fixed zero-based, half-open column window.

        Columns are tracked in logical coordinates, so repeated cells,
        empty trailing cells and ``table:covered-table-cell`` placeholders
        cannot shift the selected window. Only intersecting cells are read.
        """
        values = [''] * (col_stop - col_start)
        logical_col = 0
        for cell in row_node.childNodes:
            tag_name = getattr(cell, 'tagName', None)
            if tag_name == "table:covered-table-cell":
                logical_col += cls._repeat_count(cell, 'numbercolumnsrepeated')
                continue
            if tag_name != "table:table-cell":
                continue
            repeat = cls._repeat_count(cell, 'numbercolumnsrepeated')
            block_end = logical_col + repeat
            if block_end <= col_start:
                logical_col = block_end
                continue
            overlap_start = max(logical_col, col_start)
            overlap_end = min(block_end, col_stop)
            if overlap_start < overlap_end:
                converted = cls.read_cell(cell)
                for position in range(overlap_start, overlap_end):
                    values[position - col_start] = converted
            logical_col = block_end
        return values

    @classmethod
    def _read_ragged_row(cls, row_node, col_start):
        """Read a row with an unbounded right edge, keeping legacy semantics.

        Blank repeated cells (including trailing blanks) produce no values,
        while covered cells count as empty placeholders for their span.
        """
        row_vals = []
        logical_col = 0
        for cell in row_node.childNodes:
            tag_name = getattr(cell, 'tagName', None)
            if tag_name == "table:covered-table-cell":
                repeat = cls._repeat_count(cell, 'numbercolumnsrepeated')
                for _ in range(repeat):
                    if logical_col >= col_start:
                        row_vals.append('')
                    logical_col += 1
                continue
            if tag_name != "table:table-cell":
                continue
            repeat = cls._repeat_count(cell, 'numbercolumnsrepeated')
            is_blank_repeat = (
                not cell.hasChildNodes()
                and cell.getAttribute('numbercolumnsrepeated') is not None
            )
            if is_blank_repeat:
                logical_col += repeat
                continue
            converted = None
            for position in range(logical_col, logical_col + repeat):
                if position < col_start:
                    continue
                if converted is None:
                    converted = cls.read_cell(cell)
                row_vals.append(converted)
            logical_col += repeat
        return row_vals

    @classmethod
    def export_set(cls, dataset):
        """Returns ODF representation of Dataset."""

        wb = opendocument.OpenDocumentSpreadsheet()
        wb.automaticstyles.addElement(bold)
        wb.styles.addElement(date_style)
        wb.automaticstyles.addElement(ds)
        wb.styles.addElement(time_style)
        wb.automaticstyles.addElement(ts)
        wb.styles.addElement(datetime_style)
        wb.automaticstyles.addElement(dts)

        ws = table.Table(name=dataset.title if dataset.title else 'Tablib Dataset')
        wb.spreadsheet.addElement(ws)
        cls.dset_sheet(dataset, ws)

        stream = BytesIO()
        wb.save(stream)
        return stream.getvalue()

    @classmethod
    def export_book(cls, databook):
        """Returns ODF representation of DataBook."""

        wb = opendocument.OpenDocumentSpreadsheet()
        wb.automaticstyles.addElement(bold)

        for i, dset in enumerate(databook._datasets):
            ws = table.Table(name=dset.title if dset.title else f"Sheet{i}")
            wb.spreadsheet.addElement(ws)
            cls.dset_sheet(dset, ws)

        stream = BytesIO()
        wb.save(stream)
        return stream.getvalue()

    @classmethod
    def import_sheet(cls, dset, sheet, headers=True, skip_lines=0, window=None):
        """Populate dataset `dset` with sheet data."""

        dset.title = sheet.getAttribute('name')

        if window is not None:
            if len(window) != 2:
                raise ValueError(
                    "window must be (row_window, column_window), "
                    "each a (start, stop) pair"
                )
            if skip_lines:
                raise ValueError("skip_lines cannot be combined with a window")
            row_start, row_stop = cls._normalize_window(window[0])
            col_start, col_stop = cls._normalize_window(window[1])
            sheet_rows, sheet_cols = cls._sheet_extent(sheet)
            if row_stop is None:
                row_stop = max(sheet_rows, row_start)
            if col_stop is None:
                col_stop = max(sheet_cols, col_start)
        else:
            row_start, row_stop = cls._normalize_window((skip_lines, None))
            col_start, col_stop = 0, None

        windowed = row_stop is not None and col_stop is not None
        width = (col_stop - col_start) if windowed else 0
        header_pending = headers
        if windowed:
            expected_rows = row_stop - row_start
            seen_rows = 0
            for logical_row, row in cls._iter_logical_rows(sheet, row_start, row_stop):
                row_vals = cls._read_row_window(row, col_start, col_stop)
                if len(row_vals) < width:
                    row_vals += [''] * (width - len(row_vals))
                if header_pending:
                    dset.headers = row_vals
                    header_pending = False
                else:
                    dset.append(row_vals)
                seen_rows += 1
            if headers and dset.headers is None:
                dset.headers = [''] * width
            for _ in range(expected_rows - seen_rows):
                dset.append([''] * width)
            return

        for logical_row, row in cls._iter_logical_rows(sheet, row_start, row_stop):
            row_vals = cls._read_ragged_row(row, col_start)
            if not row_vals:
                continue
            if header_pending:
                dset.headers = row_vals
                header_pending = False
            else:
                if dset.width and len(row_vals) < dset.width:
                    row_vals += [''] * (dset.width - len(row_vals))
                dset.append(row_vals)

    @classmethod
    def read_cell(cls, cell, value_type=None):
        def convert_date(val):
            if 'T' in val:
                return dt.datetime.strptime(val, "%Y-%m-%dT%H:%M:%S")
            else:
                return dt.datetime.strptime(val, "%Y-%m-%d").date()

        if value_type is None:
            value_type = cell.getAttribute('valuetype')
        if value_type == 'date':
            date_value = cell.getAttribute('datevalue')
            if date_value:
                return convert_date(date_value)
        if value_type == 'time':
            time_value = cell.getAttribute('timevalue')
            try:
                return dt.datetime.strptime(time_value, "PT%HH%MM%SS").time()
            except ValueError:
                # backwards compatibility for times exported with older tablib versions
                return dt.datetime.strptime(time_value, "%H:%M:%S").time()
        if value_type == 'boolean':
            bool_value = cell.getAttribute('booleanvalue')
            return bool_value == 'true'
        if not cell.childNodes:
            value = getattr(cell, 'data', None)
            if value is None:
                try:
                    value = cell.getAttribute('value')
                except ValueError:
                    pass
            if value is None:
                return ''
            if value_type == 'float':
                return float(value)
            if value_type == 'date':
                return convert_date(value)
            return value  # Any other type default to 'string'

        for subnode in cell.childNodes:
            return cls.read_cell(subnode, value_type)

    @classmethod
    def import_set(cls, dset, in_stream, headers=True, skip_lines=0, window=None):
        """Populate dataset `dset` from ODS stream."""

        dset.wipe()

        ods_book = opendocument.load(in_stream)
        sheet_index = 0
        for sheet in ods_book.spreadsheet.childNodes:
            if sheet.qname[1] == 'table':
                sheet_window = cls._sheet_window(window, sheet, sheet_index)
                cls.import_sheet(dset, sheet, headers, skip_lines, sheet_window)
                sheet_index += 1

    @classmethod
    def import_book(cls, dbook, in_stream, headers=True, window=None):
        """Populate databook `dbook` from ODS stream."""

        dbook.wipe()

        ods_book = opendocument.load(in_stream)

        sheet_index = 0
        for sheet in ods_book.spreadsheet.childNodes:
            if sheet.qname[1] != 'table':
                continue
            dset = tablib.Dataset()
            cls.import_sheet(dset, sheet, headers,
                             window=cls._sheet_window(window, sheet, sheet_index))
            dbook.add_sheet(dset)
            sheet_index += 1

    @staticmethod
    def _sheet_window(window, sheet, sheet_index):
        """Resolve the window applying to one sheet.

        ``window`` is either a single ``(row_window, col_window)`` tuple used
        for every sheet, or a mapping keyed by sheet name / zero-based index.
        Sheets without an entry fall back to no window.
        """
        if window is None:
            return None
        if isinstance(window, dict):
            name = sheet.getAttribute('name')
            if name in window:
                return window[name]
            return window.get(sheet_index)
        return window

    @classmethod
    def dset_sheet(cls, dataset, ws):
        """Completes given worksheet from given Dataset."""
        _package = dataset._package(dicts=False)

        for i, sep in enumerate(dataset._separators):
            _offset = i
            _package.insert((sep[0] + _offset), (sep[1],))

        for row_number, row in enumerate(_package, start=1):
            is_header = row_number == 1 and dataset.headers
            style = bold if is_header else None
            odf_row = table.TableRow(stylename=style)
            ws.addElement(odf_row)
            for j, col in enumerate(row):
                if isinstance(col, bool):
                    cell = table.TableCell(
                        valuetype="boolean",
                        booleanvalue="true" if col else "false",
                    )
                elif isinstance(col, numbers.Number):
                    cell = table.TableCell(valuetype="float", value=col)
                elif isinstance(col, dt.datetime):
                    cell = table.TableCell(
                        valuetype="date",
                        datevalue=col.strftime('%Y-%m-%dT%H:%M:%S'),
                        stylename=dts,
                    )
                    cell.addElement(text.P(text=col.strftime('%Y-%m-%d %H:%M:%S')))
                elif isinstance(col, dt.date):
                    date_value = col.strftime('%Y-%m-%d')
                    cell = table.TableCell(valuetype="date", datevalue=date_value, stylename=ds)
                    cell.addElement(text.P(text=date_value))
                elif isinstance(col, dt.time):
                    cell = table.TableCell(
                        valuetype="time",
                        timevalue=col.strftime('PT%HH%MM%SS'),
                        stylename=ts,
                    )
                    cell.addElement(text.P(text=col.strftime('%H:%M:%S')))
                elif col is None:
                    cell = table.TableCell(valuetype="void")
                else:
                    cell = table.TableCell(valuetype="string")
                    cell.addElement(text.P(text=str(col), stylename=style))
                odf_row.addElement(cell)

    @classmethod
    def detect(cls, stream):
        if isinstance(stream, bytes):
            # load expects a file-like object.
            stream = BytesIO(stream)
        try:
            opendocument.load(stream)
            return True
        except Exception:
            return False
