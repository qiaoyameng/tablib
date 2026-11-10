""" Tablib - HTML export support.
"""

__lazy_modules__ = {"xml", "xml.etree"}

import json
from html.parser import HTMLParser
from xml.etree import ElementTree as ET


class StructuredHeaderError(ValueError):
    """Raised when structured HTML header extraction fails.

    Carries the per-cell associations computed so far in ``associations``
    when the failure happens after the grid was built.
    """

    def __init__(self, message, associations=None):
        super().__init__(message)
        self.associations = associations


class StructuredHeadersResult:
    """Outcome of a ``structured_headers=True`` HTML import.

    ``associations`` is a list of per-cell records (never a dict keyed by
    header text, so same-named headers keep distinct sources).
    """

    def __init__(self, associations, width):
        self.associations = associations
        self.width = width


class HTMLFormat:
    BOOK_ENDINGS = 'h3'

    title = 'html'
    extensions = ('html', )

    @classmethod
    def export_set(cls, dataset):
        """HTML representation of a Dataset."""

        table = ET.Element('table')
        if dataset.headers is not None:
            head = ET.Element('thead')
            tr = ET.Element('tr')
            for header in dataset.headers:
                th = ET.Element('th')
                th.text = str(header) if header is not None else ''
                tr.append(th)
            head.append(tr)
            table.append(head)

        body = ET.Element('tbody')
        for row in dataset:
            tr = ET.Element('tr')
            for item in row:
                td = ET.Element('td')
                td.text = str(item) if item is not None else ''
                tr.append(td)
            body.append(tr)
        table.append(body)

        return ET.tostring(table, method='html', encoding='unicode')

    @classmethod
    def export_book(cls, databook):
        """HTML representation of a Databook."""

        result = ''
        for i, dset in enumerate(databook._datasets):
            title = dset.title if dset.title else f'Set {i}'
            title_el = ET.Element(cls.BOOK_ENDINGS)
            title_el.text = title
            result += ET.tostring(title_el, method='html', encoding='unicode') + '\n'
            result += dset.html
            result += '\n'

        return result

    @classmethod
    def import_set(cls, dset, in_stream, table_id=None, structured_headers=False,
                   flatten=None, separator=' / '):
        """Returns dataset from HTML content."""

        dset.wipe()
        if structured_headers:
            cls._import_structured(
                dset, in_stream, table_id=table_id,
                flatten=flatten, separator=separator,
            )
            return
        parser = TablibHTMLParser(dset, table_id=table_id)
        parser.feed(in_stream.read())
        if not parser.table_found:
            if table_id:
                raise ValueError(f'No <table> found with id="{table_id}" in input HTML')
            else:
                raise ValueError('No <table> found in input HTML')

    @classmethod
    def export_associations(cls, dataset):
        """JSON manifest of the per-cell header associations collected by a
        ``structured_headers=True`` import."""

        result = getattr(dataset, '_html_structured_headers', None)
        if result is None:
            raise ValueError(
                'No structured header associations found; '
                'import with structured_headers=True first'
            )
        return json.dumps(result.associations, ensure_ascii=False, indent=2)

    @classmethod
    def _import_structured(cls, dset, in_stream, table_id, flatten, separator):
        if flatten not in (None, 'first', 'last', 'join'):
            raise ValueError(
                f"Invalid flatten strategy {flatten!r}; "
                "expected one of 'first', 'last', 'join'"
            )
        parser = _StructuredTableParser(table_id=table_id)
        parser.feed(in_stream.read())
        if not parser.table_found:
            if table_id:
                raise ValueError(f'No <table> found with id="{table_id}" in input HTML')
            else:
                raise ValueError('No <table> found in input HTML')

        rows = parser.rows
        grid, width = _build_grid(rows)
        all_cells = [cell for row in rows for cell in row.cells]
        id_map = {cell.cell_id: cell for cell in all_cells if cell.cell_id}

        associations = []
        column_paths = {}
        for cell in (c for c in all_cells if c.section != 'thead'):
            if cell.headers_attr:
                sources = _resolve_explicit_headers(cell, id_map)
                via = 'explicit'
                col_cells = [c for c in sources if _is_col_header(c)]
                row_cells = [c for c in sources if not _is_col_header(c)]
            else:
                col_cells, row_cells = _infer_headers(cell, all_cells)
                via = 'inferred'
                sources = col_cells + row_cells
            column_paths.setdefault(cell.x, []).append(tuple(col_cells))
            associations.append({
                'row': cell.y,
                'col': cell.x,
                'tag': cell.tag,
                'id': cell.cell_id,
                'text': cell.text,
                'headers': [
                    {
                        'id': src.cell_id,
                        'text': src.text,
                        'row': src.y,
                        'col': src.x,
                        'via': via,
                    }
                    for src in col_cells + row_cells
                ],
                'path': [src.text for src in col_cells],
            })

        uniform = all(
            len({tuple(id(c) for c in path) for path in paths}) == 1 and bool(paths[0])
            for paths in column_paths.values()
        )
        if not uniform and flatten is None:
            raise StructuredHeaderError(
                'Cannot form a uniform header path for every column; '
                'per-cell associations were kept. Re-import with '
                "flatten='first', flatten='last' or flatten='join' to "
                'choose how to flatten them.',
                associations=associations,
            )

        headers = []
        for x in range(width):
            paths = column_paths.get(x)
            if not paths:
                path = _column_headers_for(x, all_cells)
            elif uniform:
                path = paths[0]
            else:
                path = _flatten_column_paths(paths, flatten)
            headers.append(separator.join(c.text for c in path))
        dset.headers = headers

        for y, row in enumerate(rows):
            if row.section == 'thead':
                continue
            values = []
            for x in range(width):
                cell = grid.get((x, y))
                if cell is not None and cell.x == x and cell.y == y:
                    values.append(cell.text)
                else:
                    values.append('')
            dset.append(values)

        dset._html_structured_headers = StructuredHeadersResult(associations, width)


class TablibHTMLParser(HTMLParser):
    def __init__(self, dataset, *args, table_id=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.dset = dataset
        self.table_id = table_id
        self.table_found = False
        self.table_open = False
        self.thead_open = False
        self.cell_open = False
        self.headers = []
        self.current_row = []
        self.current_data = ''

    def handle_starttag(self, tag, attrs):
        if (
            tag == 'table' and not self.table_found and
            (not self.table_id or dict(attrs).get('id') == self.table_id)
        ):
            self.table_open = True
            self.table_found = True
        elif self.table_open:
            if tag == 'thead':
                self.thead_open = True
            elif tag in ['td', 'th']:
                self.cell_open = True

    def handle_endtag(self, tag):
        if not self.table_open:
            return
        if tag == 'table':
            self.table_open = False
        elif tag == 'thead':
            self.thead_open = False
            self.dset.headers = self.headers
        elif tag == 'tr' and self.current_row:
            self.dset.append(self.current_row)
            self.current_row = []
        elif tag in ['td', 'th']:
            if self.thead_open:
                self.headers.append(self.current_data)
            else:
                self.current_row.append(self.current_data)
            self.cell_open = False
            self.current_data = ''

    def handle_data(self, data):
        if self.cell_open:
            self.current_data += data


def _parse_span(value, default, allow_zero):
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    if number < 0 or (number == 0 and not allow_zero):
        return default
    return number


class _StructCell:
    """A table cell captured for the structured grid."""

    def __init__(self, tag, attrs, pos, section):
        self.tag = tag
        self.pos = pos
        self.section = section
        attrs = dict(attrs)
        self.cell_id = attrs.get('id')
        self.headers_attr = tuple(attrs.get('headers', '').split())
        self.scope = attrs.get('scope', '').lower()
        if self.scope not in ('row', 'col', 'rowgroup', 'colgroup'):
            self.scope = ''
        self.rowspan = _parse_span(attrs.get('rowspan'), default=1, allow_zero=True)
        self.colspan = _parse_span(attrs.get('colspan'), default=1, allow_zero=False)
        self.text_parts = []
        # Placement in the logical grid, filled by _build_grid.
        self.x = self.y = self.w = self.h = None

    @property
    def text(self):
        return ''.join(self.text_parts).strip()


class _StructRow:
    def __init__(self, section, group, pos):
        self.section = section
        self.group = group
        self.pos = pos
        self.cells = []


class _StructuredTableParser(HTMLParser):
    """Captures the first matching table as logical rows of cells.

    Nested tables are tracked by depth so their rows/cells never leak into
    the outer table; their text content is dropped entirely.
    """

    SECTION_TAGS = ('thead', 'tbody', 'tfoot')

    def __init__(self, table_id=None):
        super().__init__()
        self.table_id = table_id
        self.table_found = False
        self.table_open = False
        self.depth = 0
        self.section = None
        self.group_counter = 0
        self.current_group = None
        self.implicit_group = None
        self.rows = []
        self.current_row = None
        self.current_cell = None

    def _start_row(self):
        if self.section is None:
            if self.implicit_group is None:
                self.group_counter += 1
                self.implicit_group = self.group_counter
            group = self.implicit_group
        else:
            group = self.current_group
        self.current_row = _StructRow(self.section or 'tbody', group, self.getpos())

    def _close_cell(self):
        if self.current_cell is not None:
            if self.current_row is None:
                self._start_row()
            self.current_row.cells.append(self.current_cell)
            self.current_cell = None

    def _close_row(self):
        if self.current_row is not None:
            self.rows.append(self.current_row)
            self.current_row = None

    def handle_starttag(self, tag, attrs):
        if tag == 'table':
            if self.table_open:
                self.depth += 1
            elif not self.table_found:
                if self.table_id is None or dict(attrs).get('id') == self.table_id:
                    self.table_open = True
                    self.table_found = True
                    self.depth = 1
            return
        if not self.table_open or self.depth != 1:
            return
        if tag in self.SECTION_TAGS:
            self._close_cell()
            self._close_row()
            self.section = tag
            self.group_counter += 1
            self.current_group = self.group_counter
            self.implicit_group = None
        elif tag == 'tr':
            self._close_cell()
            self._close_row()
            self._start_row()
        elif tag in ('td', 'th'):
            self._close_cell()
            if self.current_row is None:
                self._start_row()
            self.current_cell = _StructCell(
                tag, attrs, self.getpos(), self.section or 'tbody',
            )

    def handle_endtag(self, tag):
        if not self.table_open:
            return
        if tag == 'table':
            self.depth -= 1
            if self.depth == 0:
                self._close_cell()
                self._close_row()
                self.table_open = False
            return
        if self.depth != 1:
            return
        if tag in self.SECTION_TAGS:
            self._close_cell()
            self._close_row()
            self.section = None
            self.implicit_group = None
        elif tag == 'tr':
            self._close_cell()
            self._close_row()
        elif tag in ('td', 'th'):
            self._close_cell()

    def handle_data(self, data):
        if self.table_open and self.depth == 1 and self.current_cell is not None:
            self.current_cell.text_parts.append(data)


def _build_grid(rows):
    """Place cells on a logical grid honoring rowspan/colspan.

    ``rowspan=0`` spans to the end of the cell's row group (thead/tbody/
    tfoot or the implicit tbody). Overlapping spans raise with positions.
    Returns ``(grid, width)`` where grid maps ``(x, y)`` to the owning cell.
    """
    grid = {}
    for y, row in enumerate(rows):
        x = 0
        for cell in row.cells:
            while (x, y) in grid:
                x += 1
            height = cell.rowspan
            if height == 0:
                height = 1 + sum(
                    1 for later in rows[y + 1:] if later.group == row.group
                )
            for dy in range(height):
                for dx in range(cell.colspan):
                    slot = (x + dx, y + dy)
                    if slot in grid:
                        other = grid[slot]
                        raise StructuredHeaderError(
                            f'Conflicting cell spans: <{cell.tag}> at line '
                            f'{cell.pos[0]}, column {cell.pos[1]} overlaps '
                            f'<{other.tag}> at line {other.pos[0]}, column '
                            f'{other.pos[1]}'
                        )
                    grid[slot] = cell
            cell.x, cell.y, cell.w, cell.h = x, y, cell.colspan, height
            x += cell.colspan
    width = max((x for x, _y in grid), default=-1) + 1
    return grid, width


def _is_col_header(cell):
    if cell.tag != 'th':
        return False
    if cell.scope in ('col', 'colgroup'):
        return True
    if cell.scope in ('row', 'rowgroup'):
        return False
    return cell.section == 'thead'


def _is_row_header(cell):
    return (
        cell.tag == 'th'
        and cell.scope not in ('col', 'colgroup')
        and cell.section != 'thead'
    )


def _resolve_explicit_headers(cell, id_map):
    """Resolve a cell's headers= id references, following chained references.

    Missing targets and reference cycles raise with source positions.
    """
    resolved = []
    seen = set()

    def visit(holder, ref_id, chain):
        target = id_map.get(ref_id)
        if target is None:
            raise StructuredHeaderError(
                f'headers= reference "{ref_id}" not found (referenced by '
                f'<{holder.tag}> at line {holder.pos[0]}, column {holder.pos[1]})'
            )
        if target in chain:
            cycle = ' -> '.join(
                f'"{c.cell_id or c.tag}" (line {c.pos[0]}, column {c.pos[1]})'
                for c in chain[1:] + [target]
            )
            raise StructuredHeaderError(f'Cyclic headers= reference: {cycle}')
        if target in seen:
            return
        seen.add(target)
        resolved.append(target)
        for sub_ref in target.headers_attr:
            visit(target, sub_ref, chain + [target])

    for ref_id in cell.headers_attr:
        visit(cell, ref_id, [cell])
    return resolved


def _infer_headers(cell, all_cells):
    """Positional inference: column headers above the cell, row headers to
    its left. Identity comes from grid placement, never from text."""
    col_headers = []
    row_headers = []
    for other in all_cells:
        if other is cell:
            continue
        if (
            _is_col_header(other)
            and other.x <= cell.x < other.x + other.w
            and other.y + other.h <= cell.y
        ):
            col_headers.append(other)
        elif (
            _is_row_header(other)
            and other.y <= cell.y < other.y + other.h
            and other.x + other.w <= cell.x
        ):
            row_headers.append(other)
    col_headers.sort(key=lambda c: (c.y, c.x))
    row_headers.sort(key=lambda c: (c.x, c.y))
    return col_headers, row_headers


def _column_headers_for(x, all_cells):
    cells = [
        c for c in all_cells
        if _is_col_header(c) and c.x <= x < c.x + c.w
    ]
    cells.sort(key=lambda c: (c.y, c.x))
    return tuple(cells)


def _flatten_column_paths(paths, flatten):
    if flatten == 'first':
        return paths[0]
    if flatten == 'last':
        return paths[-1]
    union = []
    for path in paths:
        for cell in path:
            if cell not in union:
                union.append(cell)
    return tuple(union)
