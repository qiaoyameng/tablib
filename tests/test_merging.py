"""Tests for column-identity merging (tablib.merging)."""

import datetime as dt
from decimal import Decimal

import pytest

import tablib
from tablib import merging
from tablib.exceptions import HeadersNeeded
from tablib.merging import (
    INTERSECTION,
    MISSING,
    UNION,
    MergePlanError,
    MergePlanStale,
    merge,
    plan_merge,
)


def _ds(headers, *rows, title=None):
    dataset = tablib.Dataset(headers=headers, title=title)
    for row in rows:
        dataset.append(row)
    return dataset


# --------------------------------------------------------------------------
# Contrast with positional stacking
# --------------------------------------------------------------------------

def test_merge_differs_from_stack_for_shuffled_same_width():
    """stack() aligns by position; merge() aligns by header identity."""
    left = _ds(('name', 'age'), ('John', 30))
    right = _ds(('age', 'name'), (25, 'Mary'))

    stacked = left.stack(right)
    # Positional stacking blindly joins the columns: ages land under 'name'.
    assert stacked.dict == [
        {'name': 'John', 'age': 30},
        {'name': 25, 'age': 'Mary'},
    ]

    merged = merge([left, right])
    assert merged.headers == ['name', 'age']
    assert merged.dict == [
        {'name': 'John', 'age': 30},
        {'name': 'Mary', 'age': 25},
    ]


def test_merge_differs_from_stack_with_extra_column():
    left = _ds(('name', 'age'), ('John', 30))
    right = _ds(('name', 'age', 'city'), ('Mary', 25, 'Rome'))

    with pytest.raises(tablib.InvalidDimensions):
        left.stack(right)

    plan = plan_merge([left, right], fill={'city': 'unknown'})
    assert plan.result_headers == ['name', 'age', 'city']
    result = plan.apply()
    assert result.dict == [
        {'name': 'John', 'age': 30, 'city': 'unknown'},
        {'name': 'Mary', 'age': 25, 'city': 'Rome'},
    ]


def test_merge_same_named_column_with_different_types():
    """Same header, int on one side and float on the other: lossless widen."""
    left = _ds(('name', 'score'), ('John', 1))
    right = _ds(('name', 'score'), ('Mary', 1.5))

    plan = plan_merge([left, right])
    score = plan.columns[1]
    assert not score.rejected
    assert score.target_kind == 'float'
    assert plan.apply().dict == [
        {'name': 'John', 'score': 1.0},
        {'name': 'Mary', 'score': 1.5},
    ]


# --------------------------------------------------------------------------
# Plan reviewability
# --------------------------------------------------------------------------

def test_plan_exposes_column_order_sources_kinds_and_conversions():
    left = _ds(('a', 'b'), (1, 'x'))
    right = _ds(('b', 'c'), ('y', True))
    plan = plan_merge([left, right])

    assert plan.ok
    assert plan.result_headers == ['a', 'b', 'c']
    a_column, b_column, c_column = plan.columns
    assert a_column.sources == {0: 0, 1: None}
    assert b_column.sources == {0: 1, 1: 0}
    assert c_column.sources == {0: None, 1: 1}
    assert b_column.target_kind == 'text'
    assert c_column.target_kind == 'bool'

    summary = plan.to_dict()
    assert summary['headers'] == ['a', 'b', 'c']
    assert summary['rejections'] == {}
    # conversions for bool cells promoted into wider columns
    assert a_column.order == 0 and c_column.order == 2


def test_plan_decimal_float_is_rejected_by_default():
    left = _ds(('amount',), (Decimal('1.10'),))
    right = _ds(('amount',), (1.1,))

    plan = plan_merge([left, right])
    assert not plan.ok
    assert 'amount' in plan.rejections
    reason = plan.rejections['amount'][0]
    assert 'lossy' in reason
    with pytest.raises(MergePlanError):
        plan.apply()
    with pytest.raises(MergePlanError):
        merge([left, right])


@pytest.mark.parametrize('target,expected', [
    ('decimal', [Decimal('1.10'), Decimal('1.1')]),
    ('float', [1.10, 1.1]),
])
def test_plan_decimal_float_caller_decides(target, expected):
    left = _ds(('amount',), (Decimal('1.10'),))
    right = _ds(('amount',), (1.1,))
    result = merge(
        [left, right],
        coercions={('decimal', 'float'): target},
    )
    assert result['amount'] == expected


def test_incompatible_types_are_rejected_even_with_coercion():
    left = _ds(('v',), (1,))
    right = _ds(('v',), (dt.date(2024, 1, 1),))
    plan = plan_merge([left, right])
    assert not plan.ok
    assert 'incompatible' in plan.rejections['v'][0]

    # No coercion can bridge incompatible families: invalid targets fail at
    # planning time, while an uncoerced plan records the rejection.
    with pytest.raises(MergePlanError):
        plan_merge([left, right], coercions={('int', 'date'): 'int'})
    assert 'incompatible' in plan_merge([left, right]).rejections['v'][0]


def test_invalid_coercion_target_rejected():
    # A target neither side can reach (int cannot hold a float losslessly)
    # is rejected at planning time.
    with pytest.raises(MergePlanError):
        plan_merge(
            [_ds(('v',), (Decimal('1'),)), _ds(('v',), (1.0,))],
            coercions={('decimal', 'float'): 'int'},
        )


def test_lossy_pair_can_be_steered_to_shared_wider_text():
    left = _ds(('v',), (Decimal('1.20'),))
    right = _ds(('v',), (3.4,))
    result = merge(
        [left, right],
        coercions={('decimal', 'float'): 'text'},
    )
    assert result['v'] == ['1.20', '3.4']


# --------------------------------------------------------------------------
# Type reconciliation coverage
# --------------------------------------------------------------------------

@pytest.mark.parametrize('left_value,right_value,target', [
    (True, 1, 'int'),
    (True, 1.0, 'float'),
    (False, Decimal('0'), 'decimal'),
    (3, 1.5, 'float'),
    (3, Decimal('3'), 'decimal'),
    (3, 'three', 'text'),
    (1.5, '1.5', 'text'),
    (Decimal('1.5'), '1.5', 'text'),
    (dt.date(2024, 1, 1), '2024', 'text'),
    (dt.datetime(2024, 1, 1, 8, 0), 'now', 'text'),
    (dt.time(9, 30), 'late', 'text'),
])
def test_lossless_default_widening(left_value, right_value, target):
    left = _ds(('v',), (left_value,))
    right = _ds(('v',), (right_value,))
    plan = plan_merge([left, right])
    assert plan.ok, plan.rejections
    assert plan.columns[0].target_kind == target
    result = plan.apply()
    assert result.height == 2
    assert all(value is not None for value in result['v'])


def test_bool_int_float_mixed_inside_one_source_column():
    left = _ds(('v',), (True,), (3,))
    right = _ds(('v',), (1.5,))
    result = merge([left, right])
    assert result['v'] == [1.0, 3.0, 1.5]


def test_date_datetime_requires_caller_decision_in_both_directions():
    day = _ds(('when',), (dt.date(2024, 1, 2),))
    moment = _ds(('when',), (dt.datetime(2024, 1, 3, 10, 0),))

    assert not plan_merge([day, moment]).ok

    widened = merge([day, moment], coercions={('date', 'datetime'): 'datetime'})
    assert widened['when'] == [
        dt.datetime(2024, 1, 2),
        dt.datetime(2024, 1, 3, 10, 0),
    ]

    narrowed = merge([day, moment], coercions={('date', 'datetime'): 'date'})
    assert narrowed['when'] == [dt.date(2024, 1, 2), dt.date(2024, 1, 3)]
    assert all(type(value) is dt.date for value in narrowed['when'])


def test_unsupported_value_type_rejected():
    left = _ds(('v',), ([1, 2],))
    right = _ds(('v',), (1,))
    plan = plan_merge([left, right])
    assert not plan.ok
    assert 'unsupported type' in plan.rejections['v'][0]


# --------------------------------------------------------------------------
# Union / intersection and fills
# --------------------------------------------------------------------------

def test_intersection_keeps_only_shared_columns():
    left = _ds(('a', 'b'), (1, 'x'))
    right = _ds(('b', 'c'), ('y', 2))
    plan = plan_merge([left, right], how=INTERSECTION)
    assert plan.result_headers == ['b']
    assert plan.apply().dict == [{'b': 'x'}, {'b': 'y'}]


def test_invalid_how_rejected():
    left = _ds(('a',), (1,))
    right = _ds(('a',), (2,))
    with pytest.raises(MergePlanError):
        plan_merge([left, right], how='sideways')


def test_missing_fill_is_distinct_from_existing_none():
    left = _ds(('a', 'b'), (1, None))
    right = _ds(('a', 'c'), (2, 'x'))
    result = merge([left, right], fill='NA')
    assert list(result[0]) == [1, None, 'NA']
    assert list(result[1]) == [2, 'NA', 'x']


def test_fill_mapping_and_callable():
    left = _ds(('a',), (1,))
    right = _ds(('a', 'b', 'c'), (2, 'y', 'z'))

    by_name = merge([left, right], fill={'b': 'B', 'c': 'C'})
    assert list(by_name[0]) == [1, 'B', 'C']

    plan = plan_merge(
        [left, right],
        fill=lambda column: f'fill-{column.name}',
    )
    result = plan.apply()
    assert list(result[0]) == [1, 'fill-b', 'fill-c']


def test_apply_fill_override():
    left = _ds(('a',), (1,))
    right = _ds(('a', 'b'), (2, 'x'))
    plan = plan_merge([left, right], fill='default')
    assert plan.apply(fill='override')['b'] == ['override', 'x']


def test_missing_sentinel_roundtrips():
    assert repr(MISSING) == 'MISSING'
    left = _ds(('a',), (1,))
    right = _ds(('a', 'b'), (2, None))
    # None already present in a shared column is left untouched.
    assert merge([left, right])['b'] == [None, None]


def test_mutable_fill_is_deep_copied_per_cell():
    left = _ds(('a', 'b'), (1, 'x'))
    right = _ds(('a', 'c'), (2, 'y'))
    result = merge([left, right], fill={'b': ['v'], 'c': ['v']})
    result._data[0][2].append('only-c')
    assert result._data[1][1] == ['v']


# --------------------------------------------------------------------------
# Duplicate headers
# --------------------------------------------------------------------------

def test_duplicate_headers_require_explicit_positional_mapping():
    left = _ds(('name', 'name'), ('John', 'Johnny'))
    right = _ds(('name', 'nick'), ('Mary', 'M'))

    with pytest.raises(MergePlanError, match='explicitly name every position'):
        plan_merge([left, right])

    plan = plan_merge(
        [left, right],
        mappings={0: {'name': 0, 'nick': 1}},
    )
    assert plan.result_headers == ['name', 'nick']
    assert plan.apply().dict == [
        {'name': 'John', 'nick': 'Johnny'},
        {'name': 'Mary', 'nick': 'M'},
    ]


def test_duplicate_mapping_must_cover_every_ambiguous_position():
    left = _ds(('name', 'name'), ('a', 'b'))
    right = _ds(('x',), ('c',))
    with pytest.raises(MergePlanError):
        plan_merge([left, right], mappings={0: {'name': 0}})


def test_mapping_rejected_for_unique_headers():
    left = _ds(('a', 'b'), (1, 2))
    right = _ds(('a',), (3,))
    with pytest.raises(MergePlanError, match='unique'):
        plan_merge([left, right], mappings={0: {'a': 0}})


# --------------------------------------------------------------------------
# Structural checks, tags and isolation
# --------------------------------------------------------------------------

def test_apply_rejects_structural_change():
    left = _ds(('a',), (1,))
    right = _ds(('a',), (2,))
    plan = plan_merge([left, right])

    right.append_col((3,), header='b')
    with pytest.raises(MergePlanStale):
        plan.apply()

    right2 = _ds(('a', 'b'), (2, 9))
    with pytest.raises(MergePlanStale):
        plan.apply([left, right2])


def test_row_tags_are_preserved():
    left = _ds(('a',), (1,))
    left.append((2,), tags=('keep',))
    right = _ds(('a',), (3,))
    right.append((4,), tags=('other',))
    result = merge([left, right])
    assert [row.tags for row in result._data] == [[], ['keep'], [], ['other']]


def test_result_edit_does_not_pollute_sources():
    left = _ds(('a', 'b'), (1, 'x'))
    right = _ds(('a', 'b'), (2, 'y'))
    result = merge([left, right])
    result[0] = (99, 'MUTATED')
    result._data[1][1] = 'ALSO'
    assert left.dict == [{'a': 1, 'b': 'x'}]
    assert right.dict == [{'a': 2, 'b': 'y'}]


def test_source_edit_after_plan_does_not_pollute_planned_result_structure():
    left = _ds(('a',), (1,))
    right = _ds(('a',), (2,))
    plan = plan_merge([left, right])
    left.append((3,))
    with pytest.raises(MergePlanStale):
        plan.apply()


def test_filled_values_are_deep_copied_from_source_template():
    # The same fill template feeds multiple cells without aliasing.
    template = ['mutable']
    source = _ds(('v',), (1,))
    other = _ds(('v', 'extra'), (2, 'present'))
    result = merge([source, other], fill={'extra': template})
    # cell of the row that actually has 'extra' is its own string value
    assert result._data[1][1] == 'present'
    # filled cell is a deep copy of the template, not the template itself
    assert result._data[0][1] == ['mutable']
    result._data[0][1].append('changed-in-result')
    assert template == ['mutable']


def test_row_objects_are_not_shared_with_sources():
    left = _ds(('a',), (1,))
    right = _ds(('a',), (2,))
    result = merge([left, right])
    assert all(row not in left._data for row in result._data)
    assert all(row not in right._data for row in result._data)


def test_dataset_methods_match_module_functions():
    left = _ds(('name', 'age'), ('John', 30))
    right = _ds(('age', 'name'), (25, 'Mary'))
    plan = left.plan_merge(right)
    assert plan.apply().dict == left.merge(right).dict


def test_merge_requires_headers_and_two_datasets():
    headerless = tablib.Dataset()
    headerless.append((1,))
    headed = _ds(('a',), (2,))
    with pytest.raises(HeadersNeeded):
        plan_merge([headerless, headed])

    with pytest.raises(MergePlanError):
        plan_merge([headed])

    with pytest.raises(TypeError):
        plan_merge([headed, 'not-a-dataset'])


def test_three_way_union_preserves_first_seen_order():
    first = _ds(('a', 'b'), (1, 'x'))
    second = _ds(('c', 'b'), (True, 'y'))
    third = _ds(('b', 'd'), ('z', 2.0))
    result = merge([first, second, third], fill='-')
    assert result.headers == ['a', 'b', 'c', 'd']
    assert result.dict == [
        {'a': 1, 'b': 'x', 'c': '-', 'd': '-'},
        {'a': '-', 'b': 'y', 'c': True, 'd': '-'},
        {'a': '-', 'b': 'z', 'c': '-', 'd': 2.0},
    ]


def test_empty_datasets_participate_without_data():
    empty = tablib.Dataset(headers=('a', 'b'))
    other = _ds(('a', 'c'), (1, 'v'))
    plan = plan_merge([empty, other])
    assert plan.ok
    assert plan.apply().height == 1
