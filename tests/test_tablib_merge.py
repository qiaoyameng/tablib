"""Tests for column-identity Dataset reconciliation (``tablib.merge``)."""

import datetime as dt
import unittest
from decimal import Decimal

import tablib
from tablib import MergePlan, ReconcilePolicy, merge_sets, plan_merge
from tablib.exceptions import (
    AmbiguousColumns,
    HeadersRequired,
    InvalidIdentityMap,
    MergeRejected,
    PlanStale,
)


class MergeTestCase(unittest.TestCase):
    # --------------------------------------------------------------
    # Proof that identity merge differs from positional ``stack``
    # --------------------------------------------------------------

    def test_shuffled_same_width_differs_from_positional_stack(self):
        """Same columns, different order, same width.

        ``stack`` only checks width and concatenates by position, so values
        land under the wrong headers. The identity merge realigns rows.
        """

        hq = tablib.Dataset(headers=("name", "age"))
        hq.append(("Ada", 30))
        branch = tablib.Dataset(headers=("age", "name"))
        branch.append((41, "Lin"))

        # Positional stacking silently misaligns the data.
        stacked = hq.stack(branch)
        self.assertEqual(stacked.dict, [
            {"name": "Ada", "age": 30},
            {"name": 41, "age": "Lin"},
        ])

        merged = hq.merge(branch, how="union")
        self.assertEqual(merged.headers, ["name", "age"])
        self.assertEqual(merged.dict, [
            {"name": "Ada", "age": 30},
            {"name": "Lin", "age": 41},
        ])

    def test_same_name_different_types_default_is_lossless(self):
        """Same header, incompatible value types.

        ``stack`` ignores types entirely and forces the rows together. The
        identity merge reports the lossy Decimal/float combination as a
        rejection in the plan instead of coercing silently.
        """

        money = tablib.Dataset(headers=("amount",))
        money.append((Decimal("1.50"),))
        measured = tablib.Dataset(headers=("amount",))
        measured.append((2.25,))

        # Positional stacking merges regardless of the type mismatch.
        stacked = money.stack(measured)
        self.assertEqual(stacked["amount"], [Decimal("1.50"), 2.25])

        plan = money.plan_merge(measured, how="union")
        self.assertFalse(plan.is_ready())
        self.assertEqual(plan.columns[0].identity, "amount")
        self.assertIsNone(plan.columns[0].target_type)
        self.assertTrue(plan.rejected[0][1].startswith("lossy numeric"))
        with self.assertRaises(MergeRejected):
            plan.apply()

        # The caller decides: accept promotion to float, or keep Decimal.
        lossy = money.merge(measured, how="union", allow_lossy=True)
        self.assertEqual(lossy["amount"], [1.5, 2.25])
        self.assertEqual(
            {type(v) for v in lossy["amount"]}, {float}
        )

        exact = money.merge(
            measured, how="union",
            converters={"amount": lambda v: Decimal(str(v))},
        )
        self.assertEqual(exact["amount"], [Decimal("1.50"), Decimal("2.25")])

    # --------------------------------------------------------------
    # Union / intersection, ordering and extra columns
    # --------------------------------------------------------------

    def test_union_includes_extra_column_in_first_seen_order(self):
        left = tablib.Dataset(headers=("a", "b"))
        left.append((1, 2))
        right = tablib.Dataset(headers=("b", "c"))
        right.append((20, 30))

        plan = left.plan_merge(right, how="union")
        self.assertEqual(plan.headers, ["a", "b", "c"])
        merged = plan.apply()
        self.assertEqual(merged.dict, [
            {"a": 1, "b": 2, "c": None},
            {"a": None, "b": 20, "c": 30},
        ])

    def test_intersection_keeps_only_shared_columns(self):
        left = tablib.Dataset(headers=("a", "b"))
        left.append((1, 2))
        right = tablib.Dataset(headers=("b", "c"))
        right.append((20, 30))

        merged = left.merge(right)  # default is intersection
        self.assertEqual(merged.headers, ["b"])
        self.assertEqual(merged.dict, [{"b": 2}, {"b": 20}])

    def test_empty_intersection_is_flagged_in_plan_notes(self):
        left = tablib.Dataset(headers=("a",))
        left.append((1,))
        right = tablib.Dataset(headers=("z",))
        right.append((9,))

        plan = left.plan_merge(right)
        self.assertEqual(plan.headers, [])
        self.assertTrue(
            any("empty" in note for note in plan.notes)
        )

    def test_result_column_order_follows_first_dataset_then_new(self):
        first = tablib.Dataset(headers=("a", "b", "c"))
        second = tablib.Dataset(headers=("c", "d", "a"))
        third = tablib.Dataset(headers=("e", "b"))
        plan = first.plan_merge(second, third, how="union")
        self.assertEqual(plan.headers, ["a", "b", "c", "d", "e"])

    # --------------------------------------------------------------
    # Type reconciliation coverage
    # --------------------------------------------------------------

    def _single_column_sets(self, left_value, right_value, header="v"):
        left = tablib.Dataset(headers=(header,))
        left.append((left_value,))
        right = tablib.Dataset(headers=(header,))
        right.append((right_value,))
        return left, right

    def test_bool_int_merge_is_lossless_to_int(self):
        left, right = self._single_column_sets(True, 7)
        merged = left.merge(right, how="union")
        self.assertEqual(merged["v"], [1, 7])
        self.assertEqual({type(v) for v in merged["v"]}, {int})

    def test_int_decimal_merge_is_lossless_to_decimal(self):
        left, right = self._single_column_sets(3, Decimal("2.5"))
        plan = left.plan_merge(right, how="union")
        self.assertEqual(plan.columns[0].target_type, "decimal")
        self.assertNotIn("v", plan.lossy)
        self.assertEqual(plan.apply()["v"], [Decimal("3"), Decimal("2.5")])

    def test_float_decimal_requires_caller_decision(self):
        left, right = self._single_column_sets(1.0, Decimal("2"))
        self.assertTrue(left.plan_merge(right).rejected)
        plan = left.plan_merge(right, allow_lossy=True)
        self.assertEqual(plan.lossy, ["v"])
        self.assertEqual({type(v) for v in plan.apply()["v"]}, {float})

    def test_date_datetime_is_lossy(self):
        left, right = self._single_column_sets(
            dt.date(2020, 1, 1), dt.datetime(2021, 6, 7, 8, 9)
        )
        self.assertTrue(left.plan_merge(right).rejected)
        merged = left.merge(right, allow_lossy=True)
        self.assertEqual(merged["v"], [
            dt.datetime(2020, 1, 1, 0, 0),
            dt.datetime(2021, 6, 7, 8, 9),
        ])

    def test_time_passes_through_when_consistent(self):
        left, right = self._single_column_sets(
            dt.time(8, 30), dt.time(9, 0)
        )
        self.assertEqual(left.merge(right, how="union")["v"],
                         [dt.time(8, 30), dt.time(9, 0)])

    def test_text_combination_is_lossless_only_when_all_text(self):
        left, right = self._single_column_sets("a", "b")
        self.assertEqual(left.merge(right, how="union")["v"], ["a", "b"])

    def test_mixed_int_and_text_is_rejected_then_lossy(self):
        left, right = self._single_column_sets(1, "x")
        self.assertTrue(left.plan_merge(right).rejected)
        self.assertEqual(left.merge(right, allow_lossy=True)["v"],
                         ["1", "x"])

    def test_incompatible_families_are_rejected(self):
        left, right = self._single_column_sets(
            dt.time(1, 2), dt.date(2020, 1, 1)
        )
        plan = left.plan_merge(right, allow_lossy=True)
        self.assertTrue(plan.rejected)

    # --------------------------------------------------------------
    # Missing columns vs existing None, and fill rules
    # --------------------------------------------------------------

    def test_existing_none_is_distinct_from_missing_fill(self):
        left = tablib.Dataset(headers=("name", "note"))
        left.append(("Ada", None))
        right = tablib.Dataset(headers=("name", "age"))
        right.append(("Lin", 41))

        merged = left.merge(right, how="union", fill="N/A")
        # Existing None survives unchanged; only the missing cell is filled.
        self.assertEqual(merged.dict, [
            {"name": "Ada", "note": None, "age": "N/A"},
            {"name": "Lin", "note": "N/A", "age": 41},
        ])

    def test_callable_fill_is_invoked_per_cell(self):
        left = tablib.Dataset(headers=("a",))
        left.append((1,))
        right = tablib.Dataset(headers=("b",))
        right.append((2,))

        merged = left.merge(right, how="union", fill=list)
        first, second = merged.dict
        self.assertEqual(first["b"], [])
        self.assertEqual(second["a"], [])
        self.assertIsNot(first["b"], second["a"])

    def test_mutable_fill_is_not_shared_between_cells(self):
        sentinel = []
        left = tablib.Dataset(headers=("a",))
        left.append((1,))
        right = tablib.Dataset(headers=("b",))
        right.append((2,))
        merged = left.merge(right, how="union", fill=sentinel)
        first, second = merged.dict
        self.assertEqual(first["b"], [])
        self.assertIsNot(first["b"], sentinel)
        self.assertIsNot(first["b"], second["a"])

    # --------------------------------------------------------------
    # Duplicate headers require explicit positional identity maps
    # --------------------------------------------------------------

    def test_duplicate_headers_raise_without_explicit_map(self):
        duplicate = tablib.Dataset(headers=("x", "x"))
        duplicate.append((1, 2))
        other = tablib.Dataset(headers=("x", "y"))
        other.append((3, 4))

        with self.assertRaises(AmbiguousColumns):
            duplicate.plan_merge(other, how="union")

    def test_explicit_positional_identity_map_disambiguates(self):
        duplicate = tablib.Dataset(headers=("x", "x"))
        duplicate.append((1, 2))
        other = tablib.Dataset(headers=("x", "y"))
        other.append((3, 4))

        # Dict form: position -> identity.
        merged = duplicate.merge(
            other, how="union",
            identity_maps={0: {0: "x", 1: "y"}},
        )
        self.assertEqual(merged.headers, ["x", "y"])
        self.assertEqual(merged.dict, [
            {"x": 1, "y": 2},
            {"x": 3, "y": 4},
        ])

        # Sequence form: one identity per column position.
        plan = plan_merge(
            [duplicate, other], how="union",
            identity_maps={0: ("p", "q")},
        )
        self.assertEqual(plan.identities[0], ["p", "q"])

    def test_identity_map_must_cover_every_position(self):
        duplicate = tablib.Dataset(headers=("x", "x"))
        duplicate.append((1, 2))
        other = tablib.Dataset(headers=("x", "y"))
        other.append((3, 4))

        with self.assertRaises(InvalidIdentityMap):
            duplicate.plan_merge(
                other, how="union", identity_maps={0: {0: "a"}}
            )
        with self.assertRaises(InvalidIdentityMap):
            duplicate.plan_merge(
                other, how="union", identity_maps={0: ("only-one",)}
            )

    def test_headerless_dataset_needs_identity_map(self):
        headerless = tablib.Dataset()
        headerless.append((1, 2))
        other = tablib.Dataset(headers=("a", "b"))
        other.append((3, 4))

        with self.assertRaises(HeadersRequired):
            headerless.plan_merge(other)

        merged = headerless.merge(
            other, identity_maps={0: ("a", "b")}
        )
        self.assertEqual(merged.dict, [
            {"a": 1, "b": 2},
            {"a": 3, "b": 4},
        ])

    # --------------------------------------------------------------
    # Plan review, structure-change detection and row labels
    # --------------------------------------------------------------

    def test_plan_lists_order_conversions_and_rejections(self):
        left = tablib.Dataset(headers=("name", "amount"))
        left.append(("Ada", Decimal("1")))
        right = tablib.Dataset(headers=("amount", "name", "extra"))
        right.append((2.0, "Lin", True))

        plan = left.plan_merge(right, how="union")
        self.assertEqual(plan.headers, ["name", "amount", "extra"])
        by_identity = {column.identity: column for column in plan.columns}
        self.assertEqual(by_identity["name"].target_type, "text")
        self.assertIsNone(by_identity["amount"].target_type)
        self.assertTrue(by_identity["amount"].rejected)
        self.assertEqual(
            by_identity["extra"].observed_types, {1: frozenset({"bool"})}
        )
        self.assertIn("amount", plan.describe())
        self.assertEqual(
            repr(plan), "<MergePlan columns=3 1 rejected>"
        )

    def test_applying_rejected_plan_is_blocked(self):
        left, right = self._single_column_sets(1.0, Decimal("2"))
        plan = left.plan_merge(right)
        with self.assertRaises(MergeRejected):
            plan.apply()

    def test_structure_change_after_planning_is_detected(self):
        left = tablib.Dataset(headers=("a",))
        left.append((1,))
        right = tablib.Dataset(headers=("a", "b"))
        right.append((2, 3))

        renamed = left.plan_merge(right, how="union")
        right.headers = ("a", "c")
        with self.assertRaises(PlanStale):
            renamed.apply()

        resized = left.plan_merge(right, how="union",
                                  identity_maps={1: ["a", "c"]})
        right.append_col((9,), header="d")
        with self.assertRaises(PlanStale):
            resized.apply()

        changed_rows = left.plan_merge(right, how="union",
                                       identity_maps={1: ["a", "c", "d"]})
        right.append((7, 8, 9))
        with self.assertRaises(PlanStale):
            changed_rows.apply()

    def test_unchanged_plan_remains_valid(self):
        left = tablib.Dataset(headers=("a",))
        left.append((1,))
        right = tablib.Dataset(headers=("b",))
        right.append((2,))
        plan = left.plan_merge(right, how="union")
        # Reading values does not invalidate the plan.
        self.assertEqual(left["a"], [1])
        merged = plan.apply()
        self.assertEqual(merged.dict, [
            {"a": 1, "b": None},
            {"a": None, "b": 2},
        ])

    def test_row_tags_are_preserved(self):
        left = tablib.Dataset(headers=("a",))
        left.append((1,), tags=("hq",))
        right = tablib.Dataset(headers=("a",))
        right.append((2,), tags=("branch",))
        merged = left.merge(right)
        self.assertEqual([row.tags for row in merged._data],
                         [["hq"], ["branch"]])
        self.assertEqual(merged.filter("hq")[0], (1,))

    # --------------------------------------------------------------
    # Isolation
    # --------------------------------------------------------------

    def test_result_is_isolated_from_later_input_edits(self):
        left = tablib.Dataset(headers=("name", "items"))
        left.append(("Ada", [1, 2]))
        right = tablib.Dataset(headers=("name", "items"))
        right.append(("Lin", [3]))

        merged = left.merge(right, how="union")
        left[0][1].append(999)
        merged[1][1].append(888)  # result row for Lin

        self.assertEqual(left[0][1], [1, 2, 999])
        self.assertEqual(merged[0][1], [1, 2])
        self.assertEqual(merged[1][1], [3, 888])
        self.assertEqual(right[0][1], [3])

    def test_row_objects_are_not_shared(self):
        left = tablib.Dataset(headers=("a",))
        left.append((1,))
        right = tablib.Dataset(headers=("a",))
        right.append((2,))
        merged = left.merge(right)

        self.assertIsNot(merged._data[0], left._data[0])
        merged._data[0][0] = 100
        self.assertEqual(left[0][0], 1)

    # --------------------------------------------------------------
    # API surface and validation
    # --------------------------------------------------------------

    def test_module_level_helpers_and_policy(self):
        left = tablib.Dataset(headers=("a",))
        left.append((1,))
        right = tablib.Dataset(headers=("b",))
        right.append((2,))

        policy = ReconcilePolicy(how="union", fill=0)
        plan = plan_merge([left, right], policy=policy)
        self.assertIsInstance(plan, MergePlan)
        self.assertEqual(merge_sets([left, right], policy=policy).dict, [
            {"a": 1, "b": 0},
            {"a": 0, "b": 2},
        ])

    def test_invalid_how_is_rejected(self):
        with self.assertRaises(ValueError):
            ReconcilePolicy(how="sideways")

    def test_rejects_non_dataset_and_empty_input(self):
        good = tablib.Dataset(headers=("a",))
        good.append((1,))
        with self.assertRaises(TypeError):
            plan_merge([good, "not a dataset"])
        with self.assertRaises(ValueError):
            plan_merge([])

    def test_policy_object_and_kwargs_cannot_be_combined(self):
        left = tablib.Dataset(headers=("a",))
        left.append((1,))
        right = tablib.Dataset(headers=("a",))
        right.append((2,))
        with self.assertRaises(ValueError):
            left.plan_merge(right, policy=ReconcilePolicy(), fill=0)

    def test_merge_three_datasets_in_order(self):
        first = tablib.Dataset(headers=("a", "b"))
        first.append((1, 2))
        second = tablib.Dataset(headers=("b", "a"))
        second.append((4, 3))
        third = tablib.Dataset(headers=("c", "a"))
        third.append((6, 5))

        merged = first.merge(second, third, how="union")
        self.assertEqual(merged.headers, ["a", "b", "c"])
        self.assertEqual(merged.dict, [
            {"a": 1, "b": 2, "c": None},
            {"a": 3, "b": 4, "c": None},
            {"a": 5, "b": None, "c": 6},
        ])


if __name__ == "__main__":
    unittest.main()
