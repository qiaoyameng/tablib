"""Tests for read-only snapshots, block-sharing views and COW branches."""

import unittest

import tablib
from tablib.exceptions import (
    HeadersNeeded,
    SnapshotMutableCellError,
    SnapshotReadOnlyError,
)
from tablib.snapshot import (
    DEFAULT_BLOCK_SIZE,
    DatasetView,
    Snapshot,
    SnapshotDataset,
    StorageBlock,
    StorageStats,
    is_immutable_scalar,
)


class SnapshotFixture:
    headers = ('first_name', 'last_name', 'gpa')

    def make_dataset(self):
        dataset = tablib.Dataset(headers=self.headers, title='Founders')
        dataset.append(('John', 'Adams', 90), tags=('admin', 'east'))
        dataset.append(('George', 'Washington', 67), tags=('east',))
        dataset.append(('Thomas', 'Jefferson', 50), tags=('west',))
        dataset.append(('James', 'Monroe', 73), tags=('admin', 'west'))
        return dataset


class SnapshotCreationTests(SnapshotFixture, unittest.TestCase):
    def setUp(self):
        StorageStats.reset()
        self.dataset = self.make_dataset()

    def test_snapshot_captures_structure_and_values(self):
        snapshot = self.dataset.snapshot()
        self.assertIsInstance(snapshot, Snapshot)
        self.assertEqual(snapshot.headers, list(self.headers))
        self.assertEqual(snapshot.title, 'Founders')
        self.assertEqual(snapshot.height, 4)
        self.assertEqual(snapshot.width, 3)
        self.assertEqual(snapshot[:], self.dataset[:])
        self.assertEqual(snapshot.row_tags(0), ['admin', 'east'])
        self.assertEqual(snapshot.row_tags(2), ['west'])

    def test_immutable_cells_are_shared_by_identity(self):
        snapshot = self.dataset.snapshot()
        for row_index in range(self.dataset.height):
            for col_index in range(self.dataset.width):
                self.assertIs(
                    snapshot[row_index][col_index],
                    self.dataset[row_index][col_index],
                )

    def test_blocks_group_rows_according_to_block_size(self):
        snapshot = self.dataset.snapshot(block_size=2)
        self.assertEqual(snapshot.blocks, (1, 2))
        self.assertEqual(snapshot.block_for_row(0), 1)
        self.assertEqual(snapshot.block_for_row(1), 1)
        self.assertEqual(snapshot.block_for_row(2), 2)
        self.assertEqual(StorageStats.as_dict()['blocks_created'], 2)

    def test_default_block_size_constant_is_positive(self):
        self.assertGreaterEqual(DEFAULT_BLOCK_SIZE, 1)

    def test_empty_dataset_snapshot(self):
        empty = tablib.Dataset(headers=('a', 'b'))
        snapshot = empty.snapshot()
        self.assertEqual(snapshot.height, 0)
        self.assertEqual(snapshot.width, 2)
        self.assertEqual(snapshot.export('csv'), 'a,b\r\n')
        self.assertEqual(snapshot.export('json'), '[]')


class SnapshotIsolationTests(SnapshotFixture, unittest.TestCase):
    def setUp(self):
        self.dataset = self.make_dataset()
        self.snapshot = self.dataset.snapshot()

    def test_source_append_does_not_affect_snapshot(self):
        self.dataset.append(('New', 'Guy', 1), tags=('east',))
        self.assertEqual(self.snapshot.height, 4)
        self.assertEqual(self.dataset.height, 5)

    def test_source_delete_row_does_not_affect_snapshot(self):
        del self.dataset[0]
        self.assertEqual(self.snapshot[0], ('John', 'Adams', 90))
        self.assertEqual(self.snapshot.height, 4)

    def test_source_delete_column_does_not_affect_snapshot(self):
        del self.dataset['gpa']
        self.assertEqual(self.snapshot.width, 3)
        self.assertEqual(self.snapshot.headers, list(self.headers))
        self.assertEqual(self.snapshot.get_col('gpa'), [90, 67, 50, 73])

    def test_source_sort_does_not_affect_snapshot_order(self):
        original_order = self.snapshot[:]
        self.dataset.sort('gpa', reverse=True)
        self.assertEqual(self.snapshot[:], original_order)

    def test_source_tag_change_does_not_affect_snapshot_tags(self):
        self.dataset._data[0].tags.append('superuser')
        self.assertEqual(self.snapshot.row_tags(0), ['admin', 'east'])

    def test_source_header_and_title_change_does_not_affect_snapshot(self):
        self.dataset.headers = ('given', 'family', 'score')
        self.dataset.title = 'Renamed'
        self.assertEqual(self.snapshot.headers, list(self.headers))
        self.assertEqual(self.snapshot.title, 'Founders')

    def test_formatter_binding_is_part_of_snapshot_structure(self):
        self.dataset.add_formatter('gpa', lambda value: value + 1)
        snapshot = self.dataset.snapshot()
        self.dataset.wipe()
        self.assertEqual(
            snapshot.dict[0],
            {'first_name': 'John', 'last_name': 'Adams', 'gpa': 91},
        )


class MutableCellPolicyTests(unittest.TestCase):
    def test_immutable_scalar_predicate(self):
        for value in (None, True, 1, 1.5, 'text', b'bytes', (1, 2)):
            self.assertTrue(is_immutable_scalar(value))
        for value in ([1, 2], {'k': 'v'}, {1, 2}, object()):
            self.assertFalse(is_immutable_scalar(value))

    def test_mutable_cell_is_rejected_by_default(self):
        dataset = tablib.Dataset(headers=('key', 'payload'))
        dataset.append(('a', [1, 2, 3]))
        with self.assertRaises(SnapshotMutableCellError):
            dataset.snapshot()

    def test_mutable_cell_copy_mode_deep_copies(self):
        dataset = tablib.Dataset(headers=('key', 'payload'))
        dataset.append(('a', [1, 2, {'nested': 1}]))
        snapshot = dataset.snapshot(on_mutable='copy')
        dataset[0][1].append(4)
        dataset[0][1][2]['nested'] = 99
        self.assertEqual(snapshot[0][1], [1, 2, {'nested': 1}])

    def test_invalid_policy_raises_value_error(self):
        dataset = tablib.Dataset(headers=('a',))
        dataset.append(([1],))
        with self.assertRaises(ValueError):
            dataset.snapshot(on_mutable='nope')

    def test_branch_deep_copies_mutable_cells_and_stays_isolated(self):
        dataset = tablib.Dataset(headers=('key', 'payload'))
        dataset.append(('a', {'x': 1}))
        snapshot = dataset.snapshot(on_mutable='copy')
        branch = snapshot.to_branch()
        self.assertIsNot(branch[0][1], snapshot[0][1])
        branch[0][1]['x'] = 42
        self.assertEqual(snapshot[0][1], {'x': 1})

    def test_branch_rejects_mutable_cells_when_requested(self):
        dataset = tablib.Dataset(headers=('key', 'payload'))
        dataset.append(('a', {'x': 1}))
        snapshot = dataset.snapshot(on_mutable='copy')
        with self.assertRaises(SnapshotMutableCellError):
            snapshot.to_branch(on_mutable='reject')


class ReadOnlyEnforcementTests(SnapshotFixture, unittest.TestCase):
    def setUp(self):
        self.snapshot = self.make_dataset().snapshot()
        self.view = self.snapshot.view(rows=[0, 1])

    def assert_read_only(self, target):
        mutating_calls = (
            lambda: target.append(('x',)),
            lambda: target.insert(0, ('x',)),
            lambda: target.wipe(),
            lambda: target.insert_col(0, [1]),
            lambda: target.add_formatter(0, str),
            lambda: target.remove_duplicates(),
            lambda: target.sort(0),
        )
        for call in mutating_calls:
            with self.subTest(call=call):
                with self.assertRaises(SnapshotReadOnlyError):
                    call()
        with self.assertRaises(SnapshotReadOnlyError):
            target[0] = ('x',)
        with self.assertRaises(SnapshotReadOnlyError):
            del target[0]
        with self.assertRaises(SnapshotReadOnlyError):
            target.headers = ['x']
        with self.assertRaises(SnapshotReadOnlyError):
            target.title = 'x'

    def test_snapshot_is_read_only(self):
        self.assert_read_only(self.snapshot)
        self.assertTrue(self.snapshot.is_read_only)

    def test_view_is_read_only(self):
        self.assert_read_only(self.view)
        self.assertIsInstance(self.view, DatasetView)

    def test_unknown_attribute_still_raises_attribute_error(self):
        with self.assertRaises(AttributeError):
            _ = self.snapshot.does_not_exist


class ViewTests(SnapshotFixture, unittest.TestCase):
    def setUp(self):
        StorageStats.reset()
        self.dataset = self.make_dataset()
        self.snapshot = self.dataset.snapshot(block_size=2)

    def test_view_projects_rows_and_columns(self):
        view = self.snapshot.view(rows=[0, 2], cols=['first_name', 'gpa'])
        self.assertEqual(view.headers, ['first_name', 'gpa'])
        self.assertEqual(view.height, 2)
        self.assertEqual(view.width, 2)
        self.assertEqual(view[:], [('John', 90), ('Thomas', 50)])
        self.assertEqual(view['gpa'], [90, 50])

    def test_view_shares_cells_and_blocks(self):
        view = self.snapshot.view(rows=[0, 1, 2], cols=['first_name', 'gpa'])
        # Creating views allocates no new blocks.
        self.assertEqual(StorageStats.as_dict()['blocks_created'], 2)
        self.assertEqual(view.block_for_row(0), self.snapshot.block_for_row(0))
        self.assertIs(view[0][0], self.snapshot[0][0])
        self.assertIs(view[1][1], self.snapshot[1][2])

    def test_overlapping_views_share_storage_blocks(self):
        left = self.snapshot.view(rows=[0, 1, 2], cols=['first_name', 'gpa'])
        right = self.snapshot.view(rows=[1, 2, 3], cols=['last_name', 'gpa'])
        # Direct block-identity check for the overlapping rows.
        self.assertEqual(
            left.block_for_row(1),
            right.block_for_row(0),
        )
        self.assertEqual(
            left.block_for_row(2),
            right.block_for_row(1),
        )
        self.assertEqual(StorageStats.as_dict()['blocks_created'], 2)

    def test_nested_view_keeps_referencing_snapshot_storage(self):
        view = self.snapshot.view(rows=[0, 1, 2], cols=['first_name', 'gpa'])
        nested = view.view(rows=[1], cols=[1])
        self.assertEqual(nested[:], [(67,)])
        self.assertEqual(nested.block_for_row(0), self.snapshot.block_for_row(1))
        self.assertEqual(StorageStats.as_dict()['blocks_created'], 2)

    def test_view_filtering_by_tags(self):
        admins = self.snapshot.view(tags='admin')
        self.assertEqual(admins['first_name'], ['John', 'James'])
        intersection = self.snapshot.view(rows=[1, 2, 3], tags='west')
        self.assertEqual(intersection['first_name'], ['Thomas', 'James'])

    def test_view_rejects_out_of_bounds_indices(self):
        with self.assertRaises(IndexError):
            self.snapshot.view(rows=[99])
        with self.assertRaises(IndexError):
            self.snapshot.view(cols=[99])
        with self.assertRaises(KeyError):
            self.snapshot.view(cols=['missing'])

    def test_view_without_headers_uses_positional_columns(self):
        dataset = tablib.Dataset()
        dataset.append((1, 'a'))
        dataset.append((2, 'b'))
        snapshot = dataset.snapshot()
        view = snapshot.view(rows=[1], cols=[1])
        self.assertIsNone(view.headers)
        self.assertEqual(view[:], [('b',)])

    def test_view_accepts_negative_indices(self):
        view = self.snapshot.view(rows=[-1], cols=[-1])
        self.assertEqual(view[:], [(73,)])
        self.assertEqual(view.block_for_row(0), self.snapshot.block_for_row(3))


class SnapshotExportTests(SnapshotFixture, unittest.TestCase):
    def setUp(self):
        self.dataset = self.make_dataset()
        self.dataset.add_formatter('gpa', lambda value: value * 2)
        self.snapshot = self.dataset.snapshot()
        self.view = self.snapshot.view(rows=[0, 3], cols=['first_name', 'gpa'])

    def test_csv_export_uses_snapshot_structure(self):
        expected = 'first_name,gpa\r\nJohn,180\r\nJames,146\r\n'
        self.assertEqual(self.view.export('csv'), expected)
        self.assertEqual(self.view.csv, expected)

    def test_json_export_uses_snapshot_structure(self):
        import json
        self.assertEqual(
            json.loads(self.view.export('json')),
            [
                {'first_name': 'John', 'gpa': 180},
                {'first_name': 'James', 'gpa': 146},
            ],
        )

    def test_full_snapshot_export_matches_dataset_export(self):
        plain = tablib.Dataset(headers=self.headers, title='Founders')
        plain.append(('John', 'Adams', 90), tags=('admin', 'east'))
        plain.append(('George', 'Washington', 67), tags=('east',))
        plain.append(('Thomas', 'Jefferson', 50), tags=('west',))
        plain.append(('James', 'Monroe', 73), tags=('admin', 'west'))
        snapshot = plain.snapshot()
        self.assertEqual(snapshot.export('csv'), plain.export('csv'))
        self.assertEqual(snapshot.export('json'), plain.export('json'))

    def test_export_does_not_mutate_snapshot(self):
        self.view.export('csv')
        self.view.export('json')
        self.assertEqual(self.view.height, 2)
        # Formatters only apply to packaged exports; raw rows stay unchanged.
        self.assertEqual(self.view[:], [('John', 90), ('James', 73)])


class BranchCopyOnWriteTests(SnapshotFixture, unittest.TestCase):
    def setUp(self):
        StorageStats.reset()
        self.dataset = self.make_dataset()
        self.snapshot = self.dataset.snapshot(block_size=2)

    def test_branch_is_ordinary_dataset_subclass(self):
        branch = self.snapshot.to_branch()
        self.assertIsInstance(branch, tablib.Dataset)
        self.assertIsInstance(branch, SnapshotDataset)
        self.assertFalse(branch.is_read_only)
        self.assertEqual(branch.headers, list(self.headers))
        self.assertEqual(branch[:], self.snapshot[:])

    def test_branch_starts_fully_shared(self):
        branch = self.snapshot.to_branch()
        self.assertEqual(branch.shared_row_count, self.snapshot.height)
        self.assertEqual(StorageStats.as_dict()['blocks_copied'], 0)

    def test_cell_write_copies_block_once_and_keeps_other_blocks_shared(self):
        branch = self.snapshot.to_branch()
        branch[(0, 0)] = 'Johnny'
        self.assertEqual(StorageStats.as_dict()['blocks_copied'], 1)
        # Second write to the same (now private) block must not copy again.
        branch[(1, 1)] = 'Carter'
        self.assertEqual(StorageStats.as_dict()['blocks_copied'], 1)
        # The untouched block is still shared by identity.
        self.assertTrue(branch.shares_block_with(self.snapshot))
        self.assertEqual(branch.block_for_row(2), self.snapshot.block_for_row(2))
        # Snapshot and source dataset remain unchanged.
        self.assertEqual(self.snapshot[0][0], 'John')
        self.assertEqual(self.dataset[0][0], 'John')

    def test_branch_mutation_never_affects_snapshot_cells(self):
        branch = self.snapshot.to_branch()
        original = self.snapshot[:]
        branch[(2, 2)] = 0
        del branch[0]
        branch.append(('New', 'Row', 1))
        self.assertEqual(self.snapshot[:], original)

    def test_row_delete_uses_copy_on_write(self):
        branch = self.snapshot.to_branch()
        branch_copy = self.snapshot.view(rows=[0, 1]).to_branch()
        StorageStats.reset()
        del branch_copy[0]
        self.assertEqual(StorageStats.as_dict()['blocks_copied'], 1)
        self.assertEqual(branch_copy.height, 1)
        # Unrelated branch still fully shared.
        self.assertEqual(branch.shared_row_count, self.snapshot.height)

    def test_appended_rows_are_private_and_copy_nothing(self):
        branch = self.snapshot.to_branch()
        StorageStats.reset()
        branch.append(('Private', 'Row', 1), tags=('east',))
        self.assertEqual(StorageStats.as_dict()['blocks_copied'], 0)
        self.assertIsNone(branch.block_for_row(branch.height - 1))
        self.assertEqual(branch.shared_row_count, self.snapshot.height)

    def test_middle_insertion_detaches_blocks_and_keeps_mapping_aligned(self):
        branch = self.snapshot.to_branch()
        StorageStats.reset()
        branch.insert(1, ('Inserted', 'Here', 1))
        self.assertEqual(StorageStats.as_dict()['blocks_copied'], 2)
        self.assertEqual(len(branch.blocks), branch.height)
        self.assertIsNone(branch.block_for_row(1))
        self.assertEqual(branch[1], ('Inserted', 'Here', 1))

    def test_append_after_middle_insertion_keeps_private_rows(self):
        branch = self.snapshot.to_branch()
        branch.insert(1, ('Inserted', 'Here', 1))
        branch.append(('Tail', 'End', 0))
        self.assertEqual(len(branch.blocks), branch.height)
        self.assertEqual(branch[-1], ('Tail', 'End', 0))

    def test_column_change_detaches_every_shared_block(self):
        branch = self.snapshot.to_branch()
        StorageStats.reset()
        del branch['gpa']
        self.assertEqual(StorageStats.as_dict()['blocks_copied'], 2)
        self.assertEqual(branch.width, 2)
        self.assertEqual(branch.shared_row_count, 0)
        self.assertEqual(self.snapshot.width, 3)

    def test_branch_row_tags_are_isolated(self):
        branch = self.snapshot.to_branch()
        branch.set_row_tags(0, ('superuser',))
        self.assertEqual(branch._data[0].tags, ['superuser'])
        self.assertEqual(self.snapshot.row_tags(0), ['admin', 'east'])

    def test_view_branch_materializes_only_projected_cells(self):
        view = self.snapshot.view(rows=[0, 2], cols=['first_name', 'gpa'])
        branch = view.to_branch()
        self.assertEqual(branch.headers, ['first_name', 'gpa'])
        self.assertEqual(branch[:], [('John', 90), ('Thomas', 50)])
        self.assertEqual(branch.shared_row_count, 2)
        branch[(0, 1)] = 99
        self.assertEqual(view[0], ('John', 90))


class OverlappingViewsModificationMatrixTests(SnapshotFixture, unittest.TestCase):
    """Overlapping views plus independent branches: only touched blocks split."""

    def setUp(self):
        StorageStats.reset()
        self.dataset = self.make_dataset()
        self.snapshot = self.dataset.snapshot(block_size=2)
        # Two overlapping views. Rows 1 and 2 appear in both; row 1 lives in
        # block 1, row 2 in block 2.
        self.view_a = self.snapshot.view(rows=[0, 1, 2], cols=['first_name', 'gpa'])
        self.view_b = self.snapshot.view(rows=[1, 2, 3], cols=['last_name', 'gpa'])

    def test_matrix_of_independent_branch_mutations(self):
        branch_a = self.view_a.to_branch()
        branch_b = self.view_b.to_branch()

        StorageStats.reset()
        # branch A touches snapshot row 0 (block 1 only).
        branch_a[(0, 0)] = 'A0'
        # branch B touches snapshot row 3 (block 2 only).
        branch_b[(2, 0)] = 'B3'

        self.assertEqual(StorageStats.as_dict()['blocks_copied'], 2)

        # Snapshot storage is untouched and still shared by both branches'
        # unmodified rows.
        self.assertEqual(self.snapshot[0][0], 'John')
        self.assertEqual(self.snapshot[3][1], 'Monroe')
        # branch A split block 1, so its row 2 (block 2) remains shared.
        self.assertEqual(branch_a.block_for_row(2), self.snapshot.block_for_row(2))
        # branch B split block 2, so its row 1 (block 1) remains shared.
        self.assertEqual(branch_b.block_for_row(0), self.snapshot.block_for_row(1))

    def test_branches_are_independent_despite_overlap(self):
        branch_a = self.view_a.to_branch()
        branch_b = self.view_b.to_branch()

        branch_a[(1, 0)] = 'RenamedInA'
        self.assertEqual(StorageStats.as_dict()['blocks_copied'], 1)

        # branch B's overlapping row 1 still reflects the snapshot value.
        self.assertEqual(branch_b[0][0], 'Washington')
        self.assertEqual(self.snapshot[1][0], 'George')

        # Mutating branch B afterwards splits the same source block again,
        # independently of branch A.
        branch_b[(0, 1)] = 670
        self.assertEqual(StorageStats.as_dict()['blocks_copied'], 2)
        self.assertEqual(branch_a[1][0], 'RenamedInA')

    def test_unmodified_overlap_keeps_identical_block_identities(self):
        before = (
            self.view_a.block_for_row(1),
            self.view_b.block_for_row(0),
        )
        branch_a = self.view_a.to_branch()
        branch_b = self.view_b.to_branch()
        self.assertEqual(branch_a.block_for_row(1), before[0])
        self.assertEqual(branch_b.block_for_row(0), before[1])
        self.assertEqual(before[0], before[1])

    def test_full_modification_matrix_values_and_storage(self):
        # Three overlapping views over the same two storage blocks.
        view_c = self.snapshot.view(rows=[0, 3], cols=['gpa'])
        branch_a = self.view_a.to_branch()
        branch_b = self.view_b.to_branch()
        branch_c = view_c.to_branch()

        StorageStats.reset()
        branch_a[(0, 1)] = 900       # touches block 1
        branch_b[(2, 1)] = 730       # touches block 2
        branch_c[(1, 0)] = 777       # touches block 2 (independent copy)

        # Two distinct source blocks were split; block 2 was copied once per
        # branch that touched it, giving three block copies in total.
        self.assertEqual(StorageStats.as_dict()['blocks_copied'], 3)

        # The snapshot matrix is exactly the original captured data.
        self.assertEqual(self.snapshot['gpa'], [90, 67, 50, 73])
        # Each branch reflects only its own modification.
        self.assertEqual(branch_a['gpa'], [900, 67, 50])
        self.assertEqual(branch_b['gpa'], [67, 50, 730])
        self.assertEqual(branch_c['gpa'], [90, 777])
        # The read-only views still project the untouched snapshot.
        self.assertEqual(self.view_a['gpa'], [90, 67, 50])
        self.assertEqual(self.view_b['gpa'], [67, 50, 73])
        # Unmodified rows keep sharing their original block identity.
        self.assertEqual(
            branch_a.block_for_row(2), self.snapshot.block_for_row(2)
        )


class StorageProofTests(SnapshotFixture, unittest.TestCase):
    def test_no_deep_copy_while_building_overlapping_views(self):
        StorageStats.reset()
        snapshot = self.make_dataset().snapshot(block_size=2)
        created_for_snapshot = StorageStats.as_dict()['blocks_created']
        snapshot.view(rows=[0, 1])
        snapshot.view(rows=[1, 2])
        snapshot.view(rows=[2, 3], cols=['gpa'])
        self.assertEqual(
            StorageStats.as_dict()['blocks_created'],
            created_for_snapshot,
        )

    def test_copy_count_matches_number_of_distinct_touched_blocks(self):
        StorageStats.reset()
        snapshot = self.make_dataset().snapshot(block_size=1)
        branch = snapshot.to_branch()
        for row_index in (0, 1, 3):
            branch[(row_index, 0)] = f'x{row_index}'
        self.assertEqual(StorageStats.as_dict()['blocks_copied'], 3)
        self.assertEqual(branch.shared_row_count, 1)

    def test_storage_block_is_immutable_container(self):
        block = StorageBlock([(1, 2), (3, 4)])
        self.assertEqual(block.block_id, StorageStats.as_dict()['blocks_created'])
        self.assertEqual(len(block), 2)
        with self.assertRaises(AttributeError):
            block.rows = ()

    def test_headers_needed_on_string_access_without_headers(self):
        dataset = tablib.Dataset()
        dataset.append((1, 2))
        snapshot = dataset.snapshot()
        with self.assertRaises(HeadersNeeded):
            _ = snapshot['anything']


if __name__ == '__main__':
    unittest.main()
