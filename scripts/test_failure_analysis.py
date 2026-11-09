"""Completion gates cannot accept missing reviews or mismatched episode evidence."""
import json
from pathlib import Path
import tempfile
import unittest
from failure_analysis import validate_reviews


class ReviewTest(unittest.TestCase):
    def fixture(self, root):
        evidence = []
        for method, success in [('full', True), ('adaptive', False)]:
            folder = root / f'outputs/formal-eval/test-42-{method}'
            folder.mkdir(parents=True)
            video = folder / 'fixture.mp4'
            video.write_bytes(b'synthetic test fixture; not experimental evidence')
            sheet = folder / 'fixture.jpg'
            sheet.write_bytes(b'synthetic test fixture')
            row = dict(task_id=0, initial_state=10, success=success, video_paths=[str(video)])
            (folder / 'episodes.jsonl').write_text(json.dumps(row) + '\n')
            evidence.append(dict(method=method, success=success, video=str(video), contact_sheet=str(sheet),
                                 decoded_frames=8, sampled_video_frame_indices=[0, 7]))
        (root / 'query.json').write_text('{}')
        return {'42': dict(training_seed=42, review_status='sampled_contact_sheets_visually_reviewed',
                           reviewed_pairs=1, reviewed_videos=2, limitations=['synthetic fixture'], method='fixture',
                           selection='fixture', query_evidence='query.json', cases=[dict(task_id=0, initial_state=10,
                           category='adaptive_only_failure', observed_frames='fixture',
                           interpretation_with_uncertainty='fixture', evidence=evidence)])}

    def test_missing_seed_review_remains_pending(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            reviews = self.fixture(root)
            self.assertEqual(validate_reviews(root, reviews), [43, 44])

    def test_outcome_mismatch_rejects_review(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            reviews = self.fixture(root)
            reviews['42']['cases'][0]['category'] = 'adaptive_recovered'
            with self.assertRaisesRegex(ValueError, 'category'):
                validate_reviews(root, reviews)

    def test_contact_sheet_generation_does_not_count_as_review(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            reviews = self.fixture(root)
            reviews['42']['review_status'] = 'ready_for_visual_review'
            self.assertEqual(validate_reviews(root, reviews), [42, 43, 44])

    def test_missing_media_rejects_recorded_review(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            reviews = self.fixture(root)
            Path(reviews['42']['cases'][0]['evidence'][0]['contact_sheet']).unlink()
            with self.assertRaisesRegex(ValueError, 'Missing reviewed media'):
                validate_reviews(root, reviews)


if __name__ == '__main__':
    unittest.main()
