import json
from pathlib import Path
import tempfile
import unittest

from evidence import validate_evidence


class EvidenceTest(unittest.TestCase):
    def test_missing_video_or_wrong_pair_cannot_be_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'input-observations').mkdir()
            video = root / 'video.mp4'
            video.write_bytes(b'video-placeholder')
            rows = [dict(task_id=0, initial_state=i, success=False, length=12,
                         video_paths=[str(video)]) for i in [10, 11]]
            queries = [dict(task_id=0, initial_state=i, chunk_decode_ms=2,
                            generation_ms=1, final_decoder_ms=.5, tokens=4,
                            vlm_forwards=4, planned_actions=[[0.] * 7] * 8) for i in [10, 10, 11, 11]]
            environment = [dict(task_id=0, initial_state=i, actual_actions=[[0.] * 7] * 12,
                                environment_step_ms=[1.] * 12) for i in [10, 11]]
            for name, content in [('episodes', rows), ('resets', rows), ('queries', queries)]:
                (root / f'{name}.jsonl').write_text('\n'.join(map(json.dumps, content)))
            (root / 'environment-steps.jsonl').write_text('\n'.join(map(json.dumps, environment)))
            for i in [10, 11]:
                (root / 'input-observations' / f'0-{i}.pt').write_bytes(b'input-placeholder')
            self.assertEqual(validate_evidence(root, 2, 10, task_count=1, require_videos=True)['episodes'], 2)
            video.unlink()
            with self.assertRaisesRegex(RuntimeError, 'missing episode video'):
                validate_evidence(root, 2, 10, task_count=1, require_videos=True)
            rows[1]['initial_state'] = 10
            (root / 'resets.jsonl').write_text('\n'.join(map(json.dumps, rows)))
            with self.assertRaisesRegex(RuntimeError, 'incorrect/repeated actual resets'):
                validate_evidence(root, 2, 10, task_count=1)
            rows[1]['initial_state'] = 11
            (root / 'resets.jsonl').write_text('\n'.join(map(json.dumps, rows)))
            environment[0]['actual_actions'] = environment[0]['actual_actions'][:-1]
            (root / 'environment-steps.jsonl').write_text('\n'.join(map(json.dumps, environment)))
            with self.assertRaisesRegex(RuntimeError, 'actual action count'):
                validate_evidence(root, 2, 10, task_count=1)


if __name__ == '__main__':
    unittest.main()
