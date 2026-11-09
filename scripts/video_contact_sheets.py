"""Sample saved rollout videos for visual failure review; never runs a policy."""
import argparse
import json
from pathlib import Path

import av
from PIL import Image, ImageDraw


def contact_sheet(video, destination, title):
    with av.open(str(video)) as container:
        frames = [frame.to_image() for frame in container.decode(video=0)]
    if not frames:
        raise RuntimeError(f'Empty video: {video}')
    indices = sorted({round(i * (len(frames) - 1) / 7) for i in range(8)})
    sheet = Image.new('RGB', (896, 32 + 2 * 246), 'white')
    draw = ImageDraw.Draw(sheet)
    draw.text((5, 8), title, fill='black')
    for cell, index in enumerate(indices):
        frame = frames[index]
        frame.thumbnail((224, 224))
        x, y = (cell % 4) * 224, 32 + (cell // 4) * 246
        sheet.paste(frame, (x, y))
        draw.text((x + 3, y + 225), f'video frame {index}/{len(frames)-1}', fill='black')
    sheet.save(destination, quality=83)
    return dict(decoded_frames=len(frames), sampled_video_frame_indices=indices)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episodes', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--failures-only', action='store_true')
    parser.add_argument('--max-videos', type=int, default=20)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in args.episodes.read_text().splitlines() if line.strip()]
    index = []
    for episode in rows:
        if args.failures_only and episode['success']:
            continue
        for video in episode.get('video_paths', []):
            if len(index) >= args.max_videos:
                break
            task, state = episode['task_id'], episode['initial_state']
            output = args.output / f'{task}-{state}-{len(index)}.jpg'
            metadata = contact_sheet(Path(video), output,
                f'task {task}, state {state}, success={episode["success"]}: {episode["instruction"]}')
            index.append(dict(episode=episode, video=video, sheet=str(output.resolve()), **metadata))
    (args.output / 'index.json').write_text(json.dumps(index, indent=2) + '\n')
    print('SAVED_CONTACT_SHEETS', len(index), flush=True)


if __name__ == '__main__':
    main()
