"""Summarize measured generation traces, without manufacturing success rates."""
import argparse
import json
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('trace', type=Path)
parser.add_argument('--warmup', type=int, default=10)
args = parser.parse_args()
records = [json.loads(line) for line in args.trace.read_text().splitlines() if line.strip()]
records = records[args.warmup:]
if not records:
    raise SystemExit('No measured records remain after warmup')
result = {'queries': len(records), 'excluded_warmup': args.warmup,
          'latency_scope': 'chunk_decode_ms includes preprocessing, VLM and all codec calls; excludes outer observation queue and controller'}
for key in ('tokens', 'blocks', 'vlm_forwards', 'prefix_decoder_calls', 'prefix_decoder_ms', 'generation_ms', 'final_decoder_ms', 'chunk_decode_ms'):
    values = np.array([row[key] for row in records], dtype=float)
    result[key] = dict(mean=float(values.mean()), p50=float(np.quantile(values, .5)),
                       p95=float(np.quantile(values, .95)))
print(json.dumps(result, indent=2))
