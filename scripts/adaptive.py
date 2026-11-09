"""Batch-one OAT prefix stability stopping; thresholds use physical action units.

Environment variables are experiment controls, never training hyperparameters.
OAT_THRESHOLD unset means the original fixed-budget generation path.
OAT_TRACE optionally records synchronized generation timing for every method.
"""
import json
import math
import os
import random
import time
from pathlib import Path

import torch


def stable(previous, current, *, execute, position_scale, rotation_scale,
           threshold, gripper_guard=True):
    if not torch.isfinite(current).all() or not torch.isfinite(previous).all():
        return False, float('inf')
    a, b = previous[:, :execute], current[:, :execute]
    distance = ((a[..., :3] - b[..., :3]).norm(dim=-1).mean() / position_scale
                + (a[..., 3:6] - b[..., 3:6]).norm(dim=-1).mean() / rotation_scale)
    grip_ok = not gripper_guard or bool(((a[..., 6] > 0) == (b[..., 6] > 0)).all())
    value = float(distance.item())
    return grip_ok and value <= threshold, value


class PrefixStopper:
    def __init__(self, policy, batch_size, eos_supervision):
        raw = os.environ.get('OAT_THRESHOLD')
        self.enabled = raw is not None
        self.trace = os.environ.get('OAT_TRACE')
        self.policy = policy
        self.previous = None
        self.calls = 0
        self.distance = None
        self.reason = 'budget'
        self.decoder_ms = 0.0
        self.threshold = float(raw) if raw is not None else 0.0
        self.position_scale = float(os.environ.get('OAT_POSITION_SCALE', '1'))
        self.rotation_scale = float(os.environ.get('OAT_ROTATION_SCALE', '1'))
        self.guard = os.environ.get('OAT_GRIPPER_GUARD', '1') == '1'
        if self.enabled:
            if batch_size != 1 or eos_supervision or policy.action_tokenizer.config.name != 'oat':
                raise ValueError('Adaptive stopping requires batch=1, OAT and fixed-budget mode')
            if not math.isfinite(self.threshold) or self.threshold < 0:
                raise ValueError('OAT_THRESHOLD must be finite and nonnegative')
            if not all(math.isfinite(v) and v > 0 for v in (self.position_scale, self.rotation_scale)):
                raise ValueError('Action scales must be finite and positive')
        self.device = next(policy.parameters()).device if (self.trace or self.enabled) else None
        self.sync()
        self.started = time.perf_counter()

    def sync(self):
        if (self.trace or self.enabled) and self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    def observe(self, blocks, covered):
        if not self.enabled:
            return False
        # Decode only at complete registered blocks; the caller provides those.
        self.sync()
        started = time.perf_counter()
        tokens = torch.cat(blocks, dim=1)
        actions = self.policy.action_tokenizer.detokenize(tokens).detach()
        self.sync()
        self.decoder_ms += 1000 * (time.perf_counter() - started)
        self.calls += 1
        stop = False
        if self.previous is not None:
            stop, self.distance = stable(
                self.previous, actions, execute=int(self.policy.config.n_action_steps),
                position_scale=self.position_scale, rotation_scale=self.rotation_scale,
                threshold=self.threshold, gripper_guard=self.guard)
        self.previous = actions
        if stop:
            self.reason = 'stable_prefix'
        return stop

    def finish(self, token_ids, blocks):
        if not self.trace:
            return
        self.sync()
        record = dict(tokens=len(token_ids[0]), blocks=blocks, vlm_forwards=blocks,
                      prefix_decoder_calls=self.calls, final_decoder_calls=1,
                      prefix_decoder_ms=self.decoder_ms, reason=self.reason,
                      distance=self.distance if self.distance is None or math.isfinite(self.distance) else None,
                      threshold=self.threshold if self.enabled else None,
                      generation_ms=1000 * (time.perf_counter() - self.started))
        # generation_ms includes the cached generation loop and stop checks,
        # excluding earlier VLM input preparation, final decoding and controller.
        self.policy._oat_query_record = record


def random_budget(policy, maximum):
    path = os.environ.get('OAT_RANDOM_DISTRIBUTION')
    if not path:
        return maximum
    if not hasattr(policy, '_oat_random_rng'):
        distribution = json.loads(Path(path).read_text())
        policy._oat_random_choices = [int(k) for k in distribution]
        policy._oat_random_weights = [float(v) for v in distribution.values()]
        boundaries = set()
        total = 0
        for size in policy._resolve_block_schedule():
            total += size
            boundaries.add(total)
        if not set(policy._oat_random_choices) <= boundaries:
            raise ValueError('Random budgets must be complete block boundaries')
        if any(v < 0 or not math.isfinite(v) for v in policy._oat_random_weights) or sum(policy._oat_random_weights) <= 0:
            raise ValueError('Invalid random budget weights')
        if max(policy._oat_random_choices) > maximum:
            raise ValueError('Random budget exceeds policy decode budget')
        policy._oat_random_rng = random.Random(int(os.environ.get('OAT_RANDOM_SEED', '20261102')))
    return policy._oat_random_rng.choices(policy._oat_random_choices, weights=policy._oat_random_weights, k=1)[0]


def query_start(policy, obs=None):
    if not os.environ.get('OAT_TRACE'):
        return None
    device = next(policy.parameters()).device
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    if obs is not None:
        policy._oat_debug_obs = obs
    return time.perf_counter()


def query_finish(policy, started, final_started, actions=None):
    if started is None:
        return
    device = next(policy.parameters()).device
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    ended = time.perf_counter()
    record = dict(policy._oat_query_record)
    record['chunk_decode_ms'] = 1000 * (ended - started)
    record['final_decoder_ms'] = 1000 * (ended - final_started)
    record['peak_allocated_bytes'] = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None
    context = json.loads(os.environ.get('OAT_QUERY_CONTEXT', '{}'))
    record.update(context)
    if actions is not None:
        record['planned_actions'] = actions[0, :int(policy.config.n_action_steps)].detach().float().cpu().tolist()
    path = Path(os.environ['OAT_TRACE'])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        f.write(json.dumps(record, allow_nan=False) + '\n')
    if context and hasattr(policy, '_oat_debug_obs'):
        # Capture outside the timed region, once per episode, for failure review.
        sample = path.parent / 'input-observations' / f"{context['task_id']}-{context['initial_state']}.pt"
        if not sample.exists():
            sample.parent.mkdir(parents=True, exist_ok=True)
            torch.save({key: value.detach().cpu() if torch.is_tensor(value) else value
                        for key, value in policy._oat_debug_obs.items()}, sample)
