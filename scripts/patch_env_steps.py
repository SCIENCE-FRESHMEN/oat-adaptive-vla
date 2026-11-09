"""Keep success resets under evaluator control and log actual environment actions."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
p = ROOT / 'vendor/praxis_eval/envs/libero/env.py'
s = p.read_text()
if '# OAT_EXPLICIT_SUCCESS_RESET' not in s:
    s = s.replace('import json\n', 'import json\nimport time\n')
    old = '        _ = kwargs\n        gym.Env.reset(self, seed=seed)'
    assert s.count(old) == 1
    s = s.replace(old, '''        _ = kwargs
        if hasattr(self, "_oat_flush_env_steps"):
            self._oat_flush_env_steps()
        self._oat_initial_state = int(self.init_state_id)
        self._oat_actions = []
        self._oat_environment_ms = []
        gym.Env.reset(self, seed=seed)''')
    old = '    def _format_raw_obs(self, raw_obs: Mapping[str, Any]) -> dict[str, Any]:'
    assert s.count(old) == 1
    methods = '''    # OAT_EXPLICIT_SUCCESS_RESET
    def step(self, action):
        if action.ndim != 1:
            raise ValueError(f"Expected one controller action, got {action.shape}")
        started = time.perf_counter()
        raw_obs, reward, done, info = self._env.step(action)
        success = bool(self._env.check_success())
        terminated = bool(done or success)
        info.update(task=self.task, task_id=self.task_id, done=done, is_success=success)
        if terminated:
            info["final_info"] = dict(task=self.task, task_id=self.task_id,
                                      done=bool(done), is_success=success)
        observation = self._format_raw_obs(raw_obs)
        if os.environ.get("OAT_ENV_STEPS"):
            self._oat_actions.append(action.tolist())
            self._oat_environment_ms.append(1000 * (time.perf_counter() - started))
            if terminated or len(self._oat_actions) >= self._max_episode_steps:
                self._oat_flush_env_steps()
        # The evaluator owns the next reset. Upstream LeRobot calls reset()
        # here on success, consuming an extra paired initial state.
        return observation, reward, terminated, False, info

    def _oat_flush_env_steps(self):
        actions = getattr(self, "_oat_actions", [])
        path = os.environ.get("OAT_ENV_STEPS")
        if actions and path:
            record = dict(task_id=int(self.task_id), initial_state=int(self._oat_initial_state),
                          actual_actions=actions, environment_step_ms=self._oat_environment_ms)
            with open(path, "a") as output:
                output.write(json.dumps(record, allow_nan=False) + "\\n")
        self._oat_actions = []
        self._oat_environment_ms = []

    def close(self):
        self._oat_flush_env_steps()
        return super().close()

'''
    s = s.replace(old, methods + old)
    p.write_text(s)
# Some async workers are terminated without invoking the wrapped env's close.
# Flush completed capped episodes before the external TimeLimit returns done.
s = p.read_text()
old = '            if terminated:\n                self._oat_flush_env_steps()'
new = '            if terminated or len(self._oat_actions) >= self._max_episode_steps:\n                self._oat_flush_env_steps()'
if old in s:
    assert s.count(old) == 1
    p.write_text(s.replace(old, new))
print('Explicit success reset and actual environment action recording installed')
