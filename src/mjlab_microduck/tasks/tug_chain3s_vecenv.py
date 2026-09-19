"""Self-play VecEnv flattener for the 3v3 tug chain (v9).

The mjlab env holds N matches of 6 ducks each (one physics world per match).
rsl_rl's VecEnv contract wants one row per AGENT, so this wrapper flattens
match × duck into N·6 rows:

  - obs:     (N, 6·61) from the inner ``ducks61`` obs term → (N·6, 61)
  - actions: (N·6, 14) from the policy → (N, 84) into the 6 per-duck
             joint_pos action terms (row order = CHAIN3S_DUCKS slot order,
             match-major: row = match·6 + slot)
  - rewards: (N, 6) per-duck terms from mdp.py → (N·6,)
  - dones:   match-level (any duck falls / leaves / NaN / time-out) broadcast
             to the match's 6 rows; the whole match resets together (a fallen
             duck loses the round for its team — keeps fall pressure ON,
             unlike v8's frozen dead-weight rule)

The inner env runs with ``auto_reset=False`` so terminal-step rewards are
computed from the true terminal state before the wrapper resets done matches
itself and blends post-reset obs into the done rows (standard auto-reset
semantics for the runner).

Reward scaling follows mjlab's RewardManager convention (value × weight ×
step_dt) so weights stay comparable with the v6–v8 runs, and per-term episode
sums are logged as ``Episode_Reward/<term>`` on match completion (the
AGENTS.md wandb discipline: every penalty term must read ≤ 0).
"""

import torch
from tensordict import TensorDict

from mjlab_microduck.tasks import mdp as microduck_mdp


class TugChainSelfPlayVecEnv:
    """rsl_rl-VecEnv-compatible flattening wrapper around ManagerBasedRlEnv."""

    DUCKS = microduck_mdp.CHAIN3S_DUCKS
    N_DUCKS = len(DUCKS)
    OBS_DIM = 61
    ACT_DIM = 14

    def __init__(self, env, spec: dict | None = None):
        # env: ManagerBasedRlEnv with num_envs = N matches.
        self.env = env
        self.num_matches = env.num_envs
        self.num_envs = self.num_matches * self.N_DUCKS
        self.num_actions = self.ACT_DIM
        self.device = env.device
        # The spec rides on the cfg when built directly; through the train CLI
        # (tyro rebuilds the cfg dataclass, dropping non-field attributes) the
        # runner passes it in explicitly.
        if spec is None:
            spec = env.cfg.chain3s_spec
        self._terms = spec["terms"]  # [(name, func, weight, params), ...]
        self._ar_stages = spec["action_rate_stages"]
        self._ep_sums = {
            name: torch.zeros(self.num_envs, device=self.device)
            for name, *_ in self._terms
        }
        self.env.reset()

    # ── VecEnv interface ────────────────────────────────────────────────────

    @property
    def unwrapped(self):
        return self.env

    @property
    def cfg(self):
        return self.env.cfg

    @property
    def max_episode_length(self) -> int:
        return self.env.max_episode_length

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self.env.episode_length_buf.repeat_interleave(self.N_DUCKS)

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor) -> None:
        # rsl_rl randomizes initial episode lengths per ROW; a match shares one
        # episode clock, so take each match's first row.
        self.env.episode_length_buf[:] = value.view(self.num_matches, self.N_DUCKS)[:, 0]

    def get_observations(self) -> TensorDict:
        return self._pack_obs(self.env.obs_buf)

    def reset(self) -> tuple[TensorDict, dict]:
        self.env.reset()
        for sums in self._ep_sums.values():
            sums.zero_()
        return self._pack_obs(self.env.obs_buf), {}

    def step(self, actions: torch.Tensor):
        env = self.env
        # Row = match·6 + slot (CHAIN3S_DUCKS order) → per-match action vector
        # (84,) whose 14-dim slices line up with the 6 action terms in cfg
        # insertion order (joint_pos, joint_pos_red_inner, ...).
        a = actions.to(self.device).reshape(self.num_matches, self.N_DUCKS * self.ACT_DIM)
        _, _, terminated, truncated, _ = env.step(a)

        obs_pre = self._pack_obs(env.obs_buf)  # terminal-state obs
        rew = self._compute_rewards()  # (N·6,) — from the pre-reset state

        done_matches = terminated | truncated
        rows_done = done_matches.repeat_interleave(self.N_DUCKS)
        log: dict = {}
        if done_matches.any():
            ids = done_matches.nonzero(as_tuple=False).squeeze(-1)
            for name, sums in self._ep_sums.items():
                log[f"Episode_Reward/{name}"] = sums[rows_done].mean().item()
                sums[rows_done] = 0.0
            env.reset(env_ids=ids)  # whole match resets together
            obs_post = self._pack_obs(env.obs_buf)
            obs = TensorDict(
                {
                    group: torch.where(rows_done[:, None], obs_post[group], obs_pre[group])
                    for group in ("actor", "critic")
                },
                batch_size=[self.num_envs],
            )
            # Inner managers (terminations, curricula) logged into extras["log"]
            # during _reset_idx — merge after the reset call.
            log.update(env.extras.get("log", {}))
        else:
            obs = obs_pre

        extras = {"time_outs": truncated.repeat_interleave(self.N_DUCKS)}
        # rsl_rl's logger aggregates keys from the FIRST ep_extras entry of an
        # iteration — only attach "log" on steps where a match actually ended
        # (otherwise an empty first dict would blank the whole iteration's
        # Episode_Reward table).
        if log:
            extras["log"] = log
        return obs, rew, rows_done.to(torch.long), extras

    def close(self) -> None:
        self.env.close()

    # ── internals ────────────────────────────────────────────────────────────

    def _pack_obs(self, obs_dict: dict) -> TensorDict:
        return TensorDict(
            {
                "actor": obs_dict["actor"].reshape(self.num_envs, self.OBS_DIM),
                "critic": obs_dict["critic"].reshape(self.num_envs, self.OBS_DIM),
            },
            batch_size=[self.num_envs],
        )

    def _action_rate_weight(self) -> float:
        step = self.env.common_step_counter
        w = self._ar_stages[0]["weight"]
        for stage in self._ar_stages:
            if step >= stage["step"]:
                w = stage["weight"]
        return w

    def _compute_rewards(self) -> torch.Tensor:
        env = self.env
        dt = env.step_dt
        total = torch.zeros(self.num_matches, self.N_DUCKS, device=self.device)
        for name, func, weight, params in self._terms:
            w = self._action_rate_weight() if name == "action_rate_l2" else weight
            if w == 0.0:
                continue
            v = func(env, **params)
            v = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0) * w * dt
            total += v
            self._ep_sums[name] += v.reshape(-1)
        return torch.nan_to_num(total, nan=0.0).reshape(-1)
