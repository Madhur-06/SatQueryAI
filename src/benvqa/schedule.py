"""Warmup-Stable-Decay learning rate as a stateless function of the step.

``planned_images`` (conversations) sets the planned number of optimizer steps. The decay
covers the final ``decay_fraction`` of it. Raising ``planned_images`` between sessions
just moves the decay later. ``--decay-now`` pins the decay start to the resume step via
``ScheduleState`` (saved in checkpoints), and the decay length is capped to what fits in
the session. NaN recovery halves ``lr_scale``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from benvqa.config import Config


@dataclass
class ScheduleState:
    decay_start: int | None = None
    decay_steps: int | None = None
    lr_scale: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)


def conversations_per_step(cfg: Config, world_size: int) -> int:
    return cfg.train.micro_batch * cfg.train.grad_accum * world_size


def planned_steps(cfg: Config, world_size: int) -> int:
    return max(1, math.ceil(cfg.train.planned_images / conversations_per_step(cfg, world_size)))


def lr_at(step: int, cfg: Config, world_size: int, state: ScheduleState) -> float:
    s = cfg.train.schedule
    base = cfg.train.lr * state.lr_scale
    if step < s.warmup_steps:
        return base * (step + 1) / s.warmup_steps
    if s.type == "constant":
        return base
    total = planned_steps(cfg, world_size)
    decay_steps = state.decay_steps or max(1, round(s.decay_fraction * total))
    decay_start = state.decay_start if state.decay_start is not None else total - decay_steps
    if step < decay_start:
        return base
    frac = min(1.0, (step - decay_start) / decay_steps)
    return base * (1.0 - frac * (1.0 - s.min_lr_ratio))


def decay_now(step: int, cfg: Config, world_size: int, steps_fit_in_session: int) -> ScheduleState:
    total = planned_steps(cfg, world_size)
    want = max(1, round(cfg.train.schedule.decay_fraction * total))
    return ScheduleState(decay_start=step, decay_steps=max(1, min(want, steps_fit_in_session)))
