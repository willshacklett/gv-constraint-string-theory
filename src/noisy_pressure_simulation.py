import math
import random

from src.gv_dynamics import GVDynamics


class NoisyPressureSimulation:
    """
    GV-CST toy simulator with stochastic constraint noise.

    Effective energy:
        E_eff = E_raw * C_noisy * GV

    Where:
        C_noisy = clamp(C_base + noise)

    This lets us test whether GV still separates:
    - stable
    - damped
    - collapse

    under random perturbations in the constraint field.
    """

    def __init__(
        self,
        beta=1.2,
        gamma=0.65,
        dim_limit=11,
        base_amp=0.10,
        freq=1.0,
        entropy_drag=0.08,
        noise_std=0.05,
        seed=42,
    ):
        self.dim_limit = dim_limit
        self.base_amp = base_amp
        self.freq = freq
        self.phase = 0.0
        self.time = 0.0
        self.noise_std = noise_std
        self.rng = random.Random(seed)

        self.dyn = GVDynamics(beta=beta, gamma=gamma, entropy_drag=entropy_drag)

    def reset(self):
        self.phase = 0.0
        self.time = 0.0
        self.dyn.reset(gv=1.0, entropy=0.0)

    def constraint_factor(self, target_dim):
        overflow = max(0, target_dim - self.dim_limit)

        if overflow == 0:
            c_base = 1.0
        else:
            c_base = max(0.0, 1.0 - 0.35 * overflow)

        return c_base, overflow

    def noisy_constraint(self, c_base):
        noise = self.rng.gauss(0.0, self.noise_std)
        c_noisy = max(0.0, min(1.0, c_base + noise))
        return c_noisy, noise

    def step(self, dt, amp, target_dim, injection):
        self.time += dt
        self.phase += 2 * math.pi * self.freq * dt

        value = amp * math.sin(self.phase)
        raw_energy = 0.5 * (amp ** 2) * (self.freq ** 2)

        c_base, overflow = self.constraint_factor(target_dim)
        c_noisy, c_noise = self.noisy_constraint(c_base)

        entropy = self.dyn.update_entropy(
            amp=amp,
            constraint=c_noisy,
            overflow=overflow,
            dt=dt,
        )
        gv = self.dyn.step(
            constraint=c_noisy,
            injection=injection,
            dt=dt,
        )

        effective_energy = raw_energy * c_noisy * gv

        return {
            "time": self.time,
            "value": value,
            "amp": amp,
            "target_dim": target_dim,
            "overflow": overflow,
            "constraint_base": c_base,
            "constraint_noise": c_noise,
            "constraint_noisy": c_noisy,
            "gv": gv,
            "entropy": entropy,
            "raw_energy": raw_energy,
            "effective_energy": effective_energy,
            "injection": injection,
        }

    def run(self, steps, dt, amp_schedule, dim_schedule, injection_schedule):
        self.reset()
        history = []

        for i in range(steps):
            amp = amp_schedule(i, self.time)
            target_dim = dim_schedule(i, self.time)
            injection = injection_schedule(i, self.time)
            history.append(self.step(dt, amp, target_dim, injection))

        return history
