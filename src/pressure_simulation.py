import math

from src.gv_dynamics import GVDynamics


class PressureSimulation:
    """
    GV-CST toy pressure simulator.

    Effective energy:
        E_eff = E_raw * C * GV

    Behavior modes:
    - gradual ramp: usually damping / recovery
    - hard step overflow: can collapse
    """

    def __init__(
        self,
        beta=1.2,
        gamma=0.65,
        dim_limit=11,
        base_amp=0.10,
        freq=1.0,
        entropy_drag=0.08,
    ):
        self.dim_limit = dim_limit
        self.base_amp = base_amp
        self.freq = freq
        self.phase = 0.0
        self.time = 0.0

        self.dyn = GVDynamics(beta=beta, gamma=gamma, entropy_drag=entropy_drag)

    def reset(self):
        self.phase = 0.0
        self.time = 0.0
        self.dyn.reset(gv=1.0, entropy=0.0)

    def constraint_factor(self, target_dim):
        overflow = max(0, target_dim - self.dim_limit)

        if overflow == 0:
            c = 1.0
        else:
            # Harder dimensional overflow suppresses persistence
            c = max(0.0, 1.0 - 0.35 * overflow)

        return c, overflow

    def step(self, dt, amp, target_dim, injection):
        self.time += dt
        self.phase += 2 * math.pi * self.freq * dt

        value = amp * math.sin(self.phase)
        raw_energy = 0.5 * (amp ** 2) * (self.freq ** 2)

        c, overflow = self.constraint_factor(target_dim)
        entropy = self.dyn.update_entropy(amp=amp, constraint=c, overflow=overflow, dt=dt)
        gv = self.dyn.step(constraint=c, injection=injection, dt=dt)

        effective_energy = raw_energy * c * gv

        return {
            "time": self.time,
            "value": value,
            "amp": amp,
            "target_dim": target_dim,
            "overflow": overflow,
            "constraint": c,
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
