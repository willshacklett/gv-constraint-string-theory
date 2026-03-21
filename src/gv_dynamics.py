from dataclasses import dataclass


@dataclass
class GVState:
    gv: float = 1.0
    entropy: float = 0.0


class GVDynamics:
    """
    GV dynamics with a real instability / collapse regime.

    Core idea:
    - Strong constraint and low entropy support persistence.
    - Low constraint, rising entropy, and weak injection can push GV downward.
    - Once GV gets too low, collapse accelerates nonlinearly.

    This creates three broad behaviors:
    - damped: GV stays healthy
    - partial: GV dips under stress but recovers enough
    - collapse: GV falls through a critical floor and runs away downward
    """

    def __init__(
        self,
        beta=1.2,
        gamma=0.65,
        entropy_drag=0.08,
        collapse_gain=1.6,
        recovery_gain=0.9,
        critical_gv=0.45,
    ):
        self.beta = beta
        self.gamma = gamma
        self.entropy_drag = entropy_drag
        self.collapse_gain = collapse_gain
        self.recovery_gain = recovery_gain
        self.critical_gv = critical_gv
        self.state = GVState()

    def reset(self, gv=1.0, entropy=0.0):
        self.state.gv = gv
        self.state.entropy = entropy

    def step(self, constraint, injection, dt):
        gv = self.state.gv
        entropy = self.state.entropy

        # -----------------------------
        # Recovery / support term
        # -----------------------------
        # High constraint + injection can rebuild GV when it is below 1.
        recovery_term = (
            self.recovery_gain * constraint * (1.0 - gv)
            + self.gamma * injection
        )

        # -----------------------------
        # Baseline degradation
        # -----------------------------
        # Low constraint and entropy both drag GV downward.
        degradation_term = (
            self.beta * (1.0 - constraint)
            + self.entropy_drag * entropy
        )

        # -----------------------------
        # Nonlinear collapse acceleration
        # -----------------------------
        # Once GV drops below a critical threshold, it becomes harder to recover.
        if gv < self.critical_gv:
            collapse_term = self.collapse_gain * (self.critical_gv - gv) ** 2
        else:
            collapse_term = 0.0

        dgv_dt = recovery_term - degradation_term - collapse_term

        gv = gv + dgv_dt * dt

        # Keep a little headroom above 1 for overshoot, but never allow negative.
        gv = max(0.0, min(1.25, gv))

        self.state.gv = gv
        return gv

    def update_entropy(self, amp, constraint, overflow, dt):
        """
        Entropy grows from:
        - amplitude / strain
        - low constraint
        - overflow pressure

        It relaxes when the system is calm.
        """
        pressure_from_amp = 0.22 * amp
        pressure_from_low_constraint = 0.30 * (1.0 - constraint)
        pressure_from_overflow = 0.55 * overflow

        rise = (
            pressure_from_amp
            + pressure_from_low_constraint
            + pressure_from_overflow
        ) * dt

        relax = 0.03 * dt

        self.state.entropy = max(0.0, self.state.entropy + rise - relax)
        return self.state.entropy
