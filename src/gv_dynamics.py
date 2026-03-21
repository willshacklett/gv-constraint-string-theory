from dataclasses import dataclass


@dataclass
class GVState:
    gv: float = 1.0
    entropy: float = 0.0
    collapsed: bool = False


class GVDynamics:
    """
    GV dynamics with irreversible collapse.

    Once GV drops below a critical threshold,
    the system enters collapse and cannot recover.
    """

    def __init__(
        self,
        beta=1.2,
        gamma=0.65,
        entropy_drag=0.08,
        collapse_threshold=0.35,
    ):
        self.beta = beta
        self.gamma = gamma
        self.entropy_drag = entropy_drag
        self.collapse_threshold = collapse_threshold
        self.state = GVState()

    def reset(self, gv=1.0, entropy=0.0):
        self.state.gv = gv
        self.state.entropy = entropy
        self.state.collapsed = False

    def step(self, constraint, injection, dt):
        gv = self.state.gv
        entropy = self.state.entropy

        # 🔥 If already collapsed, keep falling
        if self.state.collapsed:
            gv = gv - 0.6 * dt
            gv = max(0.0, gv)
            self.state.gv = gv
            return gv

        # -----------------------------
        # Recovery (weakened)
        # -----------------------------
        recovery = (
            0.5 * constraint * (1.0 - gv)
            + 0.3 * injection
        )

        # -----------------------------
        # Degradation (stronger)
        # -----------------------------
        degradation = (
            self.beta * (1.0 - constraint)
            + self.entropy_drag * entropy * 1.8
        )

        dgv_dt = recovery - degradation

        gv = gv + dgv_dt * dt

        # -----------------------------
        # 🔥 Collapse trigger
        # -----------------------------
        if gv < self.collapse_threshold:
            self.state.collapsed = True

        gv = max(0.0, min(1.2, gv))
        self.state.gv = gv

        return gv

    def update_entropy(self, amp, constraint, overflow, dt):
        # 🔥 Much stronger entropy growth under stress
        rise = (
            0.25 * amp
            + 0.45 * (1.0 - constraint)
            + 0.70 * overflow
        ) * dt

        relax = 0.02 * dt

        self.state.entropy = max(0.0, self.state.entropy + rise - relax)
        return self.state.entropy
