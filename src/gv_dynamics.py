from dataclasses import dataclass


@dataclass
class GVState:
    gv: float = 1.0
    entropy: float = 0.0


class GVDynamics:
    """
    Simple GV dynamics model:

        dGV/dt = -beta * C * (1 - GV) + gamma * injection - entropy_drag * entropy

    Notes:
    - Higher constraint C pushes GV down unless already near 1.
    - Injection can support GV temporarily.
    - Entropy adds drag.
    """

    def __init__(self, beta=1.2, gamma=0.65, entropy_drag=0.08):
        self.beta = beta
        self.gamma = gamma
        self.entropy_drag = entropy_drag
        self.state = GVState()

    def reset(self, gv=1.0, entropy=0.0):
        self.state.gv = gv
        self.state.entropy = entropy

    def step(self, constraint, injection, dt):
        gv = self.state.gv
        entropy = self.state.entropy

        dgv_dt = (
            -self.beta * constraint * (1.0 - gv)
            + self.gamma * injection
            - self.entropy_drag * entropy
        )

        gv = max(0.0, min(1.0, gv + dgv_dt * dt))
        self.state.gv = gv
        return gv

    def update_entropy(self, amp, constraint, overflow, dt):
        """
        Entropy rises with amplitude, constraint pressure, and dimensional overflow.
        It also relaxes slightly when system pressure is low.
        """
        rise = (0.18 * amp + 0.10 * constraint + 0.35 * overflow) * dt
        relax = 0.04 * dt
        self.state.entropy = max(0.0, self.state.entropy + rise - relax)
        return self.state.entropy
