# src/gv_dynamics.py

from dataclasses import dataclass


@dataclass
class GVState:
    gv: float = 1.0
    entropy: float = 0.0


class GVDynamics:
    """
    GV dynamics with stabilized entropy.

    Key idea:
    - Entropy no longer grows without bound.
    - It relaxes toward a target level (set by entropy_input).
    - This allows true equilibrium (fixed-point / plateau) to exist.

    dGV/dt = -beta * C * (1 - GV)
             + gamma * injection
             - entropy_drag * entropy
    """

    def __init__(
        self,
        beta=1.2,
        gamma=0.60,
        entropy_drag=0.09,
        collapse_threshold=0.35,
        collapse_rate=2.5,
        collapse_floor=0.0,
        collapse_entropy_drag=0.20,
        entropy_relax_rate=0.1,  # 🔥 NEW
    ):
        self.beta = beta
        self.gamma = gamma
        self.entropy_drag = entropy_drag

        self.collapse_threshold = collapse_threshold
        self.collapse_rate = collapse_rate
        self.collapse_floor = collapse_floor
        self.collapse_entropy_drag = collapse_entropy_drag

        self.entropy_relax_rate = entropy_relax_rate

        self.state = GVState()

    def reset(self, gv=1.0, entropy=0.0):
        self.state.gv = gv
        self.state.entropy = entropy

    def step(self, constraint, injection, entropy_input, dt=0.01):
        gv = self.state.gv
        entropy = self.state.entropy

        # 🔥 STABILIZED ENTROPY (KEY CHANGE)
        target_entropy = entropy_input
        entropy += self.entropy_relax_rate * (target_entropy - entropy)

        # Detect collapse regime
        collapsed = gv < self.collapse_threshold

        if collapsed:
            # Collapse dynamics
            d_gv = (
                -self.collapse_rate * constraint * (1 - gv)
                - self.collapse_entropy_drag * entropy
            )
            regime = "collapse"
        else:
            # Normal dynamics
            d_gv = (
                -self.beta * constraint * (1 - gv)
                + self.gamma * injection
                - self.entropy_drag * entropy
            )
            regime = "stable"

        # Integrate
        gv += d_gv * dt

        # Clamp GV to physical bounds
        gv = max(self.collapse_floor, min(1.0, gv))

        # Save state
        self.state.gv = gv
        self.state.entropy = entropy

        return {
            "gv": gv,
            "entropy": entropy,
            "d_gv": d_gv,
            "collapsed": collapsed,
            "regime": regime,
        }

    def run(
        self,
        steps,
        constraint_fn,
        injection_fn,
        entropy_fn,
        dt=0.01,
    ):
        history = []

        for i in range(steps):
            constraint = constraint_fn(i)
            injection = injection_fn(i)
            entropy_input = entropy_fn(i)

            out = self.step(
                constraint=constraint,
                injection=injection,
                entropy_input=entropy_input,
                dt=dt,
            )

            history.append(
                {
                    "step": i,
                    "gv": out["gv"],
                    "entropy": out["entropy"],
                    "d_gv": out["d_gv"],
                    "collapsed": out["collapsed"],
                    "regime": out["regime"],
                }
            )

        return history
