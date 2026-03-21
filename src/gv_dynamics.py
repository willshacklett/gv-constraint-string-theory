from dataclasses import dataclass


@dataclass
class GVState:
    gv: float = 1.0
    entropy: float = 0.0
    regime: str = "stable"
    collapsed: bool = False


class GVDynamics:
    """
    GV dynamics with a real threshold-triggered collapse regime.

    Stable regime:
        dGV/dt = -beta * C * (1 - GV) + gamma * injection - entropy_drag * entropy

    Collapse regime (triggered when GV < collapse_threshold):
        dGV/dt = -collapse_rate * (collapse_floor - GV) - collapse_entropy_drag * entropy

    Notes:
    - Higher constraint C pushes GV down unless already near 1.
    - Injection can temporarily support GV.
    - Entropy adds drag in all regimes.
    - Once GV crosses below collapse_threshold, the system flips into collapse mode.
    - In collapse mode, GV rapidly moves toward collapse_floor.
    """

    def __init__(
        self,
        beta=1.2,
        gamma=0.65,
        entropy_drag=0.08,
        collapse_threshold=0.35,
        collapse_rate=2.5,
        collapse_floor=0.0,
        collapse_entropy_drag=0.20,
    ):
        self.beta = beta
        self.gamma = gamma
        self.entropy_drag = entropy_drag

        self.collapse_threshold = collapse_threshold
        self.collapse_rate = collapse_rate
        self.collapse_floor = collapse_floor
        self.collapse_entropy_drag = collapse_entropy_drag

        self.state = GVState()

    def reset(self, gv=1.0, entropy=0.0):
        self.state.gv = gv
        self.state.entropy = entropy
        self.state.regime = "stable"
        self.state.collapsed = False

    def _update_regime(self):
        if self.state.gv < self.collapse_threshold:
            self.state.regime = "collapse"
            self.state.collapsed = True
        else:
            self.state.regime = "stable"
            self.state.collapsed = False

    def step(self, constraint, injection=0.0, entropy_input=0.0, dt=0.01):
        """
        Advance the system by one time step.

        Parameters
        ----------
        constraint : float
            External constraint pressure (typically 0..1+)
        injection : float
            Support/input pushing GV upward
        entropy_input : float
            New entropy added this step
        dt : float
            Time step

        Returns
        -------
        GVState
            Updated state
        """
        # accumulate entropy first
        self.state.entropy += entropy_input * dt

        # determine current regime from present GV
        self._update_regime()

        if self.state.regime == "stable":
            dgv_dt = (
                -self.beta * constraint * (1.0 - self.state.gv)
                + self.gamma * injection
                - self.entropy_drag * self.state.entropy
            )
        else:
            # collapse mode: strong pull toward floor
            dgv_dt = (
                self.collapse_rate * (self.collapse_floor - self.state.gv)
                - self.collapse_entropy_drag * self.state.entropy
            )

        # update GV
        self.state.gv += dgv_dt * dt

        # clamp GV
        if self.state.gv > 1.0:
            self.state.gv = 1.0
        if self.state.gv < self.collapse_floor:
            self.state.gv = self.collapse_floor

        # re-check regime after update
        self._update_regime()

        return GVState(
            gv=self.state.gv,
            entropy=self.state.entropy,
            regime=self.state.regime,
            collapsed=self.state.collapsed,
        )

    def run(
        self,
        steps=1000,
        dt=0.01,
        constraint_fn=None,
        injection_fn=None,
        entropy_fn=None,
    ):
        """
        Run a full simulation and return history.

        Each function should accept step index i and return a float.
        """
        history = []

        for i in range(steps):
            c = constraint_fn(i) if constraint_fn else 0.0
            inj = injection_fn(i) if injection_fn else 0.0
            ent = entropy_fn(i) if entropy_fn else 0.0

            state = self.step(
                constraint=c,
                injection=inj,
                entropy_input=ent,
                dt=dt,
            )

            history.append(
                {
                    "step": i,
                    "t": i * dt,
                    "gv": state.gv,
                    "entropy": state.entropy,
                    "regime": state.regime,
                    "collapsed": state.collapsed,
                    "constraint": c,
                    "injection": inj,
                    "entropy_input": ent,
                }
            )

        return history


if __name__ == "__main__":
    # Example: stable early, then increasing constraint to trigger collapse
    model = GVDynamics(
        beta=1.2,
        gamma=0.65,
        entropy_drag=0.08,
        collapse_threshold=0.35,
        collapse_rate=2.5,
        collapse_floor=0.0,
        collapse_entropy_drag=0.20,
    )

    model.reset(gv=0.95, entropy=0.02)

    def constraint_fn(i):
        if i < 200:
            return 0.10
        elif i < 450:
            return 0.35
        elif i < 700:
            return 0.75
        else:
            return 1.10

    def injection_fn(i):
        if 250 <= i <= 350:
            return 0.35
        return 0.05

    def entropy_fn(i):
        if i < 400:
            return 0.03
        return 0.08

    history = model.run(
        steps=1000,
        dt=0.01,
        constraint_fn=constraint_fn,
        injection_fn=injection_fn,
        entropy_fn=entropy_fn,
    )

    print("Last 10 states:")
    for row in history[-10:]:
        print(
            f"step={row['step']:4d} "
            f"t={row['t']:.2f} "
            f"gv={row['gv']:.4f} "
            f"entropy={row['entropy']:.4f} "
            f"regime={row['regime']} "
            f"collapsed={row['collapsed']}"
        )
