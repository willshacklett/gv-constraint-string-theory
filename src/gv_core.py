class GodVariable:
    def __init__(self):
        self.gv = 1.0
        self.entropy = 0.0

    def update(self, amplitude):
        """
        GV decreases as instability increases.
        """
        self.gv = max(0.0, min(1.0, 1.0 - (0.7 * amplitude + 0.3 * self.entropy)))
        return self.gv
