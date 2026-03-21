import math
from src.gv_core import GodVariable


gv = GodVariable()

amp = 0.1
freq = 1.0
phase = 0.0

for t in range(100):
    phase += 2 * math.pi * freq * 0.01
    value = amp * math.sin(phase)

    gv_val = gv.update(amp)

    energy = 0.5 * amp**2 * freq**2
    effective_energy = energy * gv_val

    print(
        f"{t:03d} | "
        f"GV={gv_val:.3f} | "
        f"E_eff={effective_energy:.5f} | "
        f"S={gv.entropy:.3f}"
    )
