import numpy as np
from Orbits_Functions import lamberts_porkchop_preset

# Example preset initial states
# Replace these with real cases from your mission work
y01 = np.array([7000.0, -50.0, 0.0, 0.0, 7.5, 1.0])
y02 = np.array([9000.0, 100.0, 0.0, 0.0, 6.8, -0.5])

summary = lamberts_porkchop_preset(
    y01=y01,
    y02=y02,
    t1_max=24 * 3600,
    t2_max=24 * 3600,
    t_step=600,
    prograde=True,
    mu=398600,
    title="LEO Rendezvous Trade Study",
    output_png="assets/porkchop_leo.png",
    output_json="assets/porkchop_leo.json",
    tof_min_hr=1.5,
    tof_max_hr=10.0,
    n_levels=12
)

print("Saved:", summary["output_png"])
print("Best solutions:")
best = summary["best_solution"]

if best is not None:
    print("Best solution:")
    print(best["delta_v_kms"], best["tof_hr"])
else:
    print("No valid solution found.")