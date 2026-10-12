# Nonlinear Regression Reference

`reference.npz` is a compressed, pickle-free snapshot of
`delays_0_20_50_100ms.npz` from the final delay-comparison experiment.
It contains no source code and needs no legacy checkout at test time.

Format version 1:

- `format_version`: integer scalar, 1.
- `indices`: zero-based post-integration row indices
  `[0, 199, 399, 599, 799, 809, 819, 839, 879, 999, 1199, 1599, 1999]`.
- `case_names`, `delays_ms`, `directions`: 8 aligned rows, ordered as
  plus_y then minus_y, each with delays 0, 20, 50, 100 ms.
- `states`: float64 array `(8, 13, 18)` at the selected indices.
- `trim_state`: original float64 initial 18-vector.
- `dt`, `duration`: float64 scalars, 0.005 s and 10 s.

State columns are position in meters (0:3), scalar-first quaternion (3:7),
world velocity in m/s (7:10), body angular velocity in rad/s (10:13), and
the five applied wing controls in degrees (13:18). Row `i` is at `(i+1)*dt`.
Quaternion/Euler APIs use radians; the linear controller uses degrees for
Euler and angular-rate errors, as in the original controller.

Fast checks: `python -m pytest -m "not slow"`.
Saved-output regression: `python -m pytest -m slow --regression-output <directory>`.
The directory must contain `results.npz` and `config.json` from the one
completed full simulation run. No test launches nonlinear simulations.
The 8 cases compare sampled states with `rtol=1e-7, atol=1e-8`; they also
check causality, wind bounds, quaternion norms, and actuator limits.
Without the output option these cases are skipped, not recalculated.
Do not regenerate the reference from the implementation being tested.
