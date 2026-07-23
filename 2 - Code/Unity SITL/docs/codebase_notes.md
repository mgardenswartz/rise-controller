# Codebase Notes: `scripts/`, `src/`, `conf/`

Orientation notes on how this Python interface drives the Unity SITL (via PX4
lockstep) to compare 5 quadcopter position controllers across 2 trajectories.
Written while reading through the code — not a spec, just "what does this do."

## Big picture

- **Unity SITL** (`quadsim` package, installed in `venv/`) simulates the
  quadcopter physics, running through PX4 in "lockstep" mode (`LockstepRunner`)
  so the sim can run faster than real time — that's the whole point vs. Gazebo.
- **PX4** (`Px4Link`, `HilBridge`) provides the flight-stack side (EKF,
  offboard mode, arming) — position/velocity are read from `px4.state`, and
  control commands are sent as NED acceleration setpoints via
  `runner.step_with_acceleration_ned(...)`.
- **The controller layer** (`src/run_sim.py`) is the actual research code:
  it implements 5 outer-loop position controllers that all output a desired
  acceleration in NED, handed to PX4's inner loop.
- **conf/config.yaml** is the single source of truth for all tunable
  parameters (trajectory shapes, safety boxes, controller gains, NN
  architecture, Optuna settings). Scripts load it, then override individual
  keys with per-trial/per-controller param dicts.
- **The 5 controllers being compared** (via `controller_type` in config):
  1. `baseline` / `baseline_no_wind` — RISE controller (Robust Integral of
     the Sign of the Error), PD + integral term + sign-of-error robustifying
     term.
  2. `pid` — plain PID (K_P, K_I, K_D tuned independently rather than derived
     from k_1/k_2/k_3).
  3. `resnet` — RISE augmented with a ResNet feedforward term (adaptive NN,
     "NN Feedforward").
  4. `integrated_resnet` (INN) — like `resnet`, but the NN output is folded
     into the *integral* term rather than added on top ("Integrated NN").
  5. `supertwisting` — a second-order sliding-mode controller (separate gain
     set k_1/k_2/k_3, no relation to the RISE k's despite the name reuse).

## `src/run_sim.py` — `SimRun`

The core simulation loop for a single trial/episode.

- **`__init__`**: loads `conf/config.yaml`, then overlays a `param_dict`
  (this is how Optuna trials / hardware-param generation / robustness sweeps
  inject specific gains without editing the YAML). Derives `K_P`, `K_I`,
  `K_D` from `k_1, k_2, k_3` for the RISE/NN controllers (closed-loop pole
  placement style relation) unless `controller_type == "pid"`, which uses its
  own independently-tuned K_P/K_I/K_D. Sets up initial position, safety
  bounds, and (if `resnet`/`integrated_resnet`) a `jax_resnet` network via
  `setup_neural_network()`.
- **`setup_neural_network()`**: builds a JIT-compiled ResNet (`jax_resnet.
  resnet_network`) and a JIT-compiled adaptive-update step
  (`compiled_update_step`) that computes the gradient of the network output
  w.r.t. its weights via `jax.vjp`, forms an unprojected weight-update
  (gradient term − leakage term `sigma_mod * theta_hat`), and passes it
  through `discrete_projection` (see `src/proj.py`) to keep weights inside a
  ball of radius `theta_bar`. This is the "adaptive law" for the NN weights —
  classic adaptive-control weight update, not backprop/SGD training.
- **`run()`**: the main control loop, ticking at `control_frequency_hz`
  (default 50 Hz) for `run_length_s` simulated seconds:
  1. Reads `q_ned`/`q_dot_ned` (position/velocity) from `px4.state`.
  2. Checks `check_boundary_escape` — if the drone leaves the configured safe
     box, aborts the trial early and adds a large penalty
     (`w_fail * (time_remaining)^2`) to the cost.
  3. Gets desired position/velocity/acceleration from `TrajectoryGenerator`.
  4. Computes tracking error `e_ned` and its derivative; also aborts (with
     penalty) if error diverges more than 1 m past its initial value —
     catches unstable trials early instead of burning the full episode.
  5. Computes `r1_ned = e_dot + k_1 * e` — the filtered/sliding-surface-like
     error signal used by RISE, NN, and supertwisting controllers.
  6. Evaluates the control law for whichever `controller_type` is active
     (see the `match` block — this is where the 5 controllers actually
     differ). All produce `u_provisional`, a desired NED acceleration.
  7. **Saturation & anti-windup**: clamps horizontal/vertical acceleration to
     `mpc_acc_hor_max_mps2` / `mpc_acc_vert_max_mps2`, and freezes integral
     accumulation on saturated axes only when accumulating further would push
     further into saturation (conditional/clamping anti-windup, not just a
     blanket freeze). Integration is trapezoidal for RISE/PID/NN controllers,
     Euler for supertwisting.
  8. Recomputes final `u_clamped_ned` from the (properly integrated) terms,
     clips again to actuator limits.
  9. Accumulates a running cost `J = ∫ (q_e * t * ||e||² + r_u * ||u||²) dt`
     via trapezoidal integration — a time-weighted tracking-error + control-
     effort cost, used as the Optuna objective (weighting later errors more
     heavily, presumably to penalize steady-state error / non-convergence
     more than initial transients).
  10. Sends the clamped acceleration + fixed yaw setpoint to PX4/Unity via
      `runner.step_with_acceleration_ned`.
  11. Logs per-step history (position, desired position, error, control
      output, NN weights if applicable) for CSV output.
  - At the end: computes RMS tracking error and RMS control effort, and (if
    `save_data`) writes a timestamped CSV to
    `output/traj{N}/{controller_type}/{timestamp}.csv`.
  - Returns `(cost_J, e_rms, u_rms)`.

## `src/desired_trajectory.py` — `TrajectoryGenerator`

Generates the desired position/velocity/acceleration trajectory at a given
time `t`, exactly (via autodiff) rather than by finite-differencing.

- **Trajectory 1**: a 3D Lissajous-like figure (different sine frequencies on
  x/y/z: `wx=2w, wy=1w, wz=4w`) parameterized by a *warped* phase `tau(t)`
  rather than `t` directly — `alpha_warp` controls how much the phase speed
  oscillates (`dtau/dt = c*(1 - alpha*sin(w*tau)^2)`), which appears to be a
  way to vary trajectory speed non-uniformly (e.g. slow down at turns) while
  keeping it periodic and smooth.
- **Trajectory 2**: a "petal"/rose-curve path (`r = radius * cos(2*theta)`,
  4-petal rose) traced at constant *linear* speed (`traj2_target_speed_mps`)
  by solving for `dtheta/dt` such that arc-length speed is constant
  (`f_theta = 1 + 3*sin(2*theta)^2` comes from the rose curve's arc-length
  element).
- Both trajectories precompute their phase variable (`tau` or `theta`) once
  via `scipy.solve_ivp` over a fine time grid (`_precompute_phases`), then at
  runtime do a cheap `jnp.interp` lookup of phase-vs-time plus **exact**
  derivatives of position w.r.t. phase via `jax.jacfwd`/`jacfwd(jacfwd(...))`,
  chained with the (also analytically known) `tau_dot`/`tau_ddot`. This
  avoids numerically differentiating a spline for velocity/acceleration.
- `get_desired_state(t)` is the public entry point, called every control
  step from `run_sim.py`.

## `src/proj.py` — `discrete_projection`

Standalone gradient projection operator used by the adaptive NN weight
update in `run_sim.py`. Given an unprojected weight step, if the resulting
weight vector would stay inside the ball `||theta|| <= theta_bar`, it's
accepted as-is; otherwise it's projected back onto (near) the ball boundary
via a diagonal (`gamma_diag`-weighted) scaling found by 30 steps of
bisection on a scalar multiplier `eta`. This is what guarantees the NN
weights stay bounded (needed for adaptive-control stability proofs) — in
current `conf/config.yaml`, `theta_bar` is set to `1e6`, i.e. effectively
unconstrained/never triggers for these experiments.

## `conf/config.yaml`

Single config, two extra top-level blocks besides the controller params:

- `aviary_rise_node.ros__parameters` — everything `SimRun`/`TrajectoryGenerator`
  read (control rate, trajectory shape/timing, safety box in NED, controller
  gains `k_1/k_2/k_3/k_rise`, NN architecture (`hidden_width`, `num_blocks`,
  activation functions, `theta_bar`, `gamma`, `sigma_mod`), cost weights
  `q_e`/`r_u`, `w_fail`, and Optuna-related knobs (`num_eval_seeds`,
  `base_seed`, `xy_rand_range_m`/`z_rand_range_m` for domain randomization,
  `stage2_base_gains`, `db_path`).
  - Naming convention `_m_ned` / `_mps2` etc. = meters in NED frame / m/s².
  - Name kept over from a ROS2 node (`ros__parameters`) even though this repo
    doesn't run ROS — likely shared with a Gazebo/ROS2 sibling implementation
    (see `docs/experiment_prep.md`, which pushes files to a
    `aviary_rise_controller` ROS2 package on real hardware).
- `quadsim` — host/port for the Unity SITL RPC connection (`localhost:5555`,
  telemetry on `port+1`).
- `px4` — PX4 SITL connection details + `sentinel_px4.params` (PX4 param file
  loaded at startup via `load_param_file`).

## `scripts/optimization.py` — Optuna tuning orchestrator (the big one)

Runs staged Bayesian optimization (Optuna, TPE by default) to tune each
controller's gains against the sim, with PX4 crash recovery built in since
these are long unattended runs.

- **Stages** (selected via `--stage`):
  - `1A` — RISE gains, no-wind scene (currently commented out of
    `run_optimization.sh` — wind-only tuning is what's actually used).
  - `1B` — RISE gains, wind scene. This is the "base" stage whose gains seed
    stage 2 (per `stage2_base_gains: "1A"` or `"1B"` in config — currently
    `"1A"`, though 1A is disabled in the shell driver, so check this before
    running from scratch).
  - `2A` — ResNet feedforward NN hyperparams (`gamma`, `sigma_mod`; k_1-3/
    k_rise/K_P etc. carried over from stage 1). Fixed architecture:
    4 blocks, hidden width 8.
  - `2B` — Integrated ResNet NN hyperparams, same search space as 2A but
    `controller_type='integrated_resnet'`.
  - `3` — Supertwisting gains (`k_st_1/2/3`, log-uniform).
  - `4` — PID gains (`K_P/K_I/K_D`, log-uniform; dummy k_1-3/k_rise=0 so
    `SimRun.__init__` doesn't choke on missing keys).
  - Each stage's trials/study are persisted to a per-trajectory SQLite DB
    (`output/traj{N}/stage_{stage}.db`), resumable (`load_if_exists=True`).
- **`evaluate_minibatch`**: the Optuna objective wrapper. For a given gain
  set, runs `num_eval_seeds` (4 fixed spawn points, defined per-trajectory as
  `traj{1,2}_fixed`, plus any beyond 4 randomized within
  `xy_rand_range_m`/`z_rand_range_m`) full episodes via `SimRun`, and returns
  the **worst-case** cost across seeds (robust/minimax optimization, not
  average-case) — also stores worst-case `e_RMS`/`u_RMS` as trial user
  attrs (used later by `extract_gains.py` for reporting).
  - Before each seed, uses `runner.fly_to_ned(...)` to reposition the drone
    under PX4 position control before handing off to the controller under
    test — so trials start from a consistent, settled hover.
  - Has retry/recovery logic: setup failures (can't reach start position)
    retry up to `MAX_SETUP_RETRIES` without penalizing the trial; controller
    crashes (exceptions during `SimRun.run()`) retry up to
    `MAX_CRASH_RETRIES`, and on exhausting retries assign a large fixed cost
    (`1e6`) rather than losing the whole study.
- **`restart_px4()`**: kills and restarts the PX4 SITL process
  (`~/PX4-Autopilot`, `make px4_sitl none_iris`) and rebuilds the
  `LockstepRunner`/`Px4Link`/`HilBridge`, capping sim speed to 1x during PX4
  boot (`PX4_BOOT_SPEED_CAP`) to avoid flooding PX4's uORB queues — this was
  clearly a pain point (see comments about "STALE sensors").
  - **Note**: `restart_px4()` references `sim` as a module global but never
    receives/declares it in its own `global` statement beyond `runner, px4,
    sim` — actually it does list `sim` in the `global` line, so it's fine;
    just flagging that `sim` must already be constructed by the caller
    before any restart can occur, which is true (`sim` is built in
    `__main__` before the try block).
- **`ETACallback`** / **`EarlyStoppingCallback`**: Optuna callbacks — one
  prints a running ETA based on average trial time, the other stops the
  study if no improvement in `--patience` trials (disabled when
  `patience=0`, which `run_optimization.sh` currently uses for all stages).
- Top-level connects to Unity (`QuadSim`), builds the PX4 bridge/link/lockstep
  runner, loads PX4 params, arms + engages offboard, then calls
  `study.optimize(...)` for the selected stage. On exit (including
  Ctrl+C/exceptions), lands and disarms before closing the runner/sim
  connection — `signal_handler` makes Ctrl+C hard-exit immediately
  (`os._exit(1)`) rather than trying to land, though, which seems intentional
  for fast iteration (there's also a `finally` block for graceful exits from
  exceptions, just not from SIGINT).

## `scripts/run_optimization.sh`

Drives the full two-trajectory optimization pipeline end-to-end:
`caffeinate` to prevent the Mac from sleeping, for each trajectory (1, 2):
patches `desired_trajectory` in `conf/config.yaml` via `sed`, kills/restarts
Unity + PX4 (`aviary_{N}.app`), then runs stages 1B → 2A → 2B → 3 → 4 in
order (1A is commented out — no-wind tuning isn't currently part of the
pipeline), then runs `extract_gains.py`. Any nonzero exit aborts the whole
script. Trial counts: 75/50/50/100/50 for stages 1B/2A/2B/3/4.

## `scripts/extract_gains.py`

Post-processes the per-stage Optuna SQLite DBs into a single
`output/traj{N}/best_gains.yaml` with one entry per controller
(`BEST_RISE_NO_WIND`, `BEST_RISE`, `BEST_ST`, `BEST_PID`, `BEST_NN`,
`BEST_INN`), keyed by what `evaluate_robustness.py`/
`generate_hardware_params.py` expect. For the two NN controllers it merges in
the base RISE gains (from whichever of stage 1A/1B `stage2_base_gains`
selects) plus a fixed architecture dict (`num_blocks=6, k_0=2, k_i=2,
hidden_width=16` — **note this differs from the search-time architecture in
optimization.py's stage 2A/2B, which used 4 blocks / width 8**; presumably
intentional — a larger fixed net used for final eval/hardware regardless of
what was searched, but worth double-checking if results seem off). Maps
supertwisting's Optuna param names (`k_st_1/2/3`) back to the generic
`k_1/2/3` keys `SimRun` expects.

## `scripts/evaluate_robustness.py`

Monte Carlo robustness comparison across all 5 controllers using the tuned
`best_gains.yaml`. For `--num_trials` trials (4 fixed spawn points + random
beyond that, same scheme as `optimization.py`), runs every controller from
the *same* spawn point each trial (fair comparison — same disturbance
realization per trial across controllers), collects Cost/e_RMS/u_RMS per
controller into a DataFrame, saves to CSV, then runs statistics:
- `print_statistics`: median/IQR/max/min per controller per metric.
- `check_normality_and_compare`: Shapiro-Wilk normality test per controller
  column, then for each metric compares `INN_Integrated` against every other
  controller pairwise on the *paired differences* — paired t-test if
  differences are normal, Wilcoxon signed-rank otherwise — one-sided
  (`alternative='less'`, i.e. testing whether INN is *better*, hypothesis-
  driven toward INN being the proposed/best method). Output is captured to
  both stdout and a markdown summary file via a small `Logger` tee class.
- Also has PX4 crash recovery (raises `RuntimeError` if the vehicle drops out
  of armed/offboard) but, unlike `optimization.py`, does **not** retry/
  restart PX4 on failure — a crash here just aborts the whole sweep. Fine for
  a supervised eval run, unlike the long unattended optimization runs.

## `scripts/run_robustness.sh`

Thin wrapper: prompts you to manually start Unity (headless) and PX4 SITL
first (unlike `run_optimization.sh`, doesn't launch them itself), then runs
`evaluate_robustness.py --num_trials 20 --config conf/config.yaml --db_dir
output/traj1` (hardcoded to trajectory 1 — edit `TRAJ_NUM` at the top to
switch).

## `scripts/generate_hardware_params.py`

Converts a `best_gains.yaml` entry into a standalone hardware-ready params
YAML (`aviary_rise_node.ros__parameters` block, matching the real ROS2 node's
expected format — see `docs/experiment_prep.md` for the adb push workflow
onto the physical quad). Merges base config + the selected controller's
tuned params, then applies hardware-specific overrides:
`save_data=True`, `mpc_acc_vert_max_mps2=6.0` (raised from the sim's 3.0 —
presumably more vertical authority is available/safe on real hardware, or
this compensates for a modeling gap), `theta_bar=1e6`, odom watchdog
settings (`odom_timeout_s`, `odom_watchdog_freq` — not used in sim at all,
only meaningful once a real odometry source is in the loop), and
`vehicle_name` (`px4_1` for Gazebo, `sentinel5` for real hardware). If the
target controller is one of the NN variants, also pre-computes and embeds
concrete `initial_weights` (flattened list of floats) using the same
`jax_resnet.init_resnet_weights` + seed as `SimRun`, so the real node starts
from the exact same initialization rather than reseeding independently.
Also carries a couple of `# TEMP overrides` (`traj1_z_amp_m_ned`,
`init_z_m_ned`) — worth checking whether these are still meant to be
temporary or have become de facto permanent.

## `scripts/debug_flight.py`

Manual/interactive single-flight runner for one controller, driven entirely
by CLI args + `output/traj{N}/best_gains.yaml` (via `--controller_type`,
mapped through the same `controller_map` naming as `extract_gains.py`'s
output keys). Connects to Unity/PX4 exactly like the other entry points,
flies to a randomized-per-seed start position for each of `num_eval_seeds`
evaluation seeds (same fixed/randomized spawn scheme), runs `SimRun`, prints
per-seed cost/RMS, and reports the worst case. Useful for eyeballing a single
controller/trajectory combination — e.g. after tuning, before committing to a
full robustness sweep, or to debug why a particular controller/trajectory
pairing performs badly.

## `scripts/plot_csv_results.py`

Post-flight visualization for a single `SimRun` output CSV (i.e. one row of
`save_data=true` logging). Handles both the sim's native column names and a
`ros_node`-style naming convention (`column_mapping`, `w_*` → `W*`) so it can
plot either sim CSVs or real-hardware-logged CSVs interchangeably. Produces,
per invocation: a static summary PNG (top-down XY path + horizontal/vertical
error over time), a NN-weight-evolution PNG (if `W*` columns are present), a
desired-trajectory 3D+projections PNG, and a real-time-synced MP4 animation
of the top-down flight path (interpolated onto a fixed FPS grid with a
5-second fading trail) via `matplotlib.animation` + ffmpeg.

## External packages worth knowing about (not in this repo, but load-bearing)

- **`quadsim`** (`venv/lib/.../quadsim`) — Python client for the Unity SITL:
  `QuadSim` (RPC connection), `Px4Link` (PX4 MAVLink/state interface),
  `HilBridge` (HIL sensor bridge), `LockstepRunner` +
  `load_param_file` (drives the lockstep sim loop, position-hold helpers
  like `fly_to_ned`/`engage_offboard`, and PX4 param loading). This is your
  friend's package.
- **`jax_resnet`** (`venv/lib/.../jax_resnet`, from
  `github.com/mgardenswartz/resnet` per `docs/experiment_prep.md`) —
  `resnet_network`, `init_resnet_weights`: the adaptive-control ResNet
  architecture used by the `resnet`/`integrated_resnet` controllers.

## Open questions / things to double check if results look off

- Stage 2A/2B search-time NN architecture (4 blocks, width 8) vs.
  `extract_gains.py`'s fixed architecture for final gains (6 blocks, width
  16) — different network sizes at tuning time vs. deployment time.
- `stage2_base_gains: "1A"` in `conf/config.yaml`, but stage 1A is commented
  out in `run_optimization.sh` — if a fresh pipeline run is done exactly as
  the script stands, stage 2 would try to seed from a `stage_1A.db` that was
  never created (`extract_gains.py`/`optimization.py` both handle this with
  warnings + fallback to `conf/config.yaml`, so it won't crash, just silently
  uses un-tuned base gains for stage 2's seed).
- `# TEMP overrides` in `generate_hardware_params.py`.
