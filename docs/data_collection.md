# Data Collection

This guide describes how demonstrations are recorded for a task. The
repository provides two data-collection methods, a scripted state machine and
keyboard teleoperation. Students choose one, adjust its parameters and the task
settings as needed, and record a LeRobot v3 dataset. The dataset is then
inspected and verified (step 3 of [Getting Started](getting_started.md)),
used for training ([LeRobot Training](lerobot_training.md)), and later
compared with rollout results.

> **Run on:** the simulator environment
> ([Getting Started §3.5](getting_started.md#35-launch-the-simulator)). Every
> recording command needs `--enable_cameras`, because the dataset contains
> camera images.

Every section starts with one of two labels:

- **Required**: something that students must do or follow.
- **Background**: an explanation that the required steps depend on. It
  contains no work items.

## 1. The two methods

**Background**

| | State machine | Keyboard teleoperation |
|---|---|---|
| Script | `scripts/datagen/generate.py` | `scripts/teleop.py` |
| Who controls the robot | A scripted controller, one per task, in `packages/simulator/src/simulator/datagen/state_machine/<task>.py` | A person, through `FrankaKeyboard` (`packages/simulator/src/simulator/devices/franka_keyboard.py`) |
| Which episodes are saved | Those that pass the state machine's `check_success` | Those that the operator marks with **N** |
| Display | GUI or `--headless` | GUI only |
| Typical use | Many episodes with consistent motion | Fewer episodes with human-like variation, or a task without a state machine |

Both scripts use the same robot and action setup as rollout
(`use_joint_position_actions()`, [section 4](#4-adjusting-the-collection)), and
both record with the repository's LeRobot recorder
([section 5](#5-the-recorder-and-the-dataset)).

The three predefined tasks:

| | Cup stacking | Cutlery arrangement | Toy blocks collection |
|---|---|---|---|
| Task id | `HCIS-CupStacking-SingleArm-v0` | `HCIS-CutleryArrangement-SingleArm-v0` | `HCIS-ToyBlocksCollection-SingleArm-v0` |
| Task configuration | [`cup_stacking_env_cfg.py`](../packages/simulator/src/simulator/tasks/cup_stacking/cup_stacking_env_cfg.py) | [`cutlery_arrangement_env_cfg.py`](../packages/simulator/src/simulator/tasks/cutlery_arrangement/cutlery_arrangement_env_cfg.py) | [`toy_blocks_collection_env_cfg.py`](../packages/simulator/src/simulator/tasks/toy_blocks_collection/toy_blocks_collection_env_cfg.py) |
| Scene objects | `blue_cup`, `pink_cup` | `plate`, `fork`, `knife` | `green_block`, `blue_block`, `red_block`, `storage_box` |
| Success check (`success` term) | Blue cup within 2 cm of the pink cup in x and in y, and 8–10 cm higher | Fork and knife each inside a 10 cm × 10 cm square centred 12 cm from the plate centre: fork at smaller x (the robot's left, the right side of the front image), knife at larger x | Every block inside the storage box footprint (0.296 m × 0.209 m) shrunk by 1 cm on each side, in the box's own frame, and 0–11 cm above the box bottom |
| Task text (`task_description`) | `pick up the blue cup and place it on the pink cup.` | `place the fork on the left and knife on the right of the plate.` | `pick up the toys and place them into the storage box.` |

Every reset moves the objects by a few centimetres and changes the light
intensity. The success check looks only at the object positions at one
moment; it does not know whether the gripper still holds an object.

## 2. Record with the state machine

**Required** if the group uses the state machine.

```bash
python scripts/datagen/generate.py \
    --task HCIS-CupStacking-SingleArm-v0 \
    --num_demos 50 \
    --device cuda --enable_cameras --headless \
    --record --use_lerobot_recorder \
    --lerobot_dataset_repo_id ${HF_USER}/cup-stacking-demos
```

| Option | Meaning |
|---|---|
| `--task` | One of the three task ids. |
| `--num_demos` | Number of attempts, successful or not (required, at least 1). |
| `--record --use_lerobot_recorder --lerobot_dataset_repo_id <id>` | Record into `datasets/<id>`. Without `--record`, the state machine runs and nothing is saved. |
| `--resume` | Append to an existing dataset with the same id. Its episodes count toward `--num_demos`. |
| `--seed` | Random seed of the environment; by default the current time. |
| `--headless` | Run without a window, which is faster. |

In each attempt, the state machine reads the object positions after the reset,
moves through its phases (for example hover, approach, grasp, lift, move,
lower, release), and judges the result with its own `check_success`. An
attempt is aborted when an object falls below the table. The script prints
`[Data Usage] <attempt>/<num_demos> success` or `fail` per attempt and the
list of successful attempts at the end. Not every attempt succeeds, so the
dataset usually has fewer episodes than `--num_demos`.

## 3. Record with keyboard teleoperation

**Required** if the group uses keyboard teleoperation.

```bash
python scripts/teleop.py \
    --task HCIS-CupStacking-SingleArm-v0 \
    --teleop_device keyboard \
    --device cuda --enable_cameras \
    --record --use_lerobot_recorder \
    --lerobot_dataset_repo_id ${HF_USER}/cup-stacking-teleop \
    --num_demos 20
```

| Option | Meaning |
|---|---|
| `--num_demos` | Stop after this many saved episodes; `0` (default) runs until Ctrl+C. |
| `--sensitivity` | Scale of the motion per key press (default 1.0). |
| `--record`, `--use_lerobot_recorder`, `--lerobot_dataset_repo_id`, `--resume`, `--seed` | As for the state machine. |

The keys move a target pose for the hand; an inverse kinematics solver turns
it into joint targets. The script prints the key table at start, and the
Isaac Sim window needs the keyboard focus.

| Key | Action |
|---|---|
| B | Start control; recording of the episode starts here |
| W / S, A / D, J / K | Move the hand in +x / −x, +y / −y, +z / −z |
| H / L, U / I, Q / E | Roll, pitch, yaw (− / +) |
| C / M | Open / close the gripper |
| N | Save the episode and reset |
| R | Discard the episode and reset |

The script does not apply the task's success check: the operator presses **N**
only when the task is solved as defined in [section 1](#1-the-two-methods),
and **R** otherwise. After every reset, **B** starts control again.
Ctrl+C in the terminal ends the session and closes the dataset.

## 4. Adjusting the collection

**Required**

Students may adjust the collection to target the data they need, for example:

- the state machine's constants at the top of
  `packages/simulator/src/simulator/datagen/state_machine/<task>.py`
  (approach and lift heights, phase durations, grasp offsets, random jitter,
  success ranges);
- the task configuration: object randomization ranges, lighting, object
  start positions, or a stricter success check;
- the teleoperation `--sensitivity`.

Such changes are part of the group's data-collection design and are reported
with it.

Rollout never uses the recording configuration. It builds the environment
from the evaluation file in `eval/`, and the final ranking uses a private
evaluation configuration. A change therefore affects only the data, and it
has to stay compatible with rollout in two ways.

**The rollout contract stays unchanged.** These settings define what the
policy receives and sends
([Rollout §4](rollout.md#4-policy-input-and-output-contract)). A dataset
recorded with other values trains a policy that cannot run in evaluation:

| Setting | Required value | Defined in |
|---|---|---|
| Robot | Franka Panda, same joints and gripper | template |
| Action space | `use_joint_position_actions()`: 7 absolute joint targets + 1 gripper command, stiffness 80, damping 4 | template |
| Control rate | `sim.dt = 1/60`, `decimation = 1`, one action per `env.step()` | template |
| Cameras | exactly `front` and `wrist`, RGB, 640×480 | template (`front` pose per task) |
| Frame builder | `build_lerobot_frame`, unchanged keys and shapes | template |

"Template" is
[`single_arm_franka_cfg.py`](../packages/simulator/src/simulator/tasks/template/single_arm_franka_cfg.py).

**Other differences from the evaluation environment are deliberate.** Each one
is a distribution shift that the policy has to bridge. The task configuration
is compared with `eval/<task>_eval.py` for object start positions and
randomization ranges, lighting, robot start pose, `front` camera pose, success
check, and episode length. A wider randomization range than the evaluation
file usually helps, because the private evaluation uses other positions; a
narrower range, a moved camera, or a different robot start pose usually hurts.

## 5. The recorder and the dataset

**Background**

`--use_lerobot_recorder` attaches the repository's `LeRobotRecorderManager`
([`lerobot_recorder.py`](../packages/simulator/src/simulator/utils/lerobot_recorder.py)),
which writes the LeRobot v3 dataset. Without it, `--record` writes an HDF5
file that cannot be trained on or verified. The recorder extends LeIsaac's
recorder by one column per randomized object, `object_position.<name>`, which
the [coverage check](object_position_coverage.md) reads.

Behavior that affects recording:

- One frame is written per `env.step()` at `fps = 60`. Camera images refresh at
  30 Hz, so each image appears on two consecutive frames.
- The first 5 steps after every reset are skipped, so that no image of the
  previous scene is recorded.
- An episode is written at the next reset if it is marked successful, and
  dropped otherwise. Only environment 0 is recorded.
- The dataset goes to `datasets/<repo_id>` (`HF_LEROBOT_HOME`). If that folder
  already exists, recording stops with `FileExistsError`; `--resume` appends to
  it instead, with the same task and `fps`. Copying the folder first protects
  the existing episodes from a failed session.
- The dataset is closed when the script ends, including after Ctrl+C. A
  process killed with `kill -9` leaves a dataset that cannot be loaded.

Columns written in addition to the
[rollout features](rollout.md#4-policy-input-and-output-contract)
(`observation.state`, `observation.images.front`,
`observation.images.wrist`, `action`):

| Column | Shape / type | Content |
|---|---|---|
| `object_position.<name>` | `(3,)` float32, `x, y, z` | Position of every object in the configuration's `tracked_object_names`, relative to the environment origin. Fixed objects (`plate`, `storage_box`, listed in `fixed_object_names`) get no column. |
| `task_index` | int64 | Number of the task text. The text itself is stored once in `meta/tasks.parquet`; LeRobot returns it as `task` when the dataset is loaded. Language-conditioned policies (for example pi0 or SmolVLA) use it as their instruction; Diffusion Policy and ACT ignore it. |

[Object Position Coverage §2](object_position_coverage.md#2-the-object-position-columns)
describes how the coverage check reads the `object_position.*` columns.

## 6. Troubleshooting

**Background**

| Symptom | Cause | Fix |
|---|---|---|
| `FileExistsError` at start | The dataset folder already exists | Choose a new `--lerobot_dataset_repo_id`, delete the old folder, or use `--resume` |
| `PermissionError` under `datasets/` | The folder was created by the container as root | Exit the container, or run `make fix-perms` |
| `LeRobot not found. Using GenericDataRecorder.` followed by an `AttributeError` | LeRobot cannot be imported in the simulator environment | Reinstall it with `make install-sim`, or rebuild the container image |
| Dataset will not load / Parquet footer error | The script was killed before it closed the dataset | Stop with Ctrl+C; record a corrupted dataset again under a new id |
| Missing `object_position.*` columns | Recorded without `--use_lerobot_recorder` | Record again with the option |
| Camera errors or missing images | Started without `--enable_cameras` | Add the option |
| Most state-machine attempts fail | Changed task settings or state-machine constants | Watch a run with the GUI and without `--record`; compare the failing phase with the constants |
| Keyboard does nothing | The window has no focus, the script runs `--headless`, or **B** was not pressed after a reset | Click the Isaac Sim window, run with the GUI, and press **B** |
| Ctrl+C has no effect | The script hangs after an error | Stop it from another terminal: `pkill -9 -f scripts/teleop.py` (or `generate.py`) |
