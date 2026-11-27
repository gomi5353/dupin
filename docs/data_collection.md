# Data Collection

This guide describes how demonstrations are recorded for a task. The
repository provides two data-collection methods, a scripted state machine and
keyboard teleoperation. Either one is used, with its parameters and the task
settings adjusted as needed, to record a LeRobot v3 dataset. The dataset is then
inspected and verified (step 3 of [Getting Started](getting_started.md)),
used for training ([LeRobot Training](lerobot_training.md)), and later
compared with rollout results.

> **Run on:** the simulator environment
> ([Getting Started §3.4](getting_started.md#34-launch-the-simulator)). Every
> recording command needs `--enable_cameras`, because the dataset contains
> camera images.

Every section starts with one of two labels:

- **Required**: something to do or follow.
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
| Task text, objects referred to by name | `pick up the blue cup and place it on the pink cup.` | `place the fork on the left and the knife on the right of the plate.` | `pick up the bridge, the cylinder and the triangle and place them into the storage box.` |
| Task text, objects referred to by color | `pick up the blue object and place it on the pink object.` | `place the pink object on the left and the orange object on the right of the plate.` | `pick up the green object, the blue object and the red object and place them into the storage box.` |

Every reset moves the objects by a few centimetres, changes the intensity and
color temperature of the scene light `rect_light`, and picks how the
episode's task text refers to the objects: by name or by color
([section 4](#4-adjusting-the-collection)). The task text is built from the
configuration's `task_description_template` and the phrases in
`object_references`; the recording configurations use the name form 99% of the time and the color form 1%. The
success check looks only at the object positions at one moment; it does not
know whether the gripper still holds an object.

## 2. Record with the state machine

**Required** when the state machine is used.

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
| `--num_demos` | Number of attempts, successful or not (required, at least 1). Counted over all environments and rounded up to whole rounds of `--num_envs`. |
| `--num_envs` | Environments simulated in parallel (default 1); see below. |
| `--record --use_lerobot_recorder --lerobot_dataset_repo_id ${HF_USER}/<repo_name>` | Record into `datasets/${HF_USER}/<repo_name>`. Without `--record`, the state machine runs and nothing is saved. |
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

With `--num_envs N`, the N environments run one attempt each per round, side by
side in one scene, each with its own randomized object positions, light and
task wording. The state machine steps them together, and the round ends when
it has finished its phases (or when an object falls in any environment). Each
environment is then judged on its own: the round prints
`Round done: <k>/<N> envs succeeded` and records the k successful episodes.
More environments record faster until the GPU is full; each one renders its
own cameras.

## 3. Record with keyboard teleoperation

**Required** when keyboard teleoperation is used.

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
| `--record`, `--use_lerobot_recorder`, `--lerobot_dataset_repo_id`, `--resume`, `--seed` | As for the state machine. Recording needs `--num_envs 1` (the default): one operator drives one robot. |

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

The collection can be adjusted to target the data needed, for example:

- the state machine's constants at the top of
  `packages/simulator/src/simulator/datagen/state_machine/<task>.py`
  (approach and lift heights, phase durations, grasp offsets, random jitter,
  success ranges);
- the task configuration: object randomization ranges, object start
  positions, or a stricter success check;
- the light ranges: `intensity_range` and `color_temperature_range` of the
  `randomize_light_conditions("rect_light", ...)` event in the task
  configuration (recording: intensity 4500–5500 and 5300–5900 K; the
  evaluation files use 2500–8000 and 3000–6500 K);
- how often the task text refers to the objects by color: `reference_weights`
  in the task configuration, for example `{"name": 0.5, "color": 0.5}`
  (recording: `{"name": 0.99, "color": 0.01}`; the evaluation files use
  `{"name": 0.5, "color": 0.5}`). One form is drawn per episode and used for
  every object. The phrases themselves are in `object_references`. Only the
  words change: the objects are never recolored;
- the teleoperation `--sensitivity`.

Rollout never uses the recording configuration. It builds the environment
from the evaluation file in `eval/`, and the final evaluation uses a private
evaluation configuration. A change therefore affects only the data, and it
should stay compatible with rollout in two ways.

**The rollout contract is best left unchanged.** These settings define what the
policy receives and sends
([Rollout §4](rollout.md#4-policy-input-and-output-contract)). Other keys or
shapes (cameras, action size) stop a policy from running in rollout; other
values of the remaining settings (control rate, gains, camera pose) usually
run, but the policy may perform much worse:

| Setting | Value | Defined in |
|---|---|---|
| Robot | Franka Panda, same joints and gripper | template |
| Action space | `use_joint_position_actions()`: 7 absolute joint targets + 1 gripper command, stiffness 80, damping 4 | template |
| Control rate | `decimation = 1` (set by the template) with Isaac Lab's default `sim.dt = 1/60`, one action per `env.step()` | template |
| Cameras | exactly `front` and `wrist`, RGB, 640×480 | template (`front` pose per task) |
| Frame builder | `build_lerobot_frame`, unchanged keys and shapes | template |

"Template" is
[`single_arm_franka_cfg.py`](../packages/simulator/src/simulator/tasks/template/single_arm_franka_cfg.py).

**Other differences from the evaluation environment are deliberate.** Each one
is a distribution shift that the policy has to bridge. The task configuration
is compared with `eval/<task>_eval.py` for object start positions and
randomization ranges, light ranges, reference weights, robot start pose, `front` camera pose, success
check, and episode length. A wider randomization range than the evaluation
file usually helps, because the private evaluation uses other positions; a
narrower range, a moved camera, or a different robot start pose usually hurts.

## 5. The recorder and the dataset

**Background**

`--use_lerobot_recorder` attaches the repository's `LeRobotRecorderManager`
([`lerobot_recorder.py`](../packages/simulator/src/simulator/utils/lerobot_recorder.py)),
which writes the LeRobot v3 dataset. Without it, `--record` writes an HDF5
file that cannot be trained on or verified. The recorder extends LeIsaac's
recorder by the columns that [data verification](data_verification.md) reads:
the object positions, the lighting, and the object colors, and it writes each
episode's own task text.

Behavior that affects recording:

- One frame is written per `env.step()` at `fps = 60`. Camera images refresh at
  30 Hz, so each image appears on two consecutive frames.
- The first 5 steps after every reset are skipped, so that no image of the
  previous scene is recorded.
- An episode is written at the next reset if it is marked successful, and
  dropped otherwise. Every environment is recorded into its own buffer, and
  episodes are numbered in the order they are written.
- The dataset goes to `datasets/${HF_USER}/<repo_name>`. If that folder
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
| `object_position.<name>` | `(3,)` float32, `x, y, z` | Position of every object in the configuration's `tracked_object_names`, relative to the environment origin, in metres. Fixed objects (`plate`, `storage_box`, listed in `fixed_object_names`) get no column. Axis Object Poses. |
| `lighting.rect_light` | `(2,)` float32, `intensity, color_temperature` | The scene light's USD `inputs:intensity` and `inputs:colorTemperature` in kelvin (6500 K when the light does not use its color temperature). One column per light in `recorded_light_names`. Axis Image Augmentations. |
| `object_color.<name>` | `(3,)` float32, `r, g, b` | Diffuse color of the material bound to the object (OmniPBR `inputs:diffuse_color_constant`), for every object in `object_references`. It shows that the object itself never changes. Axis Object Properties. |
| `task_index` | int64 | Number of the episode's task text. Each episode gets its own text, in the reference form drawn for it; the distinct texts are stored once in `meta/tasks.parquet`, and LeRobot returns the text as `task` when the dataset is loaded. Axis Object Properties. Language-conditioned policies (for example pi0 or SmolVLA) use it as their instruction; Diffusion Policy and ACT ignore it. |

The `lighting.*` and `object_color.*` values are read from the scene at every
frame; the other columns come from the recorded step. How the verifier reads
each column is described on the axis pages:
[Object Poses](object_pose.md), [Image Augmentations](image_augmentation.md),
[Object Properties](object_property.md).

## 6. Troubleshooting

**Background**

| Symptom | Cause | Fix |
|---|---|---|
| `FileExistsError` at start | The dataset folder already exists | Choose a new `--lerobot_dataset_repo_id`, delete the old folder, or use `--resume` |
| `PermissionError` under `datasets/` | The folder was created by the container as root | Exit the container, or run `make fix-perms` |
| `LeRobot not found. Using GenericDataRecorder.` followed by an `AttributeError` | LeRobot cannot be imported in the simulator environment | Reinstall it with `make install-sim`, or rebuild the container image |
| `ERROR_INCOMPATIBLE_DRIVER`, `Failed to create any GPU devices`, or `libGLU.so.1` missing at start (new cloud VM) | The host lacks the Vulkan/OpenGL libraries or the NVIDIA EGL vendor file | `bash scripts/setup_brev_host.sh` ([Getting Started §3.4](getting_started.md#34-launch-the-simulator)) |
| Stops after the `pink_cup` mass warning, one CPU core at 100%, no error; Kit log has `omni.physx handle on CUDA lib is (nil)` or `GPU solver pipeline failed` | The `libcuda.so` link is missing, so PhysX runs on the CPU | `bash scripts/setup_brev_host.sh`, then start again |
| Dataset will not load / Parquet footer error | The script was killed before it closed the dataset | Stop with Ctrl+C; record a corrupted dataset again under a new id |
| Missing `object_position.*`, `lighting.*` or `object_color.*` columns | Recorded without `--use_lerobot_recorder`, or the task does not list the light in `recorded_light_names` or the object in `object_references` | Record again with the option and the lists set |
| Camera errors or missing images | Started without `--enable_cameras` | Add the option |
| Most state-machine attempts fail | Changed task settings or state-machine constants | Watch a run with the GUI and without `--record`; compare the failing phase with the constants |
| Keyboard does nothing | The window has no focus, the script runs `--headless`, or **B** was not pressed after a reset | Click the Isaac Sim window, run with the GUI, and press **B** |
| Ctrl+C has no effect | The script hangs after an error | Stop it from another terminal: `pkill -9 -f scripts/teleop.py` (or `generate.py`) |
