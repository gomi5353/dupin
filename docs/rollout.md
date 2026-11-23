# Rollout (Policy Evaluation)

This guide describes how a trained policy is evaluated in the simulator with
`scripts/rollout.py`, and which input and output features a policy should
use. It extends step 5 of [Getting Started](getting_started.md) and
assumes that setup (step 0) is complete and that a checkpoint exists
([LeRobot Training](lerobot_training.md)).

Every section starts with one of two labels: **Required** (something to do
or follow) or **Background** (explanation without work items).

## 1. What rollout does

**Background**

Rollout runs **in the simulator environment** (`make launch-sim`, or the
Isaac Lab container), because it starts the simulator. It builds the environment from an evaluation task file, loads the
checkpoint, and runs a number of episodes (`--eval_rounds`). In each episode the policy
receives the camera images and joint state, predicts actions, and the
environment executes them until the task's success check passes or the episode
time runs out. At the end, rollout prints the success rate over all episodes. If
something goes wrong, for example an invalid flag value, it prints the error
and exits with a non-zero code.

The evaluation task files in `eval/` define the same tasks as the recording task ids, with the evaluation's own nominal object positions, randomization ranges, and episode length.

Rollout never reads the recording task configuration
(`packages/simulator/src/simulator/tasks/<task>/<task>_env_cfg.py`). A dataset
recorded under a modified configuration should still produce a policy that fits
the [contract of section 4](#4-policy-input-and-output-contract); settings that
change the recorded data are listed in
[Data Collection §4](data_collection.md#4-adjusting-the-collection).

| Task | Eval file | Episode length |
|---|---|---|
| Cup stacking | `eval/cup_stacking_eval.py` | 20 s |
| Cutlery arrangement | `eval/cutlery_arrangement_eval.py` | 40 s |
| Toy blocks collection | `eval/toy_blocks_collection_eval.py` | 60 s |

## 2. Get the checkpoint

**Required**

A checkpoint trained on the same machine is used directly:
`outputs/train/<job_name>/checkpoints/last/pretrained_model` works in the
simulator shell, and also in the container, where the repository folder is
mounted.

A checkpoint on the Hugging Face Hub is downloaded first:

```bash
hf download ${HF_USER}/<policy_name> --local-dir checkpoints/<policy_name> --repo-type model
```

`--revision <tag>` selects a tagged version, for example `--revision v1`.

## 3. Run rollout

**Required**

In the simulator shell, from the repository root:

```bash
python scripts/rollout.py \
    --task=eval/cup_stacking_eval.py \
    --policy_type=lerobot-diffusion \
    --policy_checkpoint_path=outputs/train/<job_name>/checkpoints/last/pretrained_model \
    --device=cuda \
    --enable_cameras \
    --eval_rounds=50
```

### Flag reference

| Flag | Description |
|---|---|
| `--task` | The task to evaluate on: an eval file under `eval/` (see the table above). It also accepts a registered task id or a `module:Class` reference; the module of a `module:Class` must be importable, for example by running from the repository root with `PYTHONPATH=.` ([section 5](#5-evaluating-a-self-defined-task)). |
| `--policy_type` | **Required.** `lerobot-<type>`, where `<type>` is the `--policy.type` used in training. Accepted values: `lerobot-diffusion`, `lerobot-act`, `lerobot-groot`, `lerobot-smolvla`, `lerobot-pi0`, `lerobot-pi05`, `lerobot-xvla`, `lerobot-vqbet`, `lerobot-tdmpc`, `lerobot-sac`. The options each policy needs in training are in [LeRobot Training §4](lerobot_training.md#4-policies). `lerobot-pi0` and `lerobot-pi05` need the patched `transformers` of their own environment; in the simulator environment, install it into a separate folder and put that folder first on `PYTHONPATH`: `uv pip install --python .venv-sim/bin/python --target ~/tf_fork --no-deps "transformers @ git+https://github.com/huggingface/transformers.git@fix/lerobot_openpi" "tokenizers>=0.21,<0.22"`, then `PYTHONPATH=$HOME/tf_fork:$PYTHONPATH python scripts/rollout.py ...` ; `lerobot-vqbet`, `lerobot-tdmpc`, and `lerobot-sac` are not supported: the first two take one camera only, which rollout does not feed, and the SAC rollout fails with image inputs. |
| `--policy_checkpoint_path` | **Required.** The checkpoint's `pretrained_model` folder (the one containing `config.json` and `model.safetensors`). |
| `--policy_action_horizon` | Number of actions executed from each inference before the policy runs again. Overrides `n_action_steps` from the checkpoint, which is used by default. It must be positive and at most the number of actions one inference predicts: `horizon - n_obs_steps + 1` for Diffusion Policy, `chunk_size` for ACT. ACT with temporal ensembling only accepts `1`. The values in use are printed at start. |
| `--policy_language_instruction` | Language instruction for language-conditioned policies, the same for every episode. By default each episode gets its own task text from the eval file (by name or by color), which is also logged as `instruction`. |
| `--device=cuda` | Runs simulation and inference on the GPU. |
| `--enable_cameras` | Enables camera rendering. Required, because the policy needs images. |
| `--eval_rounds` | Total number of evaluation episodes, counted over all environments (with `--num_envs 2` and `--eval_rounds 1`, one episode is run). `0` (the default) removes the time limit and never stops by itself: every episode runs until success or a manual reset (`R` in the GUI), then the next one starts, and no final success rate is printed. Use a positive value for a graded run. |
| `--episode_length_s` | Episode length in seconds. By default the task's graded length is used; the value in use is printed at start. |
| `--seed` | Seed of the object-position, lighting, and reference-form randomization (default `42`). A fixed seed compares policies on the same episodes. |
| `--num_envs` | Number of environments simulated in parallel (default 1); this many episodes run at once, and the total is still capped by `--eval_rounds`. |
| `--headless` | Runs without the GUI window. |

Besides the success rate, every rollout writes a per-episode log to
`outputs/rollout/<timestamp>.jsonl` (no flag needed). After a header line,
there is one line per episode:

```json
{"type": "episode", "episode": 3, "status": "failed",
 "factors": {"object_position": {"blue_cup": [8.91, 6.47, 1.0], "pink_cup": [9.02, 6.55, 1.0]},
             "lighting": {"rect_light": [6120.0, 4100.0]}},
 "instruction": "pick up the blue object and place it on the pink object."}
```

| Key | Content |
|---|---|
| `status` | `succeeded`, `failed`, or `aborted` (manual reset or closed app; ignored by the comparison) |
| `factors.object_position` | Initial position of every object in `tracked_object_names`, env-local, metres (axis Object Poses) |
| `factors.lighting` | `[intensity, color_temperature]` of every light in `recorded_light_names` at the start of the episode (axis Image Augmentations) |
| `instruction` | The task text the policy was given in this episode (axis Object Properties). Each environment gets its own episode text, built from the eval file's `task_description_template` in the reference form drawn for it (by name or by color); `--policy_language_instruction` replaces it for every episode. |

Object colors are not logged: the objects never change, only the
instruction does. The header line also records, once, the poses of what never
moves: `robot_base` and `fixed_objects` (`fixed_object_names`, such as the
plate), each as `pos` (env-local, metres) and `rot` (quaternion w, x, y, z).
They are for reference only; the comparison ignores them.

Putting the log into the training dataset's verification store and running
`verify` with the specs shows, for every axis, which rollout episodes
the dataset does not cover and where to record next
([Data Verification §5](data_verification.md#5-compare-the-dataset-with-a-rollout)):

```bash
uv run python scripts/store.py ingest --store stores/<repo_name> --rollout-log outputs/rollout/<timestamp>.jsonl
uv run python scripts/store.py verify --store stores/<repo_name> \
    --spec my_specs/object_pose.ttl my_specs/image_augmentation.ttl my_specs/object_property.ttl
```

## 4. Policy input and output contract

**Required**

Rollout feeds the policy the features below and applies the action below. A
policy that needs other inputs cannot be fed by rollout, and one trained with
other camera poses, controller settings, or action meanings may still run but
usually performs much worse. The dataset may contain other columns
(for example `object_position.*`, `lighting.*`, `object_color.*`); LeRobot ignores them because they are neither
`observation.*` nor `action`.

| Key | Direction | dtype | Shape | Meaning |
|---|---|---|---|---|
| `observation.state` | Input | float32 | `(9,)` | Measured joint positions in the order `panda_joint1..7, panda_finger_joint1, panda_finger_joint2`; arm in rad, fingers in m. |
| `observation.images.front` | Input | image/video | `(480, 640, 3)` | RGB uint8, height × width × channels, from the fixed front camera. |
| `observation.images.wrist` | Input | image/video | `(480, 640, 3)` | RGB uint8, from the wrist camera. |
| `action` | Output | float32 | `(8,)` | `[0:7]` absolute joint-position targets in rad; `[7]` gripper: `< 0` closes, `≥ 0` opens (±1 in recorded data). |

These are the features that the repository recorder writes
([Data Collection §5](data_collection.md#5-the-recorder-and-the-dataset)),
so a policy trained with default settings on such a dataset satisfies the
contract. Other features can be tried in training, but the policy that is
evaluated should follow this table.

## 5. Evaluating a self-defined task

**Background**

`--task` accepts three forms, resolved by `simulator.tasks.external.resolve_task`:

| Form | Example |
|---|---|
| Registered task id | `--task HCIS-CupStacking-SingleArm-v0` |
| Path to a `.py` file that registers a task when imported | `--task eval/cup_stacking_eval.py` |
| `module:Class` reference | `--task my_pkg.tasks.eval:MyEvalEnvCfg` |

The eval files use the second form. A self-defined task
([Adding a New Task](adding_new_task.md)) can be evaluated by its task id
directly, or by an own evaluation file in the same form as the files in
`eval/`: a task configuration class plus a `gym.register(...)` call with a
unique `TASK_ID`.
