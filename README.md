# Robot Learning from Evaluation Feedback

This repository provides a simulation-based learning framework for robot manipulation, covering three pick-and-place tasks. It includes tools for recording demonstrations in Isaac Lab using a provided state machine or keyboard teleoperation, verifying collected datasets, training imitation-learning policies with LeRobot, and evaluating trained policies using a provided rollout script.

> **Platform:** Linux with an NVIDIA GPU. Simulation runs on the host in the
> simulator environment (`make install-sim`, `make launch-sim`), or optionally
> in a Docker container; training and data verification run in the general
> environment.

**Start here: [Getting Started](docs/getting_started.md).** It walks through
the whole pipeline (setup, recording, verification, training, evaluation, and
diagnosis) and links to the detailed guides below.

## Python environments

Two [uv](https://docs.astral.sh/uv/) environments are kept separate, because
Isaac Sim needs older packages (NumPy 1.26, PyTorch 2.7) than training does.

| | General environment | Simulator environment |
|---|---|---|
| Folder | `.venv` | `.venv-sim` (or the Isaac Lab container) |
| Created by | `make install-gen` | `make install-sim` |
| Activated by | `make launch-gen` (a new shell; `exit` leaves it), or `source .venv/bin/activate` | `make launch-sim` (a new shell; `exit` leaves it), or `source scripts/sim_env.sh` |
| Prompt starts with | `general` | `simulator` |
| Used for | `lerobot-train`, `lerobot-dataset-viz`, `scripts/store.py`, `hf`, `make test` | `scripts/datagen/generate.py`, `scripts/teleop.py`, `scripts/rollout.py`: everything that starts Isaac Sim |

`uv run` inside the simulator shell still uses the general environment, and
prints a harmless `VIRTUAL_ENV ... does not match` warning. Setup is described
in [Getting Started §3](docs/getting_started.md#3-step-0-set-up-the-machine).

## Pipeline

| Step | Runs in | Guide |
|---|---|---|
| 0. Set up the machine | General and simulator environment | [Getting Started §3](docs/getting_started.md#3-step-0-set-up-the-machine) |
| 1. Choose a task | – | [Getting Started §4](docs/getting_started.md#4-step-1-choose-a-task) |
| 2. Record demonstrations | Simulator environment | [Data Collection](docs/data_collection.md) |
| 3. Inspect and verify the dataset | General environment | [Dataset Visualizer](docs/lerobot_dataset_visualizer.md), [Data Verification](docs/data_verification.md) |
| 4. Train a policy | General environment | [LeRobot Training](docs/lerobot_training.md) |
| 5. Evaluate the policy | Simulator environment | [Rollout](docs/rollout.md) |
| 6. Diagnose and iterate | General environment | [Getting Started §9](docs/getting_started.md#9-step-6-diagnose-and-iterate) |

## Documentation

| Document | Content |
|---|---|
| [Getting Started](docs/getting_started.md) | End-to-end walkthrough, key terms, setup, repository layout |
| [Data Collection](docs/data_collection.md) | State machine, keyboard teleoperation, adjustable settings, recorder |
| [LeRobot Dataset Visualizer](docs/lerobot_dataset_visualizer.md) | Inspecting recorded episodes |
| [Data Verification](docs/data_verification.md) | The verification store and the three generalization axes: writing specs, verifying, comparing rollouts, querying with SPARQL |
| [Object Poses: Object Position](docs/object_pose.md) | Axis Object Poses: where the objects start |
| [Image Augmentations: Lighting](docs/image_augmentation.md) | Axis Image Augmentations: light intensity and color temperature |
| [Object Properties: Referencing Objects by Color](docs/object_property.md) | Axis Object Properties: how the task text names the objects |
| [LeRobot Training](docs/lerobot_training.md) | Training flags, multi-GPU, checkpoint layout |
| [Rollout](docs/rollout.md) | Rollout flags, eval files, policy input and output contract |
| [Adding a New Task](docs/adding_new_task.md) | Writing and registering a self-defined task |

## License

MIT — see [LICENSE](LICENSE).
