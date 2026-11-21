# Robot Learning from Evaluation Feedback

Course project on learning-based robot manipulation in simulation. Each group
picks one of three pick-and-place tasks, records demonstrations for it in
Isaac Lab with a provided state machine or keyboard teleoperation, verifies
the collected data, trains an imitation-learning policy with LeRobot, and
evaluates the policy with the provided rollout script.

> **Platform:** Linux with an NVIDIA GPU. Simulation runs on the host in a
> separate simulator environment (`make install-sim`, `make launch-sim`), or
> optionally in a Docker container; training and data verification run in the
> host `uv` environment.

**Start here: [Getting Started](docs/getting_started.md).** It walks through
the whole pipeline (setup, recording, verification, training, evaluation, and
diagnosis) and links to the detailed guides below.

> **Fork first.** Students cannot push to this repository. Each group forks
> it on GitHub and clones the fork, so that its changes can be pushed and kept
> ([Getting Started §3.2](docs/getting_started.md#32-fork-and-clone-the-repository)).

## Pipeline

| Step | Runs in | Guide |
|---|---|---|
| 0. Set up the machine | Host | [Getting Started §3](docs/getting_started.md#3-step-0-set-up-the-machine) |
| 1. Choose a task | – | [Getting Started §4](docs/getting_started.md#4-step-1-choose-a-task) |
| 2. Record demonstrations | Simulator environment | [Data Collection](docs/data_collection.md) |
| 3. Inspect and verify the dataset | Host | [Dataset Visualizer](docs/lerobot_dataset_visualizer.md), [Data Verification](docs/data_verification.md) |
| 4. Train a policy | Host | [LeRobot Training](docs/lerobot_training.md) |
| 5. Evaluate the policy | Simulator environment | [Rollout](docs/rollout.md) |
| 6. Diagnose and iterate | Host | [Getting Started §9](docs/getting_started.md#9-step-6-diagnose-and-iterate) |

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
| [Adding a New Task](docs/adding_new_task.md) | Advanced Level: writing and registering an own task |
| [Adding a New Dataset Coverage Factor](docs/adding_coverage_factor.md) | Extending data verification to another axis or factor |

## License

MIT — see [LICENSE](LICENSE).
