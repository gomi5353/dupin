# Getting Started

This guide walks through the whole course-project pipeline, from installing the
software to diagnosing a trained policy. It is the main path through the
documentation. Each step explains what students do and why, gives the essential
commands, and links to a detailed guide where more explanation is needed.

Every section starts with one of two labels:

- **Required**: something that students must do or follow.
- **Background**: an explanation that the required steps depend on. It
  contains no work items.

## 1. Overview

**Background**

Students pick one manipulation task, collect demonstrations for it in
simulation with a provided data-collection method, verify the collected data, train an imitation-learning policy with
LeRobot, and evaluate the policy in the simulator.

| Step | What happens | Runs in | Output | Detailed guide |
|---|---|---|---|---|
| [0. Set up](#3-step-0-set-up-the-machine) | Install the software and start the simulator | Host | Working environments | This guide |
| [1. Choose a task](#4-step-1-choose-a-task) | Select one of the three predefined tasks | – | Task id | [Data Collection §1](data_collection.md#1-the-two-methods) |
| [2. Record](#5-step-2-record-demonstrations) | Record demonstrations with the state machine or keyboard teleoperation | Simulator environment | LeRobot v3 dataset | [Data Collection](data_collection.md) |
| [3. Inspect and verify](#6-step-3-inspect-and-verify-the-dataset) | Look at the episodes and check object-position coverage | Host | Coverage report | [Dataset Visualizer](lerobot_dataset_visualizer.md), [Object Position Coverage](object_position_coverage.md) |
| [4. Train](#7-step-4-train-a-policy) | Train a policy on the dataset | Host | Policy checkpoint | [LeRobot Training](lerobot_training.md) |
| [5. Evaluate](#8-step-5-evaluate-the-policy) | Run the policy in the simulator | Simulator environment | Success rate | [Rollout](rollout.md) |
| [6. Diagnose and iterate](#9-step-6-diagnose-and-iterate) | Relate failures to gaps in the data, then collect again | Host | Next round of data | [Object Position Coverage](object_position_coverage.md), [Adding a Coverage Factor](adding_coverage_factor.md) |

The steps form a loop rather than a single pass: a failed evaluation usually
leads back to step 2 with a better-targeted collection.

```text
record ──▶ inspect & verify ──▶ train ──▶ evaluate ──▶ diagnose
  ▲                                                       │
  └───────────────────────────────────────────────────────┘
```

Advanced Level groups that define their own task add one step before
recording: [Adding a New Task](adding_new_task.md).

## 2. Key terms

**Background**

| Term | Meaning in this project |
|---|---|
| Host machine | The GPU computer itself, with its own operating system, files, and GPU driver. |
| Docker container | An isolated environment that runs on the host. It shares the host's operating-system kernel and GPU, but has its own file system and installed software. It is started from an *image*, a packaged snapshot of an operating system and software built from the repository's `Dockerfile`. |
| Simulator environment | The Python environment that holds Isaac Sim 5.1.0, Isaac Lab, LeIsaac, and LeRobot at fixed versions, because the simulator depends on exact CUDA and Vulkan versions. Everything that starts the simulator runs here. By default it is `.venv-sim` on the host (`make install-sim`); the Isaac Lab container is an alternative with the same versions. |
| Isaac Lab container | The project's Docker container, an alternative simulator environment built from the `Dockerfile`. |
| `uv` environment | A Python virtual environment on the host, in `.venv`, created by [`uv`](https://docs.astral.sh/uv/) from the repository's `uv.lock`, so that every group gets the same package versions. Training and data verification run here. It cannot run Isaac Sim. |
| Isaac Sim | NVIDIA's robot simulator: physics, rendering, and sensors. |
| Isaac Lab | A robot-learning framework on top of Isaac Sim. It turns a simulated scene into an environment with `reset()` and `step()`. |
| LeIsaac | A library on top of Isaac Lab that connects it to LeRobot, including a recorder that writes LeRobot datasets. |
| LeRobot | Hugging Face's robot-learning library, used for the dataset format, training, and the policy models. |
| Task | One manipulation problem, for example cup stacking, with its scene, objects, randomization, and success check. |
| Task configuration | The Python class that defines a task for Isaac Lab. Each task is registered under a *task id*, such as `HCIS-CupStacking-SingleArm-v0`. |
| Isaac Lab environment | The running simulation of one task, created from the task id with `gym.make(...)`. `env.reset()` starts a new attempt with randomized objects; `env.step(action)` applies one action and advances the simulation by 1/60 s. |
| Episode | One attempt at the task, from one `env.reset()` to the next. |
| Data-collection method | How the robot is driven while demonstrations are recorded: a scripted **state machine** (`scripts/datagen/generate.py`) or **keyboard teleoperation** (`scripts/teleop.py`). It plays the expert that the policy imitates. |
| LeRobot recorder | The repository's `LeRobotRecorderManager`. It is attached to the environment and saves every successful episode, including the object positions that data verification needs. |
| Dataset | The recorded episodes of one task, stored in the LeRobot v3 format: per-frame numbers in Parquet tables, camera images as MP4 videos, and metadata in `meta/`. |
| Policy | The neural network trained on the dataset. It maps camera images and joint states to robot actions. |
| Checkpoint | A saved policy: its weights, configuration, and normalization statistics. |
| Rollout | Running a policy in the simulator for a number of episodes to measure its success rate. |
| Coverage check | The data-verification tool. It reports which regions of the table the initial object positions in a dataset cover, against a requirement that the group writes. |

## 3. Step 0: Set up the machine

**Required**

### 3.1 Requirements

- A Linux x86_64 machine with an NVIDIA RTX GPU (16 GB VRAM or more) and
  driver 570 or newer (`nvidia-smi` must work). Groups without one can use a
  cloud GPU; see
  [Cloud GPU Setup with GlowsAI](https://docs.google.com/presentation/d/1xPHm3gNNvLbX7aOWnY_xW5dfVLEa9XiBuskMbRsgxI8/edit?usp=sharing).
  Recording and rollout render camera images, which needs a GPU with RT cores
  (GeForce RTX, RTX A/Ada/PRO, L40S). Data-center GPUs without RT cores
  (A100, H100, V100) cannot run them; such GPUs are only usable for training.
- Ubuntu 22.04 or newer (the Isaac Sim packages need glibc 2.35 or newer),
  about 40 GB of free disk space for the simulator environment, and a
  desktop session for the Isaac Sim window.
- Git with submodule support.
- Only for the container alternative
  ([section 3.5](#35-launch-the-simulator)): Docker and the
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html),
  which lets containers use the GPU (`docker run --gpus ...`). Without it,
  the launch commands fail with
  `could not select device driver "" with capabilities: [[gpu]]`.
- [`uv`](https://docs.astral.sh/uv/), the Python package manager. The project
  uses Python 3.11 (`.python-version`).
- A Hugging Face account and an access token
  (<https://huggingface.co/docs/hub/en/security-tokens>).

On the GlowsAI VNC desktop, commands run as the `glows` user:
`su - glows`.

For the container alternative, Docker is installed on Ubuntu as follows:

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin

# allow Docker without sudo
sudo usermod -aG docker $USER
newgrp docker
docker --version
```

The NVIDIA Container Toolkit is installed as follows:

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# check: the GPU must be listed
docker run --rm --gpus all ubuntu nvidia-smi
```

### 3.2 Fork and clone the repository

Students cannot push to the course repository. Every group therefore works in
its own **fork**: a copy of the repository under a group member's GitHub
account. The group's work (state-machine and configuration changes, coverage
specifications, new verification factors) is committed and
pushed to the fork, so it is kept, shared within the group, and handed in
from there. A plain clone of the course repository would leave all changes on
one machine, with no place to push them.

1. On the course repository's GitHub page, one group member clicks **Fork**
   and creates the fork under their own account. The other members are added
   as collaborators (fork **Settings → Collaborators**).
2. Every member clones the **fork**, not the course repository:

   ```bash
   git clone https://github.com/<github-user>/EDRLP
   cd EDRLP
   git remote add upstream https://github.com/HCIS-lab/EDRLP
   git remote -v   # origin = the fork (push here), upstream = the course repository
   ```

3. Work is committed and pushed to the fork as usual:

   ```bash
   git add <files>
   git commit -m "<message>"
   git push origin main
   ```

4. When the course repository is updated (fixes, new evaluation files), the
   group merges the updates into its fork:

   ```bash
   git fetch upstream
   git merge upstream/main
   git push origin main
   ```

   Conflicts appear only where the group changed the same lines as the update.

Notes:

- Datasets, checkpoints, and training outputs (`datasets/`, `checkpoints/`,
  `outputs/`) are ignored by Git. They are shared through the Hugging Face Hub
  ([step 3](#6-step-3-inspect-and-verify-the-dataset) and
  [LeRobot Training §7](lerobot_training.md#7-upload-a-checkpoint)), not
  through the fork.
- The Isaac Lab submodule (`dependencies/IsaacLab/`) is fetched by
  `make install-sim` or the first `make launch-isaaclab`
  ([section 3.5](#35-launch-the-simulator)).

### 3.3 Install the host environment

From the root of the cloned fork:

```bash
uv sync                     # creates .venv with the locked package versions
source .venv/bin/activate   # makes lerobot-train, hf, ... available
```

### 3.4 Log in to Hugging Face

```bash
hf auth login --token <YOUR_HF_TOKEN>
export HF_USER=<your-huggingface-username>
```

Commands in all guides use `${HF_USER}`. The variable is set again in every
new terminal, including the simulator shell.

### 3.5 Launch the simulator

**Default: on the host.** The simulator stack is installed once into a
separate environment, `.venv-sim`, with the same package versions as the
container (`requirements-sim.txt`):

```bash
make install-sim   # once; downloads about 20 GB, takes a while
make launch-sim    # opens a shell with .venv-sim active, in the repository root
```

`make launch-sim` opens a new shell; `exit` leaves it. In an existing
terminal, `source scripts/sim_env.sh` does the same. The shell sets the
variables the container sets: it accepts the Isaac Sim license, and points
LeRobot's dataset folder (`HF_LEROBOT_HOME`) to `datasets/`, so a recorded
dataset appears in `datasets/<hf_user>/<name>`. The first start of Isaac Sim
compiles shaders and takes several minutes.
The Isaac Sim window opens on the display in `DISPLAY`, so the shell is
started from a terminal on the machine's desktop. If the window cannot open
(`GLFW initialization failed` in the log), `DISPLAY` points to no running
desktop; commands with `--headless` run without a window.

`.venv-sim` and `.venv` are kept separate because Isaac Sim needs older
versions (NumPy 1.26, PyTorch 2.7) than the `uv` environment. Training and
verification commands (`uv run ...`, `lerobot-train`) are run outside the
simulator shell.

**Alternative: the Isaac Lab container.** For machines where the host
installation is not possible, for example the GlowsAI cloud machines:

| Command | Machine |
|---|---|
| `make launch-isaaclab` | Local machine with an NVIDIA GPU |
| `make launch-isaaclab-glowsai-4090` | GlowsAI RTX 4090 (VNC display :1) |
| `make launch-isaaclab-glowsai-l40s` | GlowsAI L40S (VNC display :1) |

The first launch initializes the Git submodules and builds the image, which
takes a while. On success, the terminal is a shell inside the container, in the
repository root. All guides work the same in this shell.

What the container shares with the host:

- The repository folder is mounted at `/workspace/physicalai`. A file edited on
  the host is immediately visible inside the container, and the other way
  round.
- `datasets/` on the host is mounted as LeRobot's dataset folder
  (`~/.cache/huggingface/lerobot`), so a dataset recorded in the container
  appears in `datasets/<hf_user>/<name>` on the host.
- The host's `.venv` and `.venv-sim` are hidden inside the container; the
  container has its own Python installation.
- The container is deleted when it exits. Anything written outside the
  repository folder is lost.

Other `make` targets:

| Target | Purpose |
|---|---|
| `make install` | Initialize submodules and run `uv sync` |
| `make install-sim` | Install the simulator environment `.venv-sim` on the host |
| `make launch-sim` | Open a shell with `.venv-sim` active |
| `make build-isaaclab` | Build the image without launching it |
| `make check-isaaclab-gpu` | Check that the image sees the GPU, Vulkan, and CUDA |
| `make fix-perms` | Give files that the container wrote under `datasets/` and `outputs/` back to the host user |
| `make test` | Run the host-side tests |

### 3.6 Where to run what

| Work | Environment | Reason |
|---|---|---|
| Recording, rollout | Simulator environment (`make launch-sim`, or the container) | Both start the simulator, which needs the pinned GPU and rendering stack. |
| Training | Host `uv` environment | The container adds file-access and GPU pass-through overhead. |
| Verification store (coverage check, queries), dataset inspection, tests | Host `uv` environment | They only read files and never start the simulator. |

### 3.7 Repository layout

**Background**

| Path | Content |
|---|---|
| `packages/simulator/` | Task configurations, the shared Franka template, randomization events, assets, the state machines (`simulator.datagen.state_machine`), the keyboard device (`simulator.devices`), and the LeRobot recorder (`simulator.utils.lerobot_recorder`). |
| `packages/data_verification/` | The verification store: dataset and rollout-log readers, OWL ontology, SHACL rules, and one verifier per generalization axis (object position), registered in `registry.py`. |
| `eval/` | Public evaluation task files, one per task, used by rollout. |
| `scripts/datagen/generate.py`, `scripts/teleop.py` | The two data-collection methods: state machine and keyboard teleoperation. |
| `scripts/rollout.py` | Runs a trained policy in the simulator and reports the success rate. |
| `scripts/store.py` | Command-line interface of the verification store (`ingest`, `init`, `attach`, `verify`, `explore`, `query`). |
| `queries/` | Example SPARQL queries over a store. |
| `tests/` | Host-side tests (`make test`). |
| `dependencies/IsaacLab/` | Isaac Lab as a Git submodule. Not edited. |
| `requirements-sim.txt`, `scripts/install_sim_host.sh`, `scripts/sim_env.sh` | The host simulator environment: pinned versions, installer, and shell setup. |
| `Dockerfile`, `Makefile` | The container image, and the commands that install and launch both simulator environments. |
| `datasets/`, `outputs/`, `checkpoints/`, `stores/` | Recorded datasets, training and rollout outputs, verification stores; not tracked by Git. |

## 4. Step 1: Choose a task

**Required**

Each group works on one of three predefined tasks:

| Task | Task id (recording) | Eval file (rollout) | Episode length |
|---|---|---|---|
| Cup stacking | `HCIS-CupStacking-SingleArm-v0` | `eval/cup_stacking_eval.py` | 20 s |
| Cutlery arrangement | `HCIS-CutleryArrangement-SingleArm-v0` | `eval/cutlery_arrangement_eval.py` | 40 s |
| Toy blocks collection | `HCIS-ToyBlocksCollection-SingleArm-v0` | `eval/toy_blocks_collection_eval.py` | 60 s |

The task id names the environment used for recording. The eval file defines
the environment used for rollout: the same task with the evaluation's own
object positions and randomization. The episode length is the time a policy gets
during evaluation.

[Data Collection §1](data_collection.md#1-the-two-methods) lists each task's
scene objects, success check, and task text. Advanced Level groups that
define their own task follow [Adding a New Task](adding_new_task.md) and then
continue with step 2.

## 5. Step 2: Record demonstrations

**Required**

Demonstrations are recorded in the simulator shell with one of two provided
methods: the **state machine**, which solves the task by script, or **keyboard
teleoperation**, where a group member drives the robot. For example:

```bash
python scripts/datagen/generate.py \
    --task HCIS-CupStacking-SingleArm-v0 --num_demos 50 \
    --device cuda --enable_cameras --headless \
    --record --use_lerobot_recorder \
    --lerobot_dataset_repo_id ${HF_USER}/cup-stacking-demos
```

Groups may adjust the state machine and the task configuration, for example
the randomization ranges or the lighting, as long as the rollout contract is
kept. How many episodes to record, and which object positions they must
cover, is the group's decision and is revisited in step 6.
[Data Collection](data_collection.md) describes both methods, what may be
adjusted, and the recorder.

## 6. Step 3: Inspect and verify the dataset

**Required**

Before training, every dataset is checked in two ways, on the host:

1. **Visual inspection.** A few episodes are watched with the
   [LeRobot Dataset Visualizer](lerobot_dataset_visualizer.md), to confirm
   that the camera views, the motion, and the episode boundaries are correct.
2. **Object-position coverage check.** The dataset is put into a verification
   store (RDF files under `stores/<dataset_repo>/`). The group writes a short
   requirement file that states which region of the table the initial object
   positions must cover, how finely to divide it, and how many episodes each
   cell needs, and attaches it to the store. `verify` then reports the cells
   with too few episodes and keeps the results in the store, where they can be
   queried with SPARQL:

   ```bash
   uv run python scripts/store.py ingest --dataset datasets/${HF_USER}/cup-stacking-demos
   uv run python scripts/store.py init --store stores/cup-stacking-demos --out specs/cup_stacking.ttl
   # fill in the TODO values, then:
   uv run python scripts/store.py attach --store stores/cup-stacking-demos --spec specs/cup_stacking.ttl
   uv run python scripts/store.py verify --store stores/cup-stacking-demos
   ```

   [Object Position Coverage](object_position_coverage.md) explains how to choose the
   values, how to read the report, and how to query the store.

Running the check is required for every dataset. Its values are the group's
own decision, and the report is evidence for the group's analysis, not a grade.

To train from the Hugging Face Hub, or to use the hosted visualizer, the
dataset is uploaded first:

```bash
hf upload ${HF_USER}/cup-stacking-demos datasets/${HF_USER}/cup-stacking-demos --repo-type dataset
```

## 7. Step 4: Train a policy

**Required**

Training runs on the host with LeRobot's `lerobot-train`:

```bash
lerobot-train \
  --dataset.repo_id=${HF_USER}/cup-stacking-demos \
  --dataset.root=datasets/${HF_USER}/cup-stacking-demos \
  --policy.type=diffusion \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --output_dir=outputs/train/cup_stacking_v1 \
  --job_name=cup_stacking_v1
```

Training takes from a few hours to more than a day, depending on the dataset
size and the GPU. The result is a checkpoint folder,
`outputs/train/cup_stacking_v1/checkpoints/last/pretrained_model`.
[LeRobot Training](lerobot_training.md) covers the policy choice, all flags,
multi-GPU training, the checkpoint layout, and uploading.

## 8. Step 5: Evaluate the policy

**Required**

Rollout runs in the simulator shell. It loads the checkpoint, runs it for a
number of episodes under the task's eval file, and prints the success rate:

```bash
python scripts/rollout.py \
    --task=eval/cup_stacking_eval.py \
    --policy_type=lerobot-diffusion \
    --policy_checkpoint_path=outputs/train/cup_stacking_v1/checkpoints/last/pretrained_model \
    --device=cuda --enable_cameras \
    --eval_rounds=20
```

[Rollout](rollout.md) lists all flags and the input and output
features that a submitted policy must use.

## 9. Step 6: Diagnose and iterate

**Required**

A failed rollout does not by itself mean that the policy architecture or the
training settings are at fault. Imitation-learning policies tend to work only
in situations that the demonstrations cover well. The diagnosis therefore
starts with the data:

1. Watch the failed rollout episodes and note where the objects started.
2. Compare those positions with the coverage report from step 3. A failure in a
   cell with few or no episodes points to a gap in the data.
3. Collect more episodes where the report shows gaps, run the check again, and
   retrain.

Coverage is a diagnostic, not a proof: an uncovered region is evidence of a
possible gap in the data, while complete coverage does not guarantee success.
A group that suspects another factor, such as lighting or object orientation,
can select it from the STAR-Gen taxonomy or another paper and follow
[Adding a New Dataset Coverage Factor](adding_coverage_factor.md).

## 10. Documentation map

**Background**

| Guide | Extends | Content |
|---|---|---|
| [Data Collection](data_collection.md) | Steps 1–2 | Tasks, state machine, keyboard teleoperation, adjustable settings, recorder |
| [LeRobot Dataset Visualizer](lerobot_dataset_visualizer.md) | Step 3 | Inspecting episodes in the browser or locally |
| [Object Position Coverage](object_position_coverage.md) | Steps 3 and 6 | Writing the coverage requirement, running the check, reading the report |
| [LeRobot Training](lerobot_training.md) | Step 4 | Training flags, multi-GPU, checkpoint layout, upload |
| [Rollout](rollout.md) | Step 5 | Rollout flags, eval files, policy input and output contract |
| [Adding a New Task](adding_new_task.md) | Step 1, Advanced Level | Writing, registering, and testing a new task configuration |
| [Adding a New Dataset Coverage Factor](adding_coverage_factor.md) | Step 6 | Extending data verification to another factor |
