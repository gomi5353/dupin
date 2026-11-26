# Getting Started

This guide walks through the whole pipeline, from installing the software to
diagnosing a trained policy. It is the main path through the documentation.
Each step explains what to do and why, gives the essential commands, and links
to a detailed guide where more explanation is needed.

Every section starts with one of two labels:

- **Required**: something to do or follow.
- **Background**: an explanation that the required steps depend on. It
  contains no work items.

## 1. Overview

**Background**

One manipulation task is picked, demonstrations for it are collected in
simulation with a provided data-collection method, the collected data is
verified, an imitation-learning policy is trained with LeRobot, and the policy
is evaluated in the simulator.

| Step | What happens | Runs in | Output | Detailed guide |
|---|---|---|---|---|
| [0. Set up](#3-step-0-set-up-the-machine) | Install the software and start the simulator | General environment | Working environments | This guide |
| [1. Choose a task](#4-step-1-choose-a-task) | Select one of the three predefined tasks | – | Task id | [Data Collection §1](data_collection.md#1-the-two-methods) |
| [2. Record](#5-step-2-record-demonstrations) | Record demonstrations with the state machine or keyboard teleoperation | Simulator environment | LeRobot v3 dataset | [Data Collection](data_collection.md) |
| [3. Inspect and verify](#6-step-3-inspect-and-verify-the-dataset) | Look at the episodes and check them against specs on three generalization axes | General environment | Verification report | [Dataset Visualizer](lerobot_dataset_visualizer.md), [Data Verification](data_verification.md) |
| [4. Train](#7-step-4-train-a-policy) | Train a policy on the dataset | General environment | Policy checkpoint | [LeRobot Training](lerobot_training.md) |
| [5. Evaluate](#8-step-5-evaluate-the-policy) | Run the policy in the simulator | Simulator environment | Success rate | [Rollout](rollout.md) |
| [6. Diagnose and iterate](#9-step-6-diagnose-and-iterate) | Relate failures to gaps in the data, then collect again | General environment | Next round of data | [Data Verification §5](data_verification.md#5-compare-the-dataset-with-a-rollout) |

The steps form a loop rather than a single pass: a failed evaluation usually
leads back to step 2 with a better-targeted collection.

```text
record ──▶ inspect & verify ──▶ train ──▶ evaluate ──▶ diagnose
  ▲                                                       │
  └───────────────────────────────────────────────────────┘
```

A self-defined task adds one step before recording:
[Adding a New Task](adding_new_task.md).

## 2. Key terms

**Background**

| Term | Meaning in this project |
|---|---|
| Host machine | The GPU computer itself, with its own operating system, files, and GPU driver. |
| Docker container | An isolated environment that runs on the host. It shares the host's operating-system kernel and GPU, but has its own file system and installed software. It is started from an *image*, a packaged snapshot of an operating system and software built from the repository's `Dockerfile`. |
| General environment | The Python environment `.venv` for everything that does not start the simulator ([README](../README.md#python-environments)). |
| Simulator environment | The Python environment `.venv-sim` (or the Isaac Lab container) for everything that starts the simulator ([README](../README.md#python-environments)). |
| Isaac Lab container | The repository's Docker container, an alternative simulator environment built from the `Dockerfile`. |
| Isaac Sim | NVIDIA's robot simulator: physics, rendering, and sensors. |
| Isaac Lab | A robot-learning framework on top of Isaac Sim. It turns a simulated scene into an environment with `reset()` and `step()`. |
| LeIsaac | A library on top of Isaac Lab that connects it to LeRobot, including a recorder that writes LeRobot datasets. |
| LeRobot | Hugging Face's robot-learning library, used for the dataset format, training, and the policy models. |
| Task | One manipulation problem, for example cup stacking, with its scene, objects, randomization, and success check. |
| Task configuration | The Python class that defines a task for Isaac Lab. Each task is registered under a *task id*, such as `HCIS-CupStacking-SingleArm-v0`. |
| Isaac Lab environment | The running simulation of one task, created from the task id with `gym.make(...)`. `env.reset()` starts a new attempt with randomized objects; `env.step(action)` applies one action and advances the simulation by 1/60 s. |
| Episode | One attempt at the task, from one `env.reset()` to the next. |
| Data-collection method | How the robot is driven while demonstrations are recorded: a scripted **state machine** (`scripts/datagen/generate.py`) or **keyboard teleoperation** (`scripts/teleop.py`). It plays the expert that the policy imitates. |
| LeRobot recorder | The repository's `LeRobotRecorderManager`. It is attached to the environment and saves every successful episode, including the columns that data verification needs: object positions, lighting, object colors, and each episode's task text. |
| Dataset | The recorded episodes of one task, stored in the LeRobot v3 format: per-frame numbers in Parquet tables, camera images as MP4 videos, and metadata in `meta/`. |
| Policy | The neural network trained on the dataset. It maps camera images and joint states to robot actions. |
| Checkpoint | A saved policy: its weights, configuration, and normalization statistics. |
| Rollout | Running a policy in the simulator for a number of episodes to measure its success rate. |
| Generalization axis | One way in which the evaluation conditions can differ from the demonstrations, taken from the STAR-Gen taxonomy. The repository checks three axes, each with one *factor*: **Object Poses** (factor object position), **Image Augmentations** (factor lighting), and **Object Properties** (factor referencing objects by color). |
| Spec | A Turtle (`.ttl`) file that states what a dataset should cover on one axis, for example the table area per object and the episodes per cell. Each predefined task ships one spec per axis in `specs/<task>/`, with every value left as `TODO`; the specs are copied and filled in. |
| Data verification | The check of a dataset against specs (`scripts/store.py verify`). It reports, per axis, where the dataset has too few episodes, and compares rollouts with the dataset (`scripts/store.py compare`). |

## 3. Step 0: Set up the machine

**Required**

### 3.1 Requirements

- A Linux x86_64 machine with an NVIDIA RTX GPU (16 GB VRAM or more) and
  driver 570 or newer (`nvidia-smi` must work). Without one, a cloud GPU can be
  used; see
  [Cloud GPU Setup with GlowsAI](https://docs.google.com/presentation/d/1xPHm3gNNvLbX7aOWnY_xW5dfVLEa9XiBuskMbRsgxI8/edit?usp=sharing).
  Recording and rollout render camera images, which needs a GPU with RT cores
  (GeForce RTX, RTX A/Ada/PRO, L40S). Data-center GPUs without RT cores
  (A100, H100, V100) cannot run them; such GPUs are only usable for training.
- Ubuntu 22.04 or newer (the Isaac Sim packages need glibc 2.35 or newer),
  about 40 GB of free disk space for the simulator environment, and a
  desktop session for the Isaac Sim window.
- Git with submodule support.
- Only for the container alternative
  ([section 3.4](#34-launch-the-simulator)): Docker and the
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

### 3.2 Install the general environment

From the repository root:

```bash
make install-gen   # once; creates .venv with the locked package versions
make launch-gen    # opens a shell with .venv active (lerobot-train, hf, ... available), in the repository root
```

`make install-gen` first checks that `uv` and `ffmpeg` are installed (it stops with the install command if not), initializes the submodules,
then runs `uv sync`. Without `ffmpeg`, `lerobot-train` and `lerobot-dataset-viz` fail when they decode videos; install it with
`sudo apt-get install -y ffmpeg`.

`make launch-gen` opens a new shell, whose prompt starts with `(general)`; `exit` leaves it. To activate in the current shell instead, run
`source .venv/bin/activate`; `deactivate` leaves it. The simulator environment is installed separately in
[section 3.4](#34-launch-the-simulator).

### 3.3 Log in to Hugging Face and Weights & Biases

Log in to **both** services before any training or rollout. Many errors that
look unrelated (a model that cannot be downloaded, a training run that stops at
start-up) come from a missing login.

```bash
hf auth login --token <YOUR_HF_TOKEN>   # also needed to upload datasets and policies
wandb login                             # asks for the Weights & Biases API key
export HF_USER=<your-huggingface-username>
```

Commands in all guides use `${HF_USER}` and `--wandb.enable=true`. The variable
is set again in every new terminal, including the simulator shell.

Some pretrained models are gated: the Hugging Face account must first accept
the model's terms on its web page (for example `google/paligemma-3b-pt-224`,
used by the `pi0` and `pi05` policies; see
[LeRobot Training §4](lerobot_training.md#4-policies)).

### 3.4 Launch the simulator

**Default: on the host.** The simulator stack is installed once into the
simulator environment `.venv-sim`, with the same package versions as the
container (`requirements-sim.txt`):

```bash
make install-sim   # once; downloads about 20 GB, takes a while
make launch-sim    # opens a shell with .venv-sim active, in the repository root
```

`make launch-sim` opens a new shell, whose prompt starts with `(simulator)`;
`exit` leaves it. In an existing terminal, `source scripts/sim_env.sh` does the
same. The shell sets the
variables the container sets: it accepts the Isaac Sim license, and points
LeRobot's dataset folder to `datasets/`, so a recorded
dataset appears in `datasets/${HF_USER}/<repo_name>`. The first start of Isaac Sim
compiles shaders and takes several minutes.
The Isaac Sim window opens on the display in `DISPLAY`, so the shell is
started from a terminal on the machine's desktop. If the window cannot open
(`GLFW initialization failed` in the log), `DISPLAY` points to no running
desktop; commands with `--headless` run without a window.

**On a Brev cloud VM** (Isaac Lab image), run the setup script once before
`make install-sim`:

```bash
bash scripts/setup_brev_host.sh
```

A terminal can be opened in two places, and they are different file systems:
the VM's SSH shell (prompt `ubuntu@brev-...`), and a terminal on the noVNC
desktop (`localhost:6080`), which runs inside the Isaac Lab Docker container
(prompt `root@<hash>`; `su ubuntu` switches to `ubuntu`). A folder created in
one is not visible in the other. Keep the repository and `make launch-sim` in
one of them; the noVNC terminal needs no display setup, because `DISPLAY=:0` is
local there. The script works in both places. It checks each item and changes
only what is missing, so as `ubuntu` in the container it only reports; if it
asks for root, run it once as `root`. It covers the Vulkan and OpenGL
libraries, `uv`, the build tools (`cmake`, `gcc`, `make`; `make install-sim` fails without them), and two files a fresh VM host lacks: the NVIDIA EGL vendor
file (otherwise Isaac Sim stops with `ERROR_INCOMPATIBLE_DRIVER`) and the
`libcuda.so` link (otherwise PhysX silently runs on the CPU and the simulation
hangs after the first scene messages). In the SSH shell it also connects the
noVNC desktop to the host, so a run without `--headless` and with
`DISPLAY=:0` appears at `localhost:6080`; that connection is lost on reboot, run
the script again to restore it.

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
  appears in `datasets/${HF_USER}/<repo_name>` on the host.
- The host's `.venv` and `.venv-sim` are hidden inside the container; the
  container has its own Python installation.
- The container is deleted when it exits. Anything written outside the
  repository folder is lost.

Other `make` targets:

| Target | Purpose |
|---|---|
| `make install-gen` | Install the general environment `.venv` (checks `uv` and `ffmpeg`, initializes the submodules, runs `uv sync`) |
| `make launch-gen` | Open a shell with `.venv` active |
| `make install-sim` | Install the simulator environment `.venv-sim` on the host (initializes the submodules first) |
| `make launch-sim` | Open a shell with `.venv-sim` active |
| `make build-isaaclab` | Build the image without launching it |
| `make check-isaaclab-gpu` | Check that the image sees the GPU, Vulkan, and CUDA |
| `make fix-perms` | Give files that the container wrote under `datasets/` and `outputs/` back to the host user |
| `make test` | Run the host-side tests |

### 3.5 Where to run what

| Work | Environment | Reason |
|---|---|---|
| Recording, rollout | Simulator environment (`make launch-sim`, or the container) | Both start the simulator, which needs the pinned GPU and rendering stack. |
| Training | General environment | The container adds file-access and GPU pass-through overhead. |
| Verification store (`verify`, queries), dataset inspection, tests | General environment | They only read files and never start the simulator. |

### 3.6 Repository layout

**Background**

| Path | Content |
|---|---|
| `packages/simulator/` | Task configurations, the shared Franka template, randomization events, assets, the state machines (`simulator.datagen.state_machine`), the keyboard device (`simulator.devices`), and the LeRobot recorder (`simulator.utils.lerobot_recorder`). |
| `packages/data_verification/` | The verification store: dataset and rollout-log readers, OWL ontology, SHACL rules, and one verifier per generalization axis under `axes/` (`object_pose`, `image_augmentation`, `object_property`), registered in `registry.py`. |
| `eval/` | Public evaluation task files, one per task, used by rollout. |
| `scripts/datagen/generate.py`, `scripts/teleop.py` | The two data-collection methods: state machine and keyboard teleoperation. |
| `scripts/rollout.py` | Runs a trained policy in the simulator and reports the success rate. |
| `scripts/store.py` | Command-line interface of the verification store (`ingest`, `verify`, `compare`, `explore`, `query`). |
| `specs/<task>/` | The shipped specs of each predefined task, one per axis: `object_pose.ttl`, `image_augmentation.ttl`, `object_property.ttl`. Every value is `TODO`; they are copied and filled in. |
| `queries/` | Example SPARQL queries over a store. |
| `tests/` | Tests of the general environment (`make test`). |
| `dependencies/IsaacLab/` | Isaac Lab as a Git submodule. Not edited. |
| `requirements-sim.txt`, `scripts/install_sim_host.sh`, `scripts/sim_env.sh` | The host simulator environment: pinned versions, installer, and shell setup. |
| `Dockerfile`, `Makefile` | The container image, and the commands that install and launch both simulator environments. |
| `datasets/`, `outputs/`, `checkpoints/` | Recorded datasets, training and rollout outputs (including the rollout logs in `outputs/rollout/`); not tracked by Git. |
| `stores/` | Verification stores, one per dataset (`stores/<repo_name>/`), written by `scripts/store.py`; not tracked by Git. |

## 4. Step 1: Choose a task

**Required**

Three tasks are predefined:

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
scene objects, success check, and task text. A self-defined task follows
[Adding a New Task](adding_new_task.md) and then continues with step 2.

## 5. Step 2: Record demonstrations

**Required**

Demonstrations are recorded in the simulator shell with one of two provided
methods: the **state machine**, which solves the task by script, or **keyboard
teleoperation**, where a person drives the robot. For example:

```bash
python scripts/datagen/generate.py \
    --task HCIS-CupStacking-SingleArm-v0 --num_demos 50 \
    --device cuda --enable_cameras --headless \
    --record --use_lerobot_recorder \
    --lerobot_dataset_repo_id ${HF_USER}/cup-stacking-demos
```

| Argument | Meaning |
|---|---|
| `--task` | Task id of the task to record |
| `--num_demos` | Number of attempts; only successful ones are saved |
| `--device cuda --enable_cameras` | Simulate on the GPU, and render the camera images that the dataset contains |
| `--headless` | Run without the Isaac Sim window |
| `--record --use_lerobot_recorder` | Save the attempts as a LeRobot dataset |
| `--lerobot_dataset_repo_id` | Repo id `${HF_USER}/<repo_name>`; the dataset is written to `datasets/${HF_USER}/<repo_name>` |

The state machine and the task configuration can be adjusted, for example the
object randomization ranges, the light ranges, or how often the task text
refers to the objects by color. How many episodes to record, and which
positions, lights, and wordings they cover, is open and is revisited in step 6.
[Data Collection](data_collection.md) describes both methods, the adjustable
settings, and the recorder.

## 6. Step 3: Inspect and verify the dataset

**Required**

Before training, a dataset is checked in two ways, in the general environment:

1. **Visual inspection.** A few episodes are watched with the
   [LeRobot Dataset Visualizer](lerobot_dataset_visualizer.md), to confirm
   that the camera views, the motion, and the episode boundaries are correct.
2. **Data verification.** The dataset is put into a verification store (RDF
   files under `stores/<repo_name>/`). The shipped specs of the task are
   copied from `specs/<task>/`, one file per generalization axis, and every
   `TODO` is replaced with own values: for example the table area each
   object must cover and the episodes per cell (Object Poses), the light ranges
   (Image Augmentations), or the phrases that refer to each object by name and
   by color (Object Properties). `verify` checks the dataset against every
   spec it is given, reports where the dataset has too few episodes, and keeps
   the specs and results in the store, where they can be queried with SPARQL:

   ```bash
   uv run python scripts/store.py ingest --dataset datasets/${HF_USER}/cup-stacking-demos
   mkdir -p my_specs
   cp specs/cup_stacking/object_pose.ttl my_specs/
   # replace every "TODO" in the copy, then:
   uv run python scripts/store.py verify --store stores/cup-stacking-demos \
       --spec my_specs/object_pose.ttl
   ```

   To check another axis, copy its spec the same way and append it after the
   first: `--spec my_specs/object_pose.ttl my_specs/image_augmentation.ttl`
   (the same as repeating `--spec`).

   | Argument | Meaning |
   |---|---|
   | `ingest --dataset` | Reads the dataset folder into a new store, named after the folder (`stores/cup-stacking-demos`) |
   | `verify --store` | The store to check |
   | `verify --spec` | The filled-in spec file; append more files, one per axis, to check more axes. An axis without a spec file is not checked |

   [Data Verification](data_verification.md)
   explains the workflow and the report; the axis pages
   ([Object Poses](object_pose.md), [Image Augmentations](image_augmentation.md),
   [Object Properties](object_property.md)) explain how to choose each
   spec's values.

The values in the specs are chosen freely. The report is a diagnostic that
supports the analysis, not a pass/fail grade.

To train from the Hugging Face Hub, or to use the hosted visualizer, the
dataset is uploaded first:

```bash
hf upload ${HF_USER}/cup-stacking-demos datasets/${HF_USER}/cup-stacking-demos --repo-type dataset
```

`hf upload <repo id> <local folder>` copies the folder to the Hub repository,
and `--repo-type dataset` marks it as a dataset.

## 7. Step 4: Train a policy

**Required**

Training runs in the general environment with LeRobot's `lerobot-train`:

```bash
lerobot-train \
  --dataset.repo_id=${HF_USER}/<repo_name> \
  --dataset.root=datasets/${HF_USER}/<repo_name> \
  --policy.type=diffusion \
  --policy.device=cuda \
  --policy.repo_id=${HF_USER}/<policy_name> \
  --output_dir=outputs/train/<job_name> \
  --job_name=<job_name> \
  --wandb.enable=true
```

| Argument | Meaning |
|---|---|
| `--dataset.repo_id`, `--dataset.root` | The dataset's repo id and its local folder |
| `--policy.type`, `--policy.device` | Policy type ([LeRobot Training §4](lerobot_training.md#4-policies)) and the device to train on |
| `--policy.repo_id` | Hub repository to which the trained policy is pushed (needs `hf auth login`); `--policy.push_to_hub=false` skips the push |
| `--output_dir`, `--job_name` | Folder for the results, which must not exist yet, and the run's name |
| `--wandb.enable` | Logs the training curves to Weights & Biases (needs `wandb login`) |

Training takes from a few hours to more than a day, depending on the dataset
size and the GPU. The result is a checkpoint folder,
`outputs/train/<job_name>/checkpoints/last/pretrained_model`.
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
    --policy_checkpoint_path=outputs/train/<job_name>/checkpoints/last/pretrained_model \
    --device=cuda --enable_cameras \
    --eval_rounds=50
```

| Argument | Meaning |
|---|---|
| `--task` | The evaluation file of the task |
| `--policy_type`, `--policy_checkpoint_path` | The policy architecture (`lerobot-<type>`) and the checkpoint's `pretrained_model` folder |
| `--device cuda --enable_cameras` | Simulate on the GPU, and render the images that the policy needs |
| `--eval_rounds` | Number of evaluation episodes |

[Rollout](rollout.md) lists all flags and the input and output
features that a policy should use.

## 9. Step 6: Diagnose and iterate

**Required**

A failed rollout does not by itself mean that the policy architecture or the
training settings are at fault. Imitation-learning policies tend to work only
in situations that the demonstrations cover well. The diagnosis therefore
starts with the data:

1. Run `compare` with the dataset's store, your spec and the rollout log
   (`outputs/rollout/<timestamp>.jsonl`)
   ([Data Verification §5](data_verification.md#5-compare-the-dataset-with-a-rollout)).
   For every axis, the report shows which runs the dataset covers and the
   success rate of covered and uncovered runs: where the objects started, under
   which light, and with which wording of the task text.
2. Watch the failed rollout episodes. A failure in a region with few or no
   dataset episodes points to a gap in the data.
3. Collect more episodes where the report shows gaps, run the check again, and
   retrain.

Coverage is a diagnostic, not a proof: an uncovered region is evidence of a
possible gap in the data, while complete coverage does not guarantee success.

## 10. Documentation map

**Background**

| Guide | Extends | Content |
|---|---|---|
| [Data Collection](data_collection.md) | Steps 1–2 | Tasks, state machine, keyboard teleoperation, adjustable settings, recorder |
| [LeRobot Dataset Visualizer](lerobot_dataset_visualizer.md) | Step 3 | Inspecting episodes in the browser or locally |
| [Data Verification](data_verification.md) | Steps 3 and 6 | The three generalization axes, the workflow, specs, running the check, comparing rollouts, querying the store |
| [Object Poses: Object Position](object_pose.md) | Steps 3 and 6 | The object-position spec, its report, and its rollout comparison |
| [Image Augmentations: Lighting](image_augmentation.md) | Steps 3 and 6 | The lighting spec, its report, and its rollout comparison |
| [Object Properties: Referencing Objects by Color](object_property.md) | Steps 3 and 6 | The color-reference spec, its report, and its rollout comparison |
| [LeRobot Training](lerobot_training.md) | Step 4 | Training flags, multi-GPU, checkpoint layout, upload |
| [Rollout](rollout.md) | Step 5 | Rollout flags, eval files, policy input and output contract |
| [Adding a New Task](adding_new_task.md) | Step 1, self-defined task | Writing, registering, and testing a new task configuration |
