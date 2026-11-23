# LeRobot Training

This guide describes how a policy is trained on a recorded dataset with
LeRobot's `lerobot-train`, and what the resulting checkpoint contains. It
extends step 4 of [Getting Started](getting_started.md) and assumes that setup
(step 0) is complete and that the dataset has passed the checks of step 3.

Every section starts with one of two labels: **Required** (something to do
or follow) or **Background** (explanation without work items).

## 1. Where training runs

**Background**

Training runs in the **general environment** (`make launch-gen`,
after `make install-gen`), not in the simulator environment: `.venv-sim` pins the older
packages that Isaac Sim needs, and the Isaac Lab container adds file-access and
GPU pass-through overhead that slows training down considerably. `nvidia-smi`
must list the GPU.

The official LeRobot training documentation is at
<https://huggingface.co/docs/lerobot/il_robots#train-a-policy>.

## 2. Get the dataset

**Required**

A dataset recorded on the same machine is used directly: the simulator shell
and the container both write datasets to the repository's `datasets/` folder,
so the dataset is at `datasets/${HF_USER}/<repo_name>`.

A dataset on the Hugging Face Hub is downloaded first:

```bash
hf download ${HF_USER}/<repo_name> --local-dir datasets/${HF_USER}/<repo_name> --repo-type dataset
```

`--revision <tag>` selects a tagged version, for example `--revision v1`.
Without `--dataset.root`, `lerobot-train` downloads the dataset by itself
([section 3](#3-training-command-and-flags)).

## 3. Training command and flags

**Required**

```bash
lerobot-train \
  --dataset.repo_id=${HF_USER}/<repo_name> \
  --dataset.root=datasets/${HF_USER}/<repo_name> \
  --policy.type=<type> \
  --policy.device=cuda \
  --output_dir=outputs/train/<job_name> \
  --job_name=<job_name> \
  --wandb.enable=true \
  --policy.repo_id=${HF_USER}/<policy_name>
```

| Flag | Description |
|---|---|
| `--dataset.repo_id=${HF_USER}/<repo_name>` | The dataset to train on: the repo id used when recording. |
| `--dataset.root=datasets/${HF_USER}/<repo_name>` | Local folder of the dataset. Without this flag, the dataset is downloaded from the Hugging Face Hub, which requires uploading it first ([section 2](#2-get-the-dataset), [Getting Started §6](getting_started.md#6-step-3-inspect-and-verify-the-dataset)). |
| `--policy.type=<type>` | The policy to train, one of the policies in [section 4](#4-policies); rollout accepts the types of [Rollout §3](rollout.md#flag-reference). |
| `--policy.*` | Options of the chosen policy, listed per policy in [section 4](#4-policies). |
| `--policy.device=cuda` | Trains on the GPU. `cpu` is only useful for quick debugging. |
| `--output_dir` | Folder for checkpoints, logs, and the configuration, for example `outputs/train/diffusion_cupstacking_v1`. It must not exist yet. |
| `--job_name` | Name of the run in the logs and on the W&B dashboard. |
| `--wandb.enable` | `true` logs loss curves to Weights & Biases (requires `wandb login`); `false` disables it. |
| `--policy.repo_id=${HF_USER}/<policy_name>` | Hub repository to which LeRobot pushes the trained policy when training finishes. `--policy.push_to_hub=false` disables the upload; this flag is then not needed. |
| `--batch_size` | Training batch size (default `8`). A smaller value reduces GPU memory use. |
| `--steps` | Number of training steps (default `100000`). A checkpoint is saved every `--save_freq` steps (default `20000`) and at the end. |

Training takes from a few hours to more than a day, depending on the dataset
size and the GPU. Progress is printed in the terminal, and on the Weights &
Biases dashboard when it is enabled (`wandb login` first).

## 4. Policies

**Background**

`--policy.type` accepts the policies below.
The GPU column states what to expect on an RTX 4090 (24 GB). Where the memory is
not enough, or only a very small batch size fits, train on an L40S (46 GB); the
rollout of the resulting checkpoint still runs on an RTX 4090.

| `--policy.type` | RTX 4090 (24 GB) | Notes |
|---|---|---|
| `diffusion` | Fits. |  |
| `act` | Fits. | |
| `groot` | Runs out of memory with the default settings, even with `--batch_size=1`. Runs with `--policy.tune_diffusion_model=false` (the action head stays frozen). | On an L40S the default settings fit. |
| `smolvla` | Fits. | |
| `xvla` | `--batch_size=4` fits; `--batch_size=8` runs out of memory. | Use an L40S for a larger batch size. |
| `pi0`, `pi05` | `--batch_size=8` fits with the memory options in the command below. | Needs a separate environment and access to a gated model. |
| `vqbet`, `tdmpc`, `sac` | Not supported | See the end of this section. |

Every command below uses the placeholders of [section 3](#3-training-command-and-flags);
add `--wandb.enable=true` as there.

### Training Diffusion Policy

```bash
lerobot-train \
  --dataset.repo_id=${HF_USER}/<repo_name> \
  --dataset.root=datasets/${HF_USER}/<repo_name> \
  --policy.type=diffusion \
  --policy.device=cuda \
  --output_dir=outputs/train/<job_name> \
  --job_name=<job_name> \
  --wandb.enable=true \
  --policy.repo_id=${HF_USER}/<policy_name>
```

Diffusion Policy generates smooth action sequences. Its options include
`--policy.n_obs_steps`, `--policy.horizon` and `--policy.n_action_steps`, the
number of actions that rollout executes after each inference (each action
corresponds to one simulation control step).

### Training ACT

ACT (Action Chunking Transformer) predicts a chunk of actions per inference.

```bash
lerobot-train \
  --dataset.repo_id=${HF_USER}/<repo_name> \
  --dataset.root=datasets/${HF_USER}/<repo_name> \
  --policy.type=act \
  --policy.device=cuda \
  --policy.chunk_size=100 \
  --policy.n_action_steps=100 \
  --output_dir=outputs/train/<job_name> \
  --job_name=<job_name> \
  --wandb.enable=true \
  --policy.repo_id=${HF_USER}/<policy_name>
```

| Flag | Description |
|---|---|
| `--policy.chunk_size` | Number of actions predicted by one inference |
| `--policy.n_action_steps` | Number of those actions executed before the next inference; at most `chunk_size` |

All other flags are as in [section 3](#3-training-command-and-flags).

### Training GR00T

`--policy.type=groot` trains NVIDIA's GR00T N1.5 (3B), fine-tuned from `nvidia/GR00T-N1.5-3B`.
The first run downloads about 6 GB from the Hugging Face Hub. The required packages and a prebuilt
FlashAttention wheel are installed by `make install-gen` ([Getting Started §3.2](getting_started.md#32-install-the-general-environment)).

```bash
lerobot-train \
  --dataset.repo_id=${HF_USER}/<repo_name> \
  --dataset.root=datasets/${HF_USER}/<repo_name> \
  --policy.type=groot \
  --policy.device=cuda \
  --policy.tune_diffusion_model=false \
  --policy.push_to_hub=false \
  --output_dir=outputs/train/<job_name> \
  --job_name=<job_name> \
  --wandb.enable=true
```

By default GR00T also trains its diffusion action head, 1.07 B trainable parameters in total,
which does not fit on a 24 GB GPU (an RTX 4090 runs out of memory even with `--batch_size=1`).
`--policy.tune_diffusion_model=false` freezes the action head and trains only the projector
(518 M parameters), which fits on 24 GB. Training the action head as well needs an L40S (46 GB).

### Training SmolVLA

```bash
lerobot-train \
  --dataset.repo_id=${HF_USER}/<repo_name> \
  --dataset.root=datasets/${HF_USER}/<repo_name> \
  --policy.type=smolvla \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --output_dir=outputs/train/<job_name> \
  --job_name=<job_name> \
  --wandb.enable=true
```

The first run downloads the vision-language backbone from the Hugging Face Hub.

### Training X-VLA

X-VLA is fine-tuned from the pretrained `lerobot/xvla-base`. Its camera names and
action format differ from this repository's contract ([Rollout §4](rollout.md#4-policy-input-and-output-contract)),
so three options adapt it: the inputs are redefined as the `front` and `wrist` cameras,
the unused third camera is left empty, and `--policy.action_mode=auto` keeps the 8-value action
(the default is a 20-value action that rollout cannot execute).

```bash
FEATURES='{"observation.state":{"type":"STATE","shape":[9]},"observation.images.front":{"type":"VISUAL","shape":[3,480,640]},"observation.images.wrist":{"type":"VISUAL","shape":[3,480,640]}}'

lerobot-train \
  --dataset.repo_id=${HF_USER}/<repo_name> \
  --dataset.root=datasets/${HF_USER}/<repo_name> \
  --policy.path=lerobot/xvla-base \
  --policy.device=cuda \
  "--policy.input_features=$FEATURES" \
  --policy.empty_cameras=1 \
  --policy.action_mode=auto \
  --policy.push_to_hub=false \
  --batch_size=4 \
  --output_dir=outputs/train/<job_name> \
  --job_name=<job_name> \
  --wandb.enable=true
```

`--batch_size=4` is the largest value that fits on an RTX 4090; use an L40S for more.
`$FEATURES` must be defined in the same terminal; an empty value fails with
`'str' object has no attribute 'items'`.

### Training pi0 and pi05

These policies build on `google/paligemma-3b-pt-224`, a gated model: open
<https://huggingface.co/google/paligemma-3b-pt-224> with the same account as `hf auth login` and
accept the terms once. They also need a patched `transformers` that conflicts with the
version of this repository's environment, so they run in their own environment, created **outside**
the repository:

```bash
uv venv --python 3.11 ~/venv_pi
uv pip install --python ~/venv_pi/bin/python torch==2.7.0 torchvision==0.22.0 torchcodec==0.5 lerobot==0.4.4 \
  "transformers @ git+https://github.com/huggingface/transformers.git@fix/lerobot_openpi"
source ~/venv_pi/bin/activate
```

```bash
lerobot-train \
  --dataset.repo_id=${HF_USER}/<repo_name> \
  --dataset.root=datasets/${HF_USER}/<repo_name> \
  --policy.type=pi0 \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true \
  --policy.train_expert_only=true \
  --policy.push_to_hub=false \
  --batch_size=8 \
  --output_dir=outputs/train/<job_name> \
  --job_name=<job_name> \
  --wandb.enable=true
```

Use `--policy.type=pi05` for pi0.5. Do not use `--policy.path=lerobot/pi0_base`: its
processor files do not match `lerobot==0.4.4`. The rollout of these checkpoints needs the same
patched `transformers` ([Rollout §3](rollout.md#flag-reference)).

### Not supported

- `vqbet`, `tdmpc`: these LeRobot policies accept only one camera image (`tdmpc` also only square images),
  while this repository's observations have two cameras (`front`, `wrist`) of 480 × 640. With a single
  camera they train, but rollout expects every camera as a policy input and stops with a `KeyError`.
- `sac`: a reinforcement-learning policy that `lerobot-train` cannot train, and its rollout fails with
  image inputs (`TypeError: list indices must be integers`).

## 5. Multi-GPU training

**Background**

On a machine with several GPUs, the command is wrapped with `accelerate`,
where `N` is the number of GPUs:

```bash
accelerate launch --multi_gpu --num_processes=N $(which lerobot-train) <args>
```

## 6. Checkpoint layout

**Background**

`lerobot-train` writes one checkpoint per saved step to
`<output_dir>/checkpoints/<step>/pretrained_model/`; `checkpoints/last` points
to the newest one. This `pretrained_model` folder is what rollout loads
(`--policy_checkpoint_path`). Its content (`lerobot==0.4.4`):

```text
pretrained_model/
├── config.json
├── model.safetensors
├── train_config.json
├── policy_preprocessor.json
├── policy_preprocessor_step_3_normalizer_processor.safetensors
├── policy_postprocessor.json
└── policy_postprocessor_step_0_unnormalizer_processor.safetensors
```

| File | Description |
|---|---|
| `config.json` | Policy configuration: policy `type`, architecture hyperparameters, and `input_features` / `output_features`. It tells LeRobot which class to instantiate. |
| `model.safetensors` | Network weights. |
| `train_config.json` | Full training configuration (dataset, optimizer, scheduler, seed). Needed to resume training, not for inference. |
| `policy_preprocessor.json` | Ordered list of processing steps applied to observations before the policy (rename, batch, device, normalize, ...). |
| `policy_preprocessor_step_3_normalizer_processor.safetensors` | Per-feature observation normalization statistics (mean, std, min, max) for step 3 of the preprocessor. |
| `policy_postprocessor.json` | Processing steps applied to the policy's output actions (unnormalize, device, ...). |
| `policy_postprocessor_step_0_unnormalizer_processor.safetensors` | Per-feature action statistics that map normalized actions back to robot units. |

Processor files follow the pattern
`{pipeline}_step_{N}_{registry_name}.safetensors`, where `N` is the step's
index in the pipeline. Inference loads `config.json`, then
`model.safetensors`, then the preprocessor and postprocessor with their files.

## 7. Upload a checkpoint

**Background**

If the automatic push was disabled, or an extra tagged version is wanted, the
checkpoint is uploaded from the general environment:

```bash
hf upload ${HF_USER}/<policy_name> outputs/train/<job_name>/checkpoints/last/pretrained_model --repo-type model --revision <tag>
```

`<tag>` is a version label such as `v1`.
[Rollout §2](rollout.md#2-get-the-checkpoint) describes how
to download it on another machine.

## 8. Troubleshooting

**Background**

| Problem | Fix |
|---|---|
| `nvidia-smi` not found or shows an error | The NVIDIA driver or CUDA is not installed correctly; follow NVIDIA's installation guide. |
| `lerobot-train: command not found` | Run `make install-gen` at the repository root, then `make launch-gen`. |
| Video decoding error at the first batch (`ffmpeg` not found, or a `torchcodec` / `libavutil` load error) | `ffmpeg` is not installed. Run `sudo apt-get install -y ffmpeg`, check with `ffmpeg -version`, and start again. |
| Out of memory (OOM) | Reduce `--batch_size`, for example `--batch_size=4`. For GR00T, also `--policy.tune_diffusion_model=false` ([section 4](#4-policies)). |
| `FileExistsError` for the output directory | Choose a new `--output_dir`. |
| Training loss does not decrease | Check the dataset: consistent, complete demonstrations ([Getting Started §6](getting_started.md#6-step-3-inspect-and-verify-the-dataset)). |
