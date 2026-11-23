# LeRobot Training

This guide describes how a policy is trained on a recorded dataset with
LeRobot's `lerobot-train`, and what the resulting checkpoint contains. It
extends step 4 of [Getting Started](getting_started.md) and assumes that setup
(step 0) is complete and that the dataset has passed the checks of step 3.

Every section starts with one of two labels: **Required** (something students
must do or follow) or **Background** (explanation without work items).

## 1. Where training runs

**Background**

Training runs on the **host**, in the `uv` environment (`source .venv/bin/activate`
after `uv sync`), not in the simulator environment: `.venv-sim` pins the older
packages that Isaac Sim needs, and the Isaac Lab container adds file-access and
GPU pass-through overhead that slows training down considerably. `nvidia-smi`
must list the GPU.

The official LeRobot training documentation is at
<https://huggingface.co/docs/lerobot/il_robots#train-a-policy>.

## 2. Get the dataset

**Required**

A dataset recorded on the same machine is used directly: the simulator shell
and the container both write datasets to the repository's `datasets/` folder,
so the dataset is at `datasets/${HF_USER}/<repo_id>`.

A dataset on the Hugging Face Hub is downloaded first:

```bash
hf download ${HF_USER}/<repo_id> --local-dir datasets/${HF_USER}/<repo_id> --repo-type dataset
```

`--revision <tag>` selects a tagged version, for example `--revision v1`.
Without `--dataset.root`, `lerobot-train` downloads the dataset by itself
([section 4](#4-flag-reference)).

## 3. Training command

**Required**

```bash
lerobot-train \
  --dataset.repo_id=${HF_USER}/<repo_id> \
  --dataset.root=datasets/${HF_USER}/<repo_id> \
  --policy.type=diffusion \
  --policy.device=cuda \
  --output_dir=outputs/train/<job_name> \
  --job_name=<job_name> \
  --wandb.enable=true \
  --policy.repo_id=${HF_USER}/<policy_name>
```

Training takes from a few hours to more than a day, depending on the dataset
size and the GPU. Progress is printed in the terminal, and on the Weights &
Biases dashboard when it is enabled (`wandb login` first).

## 4. Flag reference

**Background**

| Flag | Description |
|---|---|
| `--dataset.repo_id=${HF_USER}/<repo_id>` | The dataset to train on: the repo id used when recording. |
| `--dataset.root=datasets/${HF_USER}/<repo_id>` | Local folder of the dataset. Without this flag, the dataset is downloaded from the Hugging Face Hub, which requires uploading it first ([section 2](#2-get-the-dataset), [Getting Started §6](getting_started.md#6-step-3-inspect-and-verify-the-dataset)). |
| `--policy.type` | Policy architecture. `diffusion` (Diffusion Policy) generates smooth action sequences and is a good default for manipulation; `act` (Action Chunking Transformer) is the other recommended choice. Other LeRobot policies are possible as long as rollout supports them ([Rollout §3](rollout.md#flag-reference)). |
| `--policy.*` | Policy-specific options follow `--policy.type`, for example `--policy.n_obs_steps`, `--policy.horizon`, and `--policy.n_action_steps` for Diffusion Policy, or `--policy.chunk_size` and `--policy.n_action_steps` for ACT. `n_action_steps` is the number of actions that rollout executes after each inference (each lasting one 1/60 s step). |
| `--policy.device=cuda` | Trains on the GPU. `cpu` is only useful for quick debugging. |
| `--output_dir` | Folder for checkpoints, logs, and the configuration, for example `outputs/train/diffusion_cupstacking_v1`. It must not exist yet. |
| `--job_name` | Name of the run in the logs and on the W&B dashboard. |
| `--wandb.enable` | `true` logs loss curves to Weights & Biases (requires `wandb login`); `false` disables it. |
| `--policy.repo_id=${HF_USER}/<policy_name>` | Hub repository to which LeRobot pushes the trained policy when training finishes. `--policy.push_to_hub=false` disables the upload; this flag is then not needed. |
| `--batch_size` | Training batch size (default `8`). A smaller value reduces GPU memory use. |
| `--steps` | Number of training steps (default `100000`). A checkpoint is saved every `--save_freq` steps (default `20000`) and at the end. |

## 5. Other policies

**Background**

Diffusion Policy and ACT are the recommended policies. The policies below were
also trained and rolled out in this repository (RTX 4090, 24 GB, the locked
`torch==2.7.0` environment). They are optional; the results of the shortest
runs only show that the pipeline works, not that the policy performs well.

| `--policy.type` | Status | Notes |
|---|---|---|
| `diffusion`, `act` | Supported | Recommended. |
| `groot` | Supported (optional) | `--policy.tune_diffusion_model=false` on a 24 GB GPU. |
| `smolvla` | Supported (optional) | No extra steps. |
| `xvla` | Supported (optional), extra options | `--batch_size=4` on a 24 GB GPU; 8 runs out of memory. |
| `pi0`, `pi05` | Supported (optional), separate environment | Needs a patched `transformers` and access to `google/paligemma-3b-pt-224`. |
| `vqbet`, `tdmpc`, `sac` | **Not supported** | See the end of this section. |

Every command below uses the placeholders of [section 3](#3-training-command);
add `--wandb.enable=true` as there. If a 24 GB GPU is too small, an L40S (46 GB) can
train the policy and an RTX 4090 can still run the rollout.

### Training GR00T

`--policy.type=groot` trains NVIDIA's GR00T N1.5 (3B), fine-tuned from `nvidia/GR00T-N1.5-3B`.
The first run downloads about 6 GB from the Hugging Face Hub. The required packages and a prebuilt
FlashAttention wheel are installed by `uv sync` ([Getting Started §3.3](getting_started.md#33-install-the-host-environment)).

```bash
lerobot-train \
  --dataset.repo_id=${HF_USER}/<repo_id> \
  --dataset.root=datasets/${HF_USER}/<repo_id> \
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
(518 M parameters), which fits on 24 GB. On an L40S (46 GB) the default also fits.

### Training SmolVLA

```bash
lerobot-train \
  --dataset.repo_id=${HF_USER}/<repo_id> \
  --dataset.root=datasets/${HF_USER}/<repo_id> \
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
  --dataset.repo_id=${HF_USER}/<repo_id> \
  --dataset.root=datasets/${HF_USER}/<repo_id> \
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
  --dataset.repo_id=${HF_USER}/<repo_id> \
  --dataset.root=datasets/${HF_USER}/<repo_id> \
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
patched `transformers` ([Rollout §3](rollout.md#flag-reference)). If this environment fails to
install, skip these two policies.

### Not supported

- `vqbet`, `tdmpc`: these LeRobot policies accept only one camera image (`tdmpc` also only square images),
  while this repository's observations have two cameras (`front`, `wrist`) of 480 × 640. With a single
  camera they train, but rollout expects every camera as a policy input and stops with a `KeyError`.
- `sac`: a reinforcement-learning policy that `lerobot-train` cannot train, and its rollout fails with
  image inputs (`TypeError: list indices must be integers`).

## 6. Multi-GPU training

**Background**

On a machine with several GPUs, the command is wrapped with `accelerate`,
where `N` is the number of GPUs:

```bash
accelerate launch --multi_gpu --num_processes=N $(which lerobot-train) <args>
```

## 7. Checkpoint layout

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

## 8. Upload a checkpoint

**Background**

If the automatic push was disabled, or an extra tagged version is wanted, the
checkpoint is uploaded from the host:

```bash
hf upload ${HF_USER}/<policy_name> outputs/train/<job_name>/checkpoints/last/pretrained_model --repo-type model --revision <tag>
```

`<tag>` is a version label such as `v1`.
[Rollout §2](rollout.md#2-get-the-checkpoint) describes how
to download it on another machine.

## 9. Troubleshooting

**Background**

| Problem | Fix |
|---|---|
| `nvidia-smi` not found or shows an error | The NVIDIA driver or CUDA is not installed correctly; follow NVIDIA's installation guide. |
| `lerobot-train: command not found` | Run `uv sync` at the repository root and activate `.venv`. |
| Out of memory (OOM) | Reduce `--batch_size`, for example `--batch_size=4`. For GR00T, also `--policy.tune_diffusion_model=false` ([section 5](#5-other-policies)). |
| `FileExistsError` for the output directory | Choose a new `--output_dir`. |
| Training loss does not decrease | Check the dataset: consistent, complete demonstrations ([Getting Started §6](getting_started.md#6-step-3-inspect-and-verify-the-dataset)). |
