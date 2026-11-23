# LeRobot Dataset Visualizer

This guide describes how recorded episodes are inspected visually before
training. It extends step 3 of [Getting Started](getting_started.md).

Every section starts with one of two labels: **Required** (something to do or follow) or **Background** (explanation without work items).

## 1. Purpose

**Background**

The visualizer shows the recorded camera videos together with per-frame plots
of `observation.state` and `action`, and an episode picker. It is used to
check episode boundaries, the continuity of the motion, and the camera
framing before GPU time is spent on training. It only reads datasets; it does
not record or edit them.

Two versions exist:

| Version | Where | Needs |
|---|---|---|
| Hosted Space, <https://huggingface.co/spaces/lerobot/visualize_dataset> | Browser | The dataset uploaded to the Hugging Face Hub |
| `lerobot-dataset-viz` | General environment, on a desktop session | Only the local dataset folder |

## 2. Inspect a few episodes

**Required**

### 2.1 Web-based viewer

The web-based viewer allows recorded episodes to be inspected directly in a browser using the Hugging Face Dataset Visualizer. Unlike the local viewer in §2.2, this version requires the dataset to be uploaded to the Hugging Face Hub first.

#### Step 1: Upload the dataset

Run the following command in the general environment to upload the local dataset folder to the Hugging Face Hub:

```bash
hf upload ${HF_USER}/<repo_name> datasets/${HF_USER}/<repo_name> --repo-type dataset
```
#### Step 2: Open the dataset in the visualizer

Open the LeRobot Dataset Visualizer and enter the dataset repository ID:

```text
${HF_USER}/<repo_name>
```

Alternatively, open a specific episode directly using the following URL:

```text
https://huggingface.co/spaces/lerobot/visualize_dataset?dataset=${HF_USER}/<repo_name>&episode=<N>
```

Public datasets open without signing in; private datasets require a Hub
account with read access. A link can be shared with others.

### 2.2 Local viewer

The viewer opens a window, and it should be started in the general environment from a terminal of a desktop session (on a Brev VM: the noVNC desktop, [Getting Started §3.4](getting_started.md#34-launch-the-simulator)):

```bash
lerobot-dataset-viz \
    --repo-id ${HF_USER}/<repo_name> \
    --root datasets/${HF_USER}/<repo_name> \
    --episode-index 0 \
    --num-workers 0
```

| Argument | Meaning |
|---|---|
| `--repo-id` | Repo id of the dataset, `${HF_USER}/<repo_name>` |
| `--root` | Local dataset folder; without it, the dataset is looked up in the Hugging Face cache |
| `--episode-index` | Episode to show |
| `--num-workers 0` | Loads the data in the main process; avoids the "shared memory insufficient" error of the default worker processes |

It opens the same plots and videos in a local rerun.io viewer. This version
needs no upload and suits quick checks while the data collection is still
being adjusted.

## 3. What to look for

**Required**

| Panel | Source | What a good dataset shows |
|---|---|---|
| Video player | `observation.images.front`, `observation.images.wrist` | Correct framing, smooth playback. The first frame shows the freshly reset scene; the last frame shows the finished task. |
| State plot | `observation.state` (Franka joint positions) | Continuous trajectories within joint limits; the curves move when the arm moves. |
| Action plot | `action` (joint targets and gripper command) | Close to the state curves; gripper open and close switches where the task needs them. |
| Episode picker | `episode_index` | The expected number of episodes, with plausible lengths. |
| Frame scrubber | `frame_index`, `timestamp` | About `60 × episode seconds` frames (60 fps, one frame per simulation step; each camera image repeats on two consecutive frames). |

The visualizer reads the features from the dataset's `meta/info.json`; a
feature that the recorder did not declare does not appear. The
verification columns (`object_position.*`, `lighting.*`, `object_color.*`) are
not displayed; they are checked with
[Data Verification](data_verification.md).

## 4. Common issues

**Background**

| Symptom | Cause |
|---|---|
| "Dataset not found" in the Space | Typo in the repo id, or a private dataset while not signed in. Opening the dataset's Hub page shows whether it is visible. |
| Video panel blank while the plots load | Videos failed to encode during recording, or were not uploaded. The local folder should contain `videos/observation.images.*/chunk-*/file-*.mp4`. |
| Local viewer fails when decoding a video (`ffmpeg` not found, or a `torchcodec` / `libavutil` load error) | `ffmpeg` is not installed. Run `sudo apt-get install -y ffmpeg` and check with `ffmpeg -version`. |
| A camera is missing | The scene's cameras were changed ([Data Collection §4](data_collection.md#4-adjusting-the-collection)). |
| Episode ends early or late | A changed state-machine phase duration or success range, or **N** pressed too early in teleoperation; compare the last frame with the task's success check ([Data Collection §1](data_collection.md#1-the-two-methods)). |

## 5. References

- Hosted visualizer: <https://huggingface.co/spaces/lerobot/visualize_dataset>
- LeRobot documentation: <https://huggingface.co/docs/lerobot>
