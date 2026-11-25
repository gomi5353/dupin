# Image Augmentations: Lighting

This guide describes the lighting check, one of the three generalization axes
of the data-verification tool
([Data Verification §1](data_verification.md#1-the-three-generalization-axes)).
It explains why lighting is varied, where the task sets it, which dataset
column records it, how to fill in the specification, and how to read the
report for the dataset and for a rollout. The workflow, the store, the
`verify` command and the queries are common to every axis and are described
in [Data Verification](data_verification.md).

Every section starts with one of two labels: **Required** (something to do or
follow) or **Background** (explanation without work items).

## 1. What the check does and why

**Background**

The axis follows STAR-Gen (Gao et al., *A Taxonomy for Evaluating Generalist
Robot Manipulation Policies*, arXiv:2503.01238). STAR-Gen's axis **Image
Augmentations (V-AUG)** is a purely visual change: the scene, the objects and
the task stay the same, only the images look different. STAR-Gen lists
several such factors (for example lighting, blur and contrast). The
repository uses one of them, **lighting**:

| STAR-Gen axis | Factor | Axis class / verifier / spec file | Requirement class | Dataset column |
|---|---|---|---|---|
| Image Augmentations, V-AUG (Visual) | lighting | `cp:ImageAugmentationAxis` / `image_augmentation` / `image_augmentation.ttl` | `cp:LightingRequirement` | `lighting.rect_light` |

Why lighting matters: the policy sees the scene only through the `front` and
`wrist` cameras. A brighter or dimmer light, or a warmer (orange) or cooler
(blue) light, changes every pixel of those images, although the robot has to
do exactly the same thing. A policy trained under one light only may treat a
different light as an unknown scene and fail. The evaluation files vary the
light over a much wider range than the recording defaults
([section 2](#2-where-the-lighting-is-set)), so a dataset recorded with the
defaults covers only a small part of what the policy meets at evaluation.

The check answers one question: **under which light conditions does the
dataset actually have episodes, measured against a requirement written in a
spec?** The spec states a range of light intensity, a range of color
temperature, a grid that divides the two ranges into cells, and a minimum
number of episodes per cell. The check counts the episodes whose light
*started* in each cell and reports the cells below the minimum. It works like
the object-position grid of [Object Poses](object_pose.md), with intensity in
place of x and color temperature in place of y.

## 2. Where the lighting is set

**Required**

Each task scene has one rectangular area light, a USD `RectLight`, wrapped in
the task configuration as the scene entity `rect_light`
(`packages/simulator/src/simulator/tasks/<task>/<task>_env_cfg.py`):

```python
rect_light: AssetBaseCfg = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Scene/RectLight")
```

A reset event draws a new intensity and a new color temperature for it at the
start of every episode:

```python
randomize_light_conditions(
    "rect_light",
    intensity_range=(4500.0, 5500.0),
    color_variation=0.0,
    textures=[],
    color_temperature_range=(5300.0, 5900.0),
),
```

| Setting | Recording default (all three tasks) | Evaluation (`eval/<task>_eval.py`) |
|---|---|---|
| `intensity_range` | 4500–5500 | 2500–8000 |
| `color_temperature_range` | 5300–5900 K | 3000–6500 K |

- **Intensity** is the USD `inputs:intensity` of the light, without a unit.
- **Color temperature** is in kelvin: lower values are warmer (more orange),
  higher values cooler (more blue).
- Both values are drawn uniformly from their range, from a random generator
  seeded with the environment's seed, so the same seed gives the same
  sequence of lights.
- `color_variation=0.0` and `textures=[]` keep the light's tint and texture
  fixed; only intensity and color temperature change.

To cover the evaluation's lighting, widen the two ranges in the task
configuration and record again. The recording configuration is never read by
rollout ([Rollout §1](rollout.md#1-what-rollout-does)), so this changes the
data only.

The lights to record are listed in `recorded_light_names = ["rect_light"]` of
the task configuration. A task that adds a light lists it there as well.

## 3. The lighting column

**Background**

The repository's `LeRobotRecorderManager`
([Data Collection §5](data_collection.md#5-the-recorder-and-the-dataset))
writes one column per light in `recorded_light_names`:

| Key | dtype | Shape | Content |
|---|---|---|---|
| `lighting.<name>` | float32 | `(2,)` | `[intensity, color_temperature]` |

- The values are read from the USD stage (`inputs:intensity`,
  `inputs:colorTemperature`), the same values the renderer uses. When the
  light does not use its color temperature, 6500 K (the white point) is
  written.
- Every frame is recorded. The check uses the first recorded frame of each
  episode; the light does not change within an episode.
- Only environment 0 is recorded (a LeIsaac limitation), as for every column.

A quick test after recording a few episodes (general environment):

```bash
uv run python -c "import json; f=json.load(open('datasets/${HF_USER}/<repo_name>/meta/info.json'))['features']; print([k for k in f if k.startswith('lighting.')])"
```

The command reads `meta/info.json` of the dataset folder
`datasets/${HF_USER}/<repo_name>` and prints the feature keys that start with
`lighting.`. The output is `['lighting.rect_light']`. An empty list means that the
dataset was recorded without `--use_lerobot_recorder`.

## 4. Write the specification

**Required**

Copy `specs/<task>/image_augmentation.ttl` (the same file for every task) into
`my_specs/` and fill in every `TODO`
([Data Verification §3](data_verification.md#3-specs)). The file has one
block per light:

```turtle
spec:image_augmentation_axis a cp:ImageAugmentationAxis .

spec:image_augmentation_axis cp:hasRequirement spec:rect_light_requirement .

spec:rect_light_requirement a cp:LightingRequirement ;
    cp:lightName "rect_light" ;
    cp:intensityMin "TODO" ; cp:intensityMax "TODO" ;
    cp:colorTemperatureMin "TODO" ; cp:colorTemperatureMax "TODO" ;
    cp:numBinsIntensity "TODO" ; cp:numBinsColorTemperature "TODO" ;
    cp:minEpisodesPerCell "TODO" .
```

| Field | Meaning | Type |
|---|---|---|
| `cp:lightName` | Light to check; matches `lighting.<name>` in the dataset | Text in quotes (already filled: `"rect_light"`) |
| `cp:intensityMin`, `cp:intensityMax` | Intensity range the dataset should cover | Number, **no quotes** (`2500.0`) |
| `cp:colorTemperatureMin`, `cp:colorTemperatureMax` | Color temperature range, in kelvin | Number, no quotes (`3000.0`) |
| `cp:numBinsIntensity` | Number of cells along the intensity range (columns, numbered from 0 at `intensityMin`) | Whole number ≥ 1, no quotes (`3`) |
| `cp:numBinsColorTemperature` | Number of cells along the color temperature range (rows, numbered from 0 at `colorTemperatureMin`) | Whole number ≥ 1, no quotes (`2`) |
| `cp:minEpisodesPerCell` | Episodes needed in each cell | Whole number ≥ 1, no quotes (`2`) |

The minimum must be smaller than the maximum in both ranges. One requirement
per light; a second block for the same `cp:lightName` is a violation.

### 4.1 Choosing the values

The values can be chosen freely. The following questions help.

**Ranges.** The evaluation files use intensity 2500–8000 and 3000–6500 K. A
spec with exactly these ranges asks: *does the dataset cover the public
evaluation's lighting?* With the recording defaults (4500–5500,
5300–5900 K), the answer is no, and the report shows which cells are missing.
A narrower spec range only checks a part of the evaluation range; runs
outside it are not covered in the rollout comparison
([section 6](#6-compare-with-a-rollout)). The private evaluation
configuration is not published, so ranges wider than the public ones can be
chosen; the dataset then needs episodes there too.

**Bins.** Cell width = `(intensityMax - intensityMin) / numBinsIntensity`,
cell height = `(colorTemperatureMax - colorTemperatureMin) /
numBinsColorTemperature`. A useful scale is how much the light can change
before the images look clearly different. More cells show gaps in more
detail but need more episodes: the minimum dataset size is
`numBinsIntensity × numBinsColorTemperature × minEpisodesPerCell`.

**Minimum per cell.** `1` only says that one example exists under that
light. Since intensity and color temperature are drawn at random, more
episodes than the minimum dataset size are needed to fill every cell.

**Example** (illustrative values, not a recommendation):

```turtle
    cp:intensityMin 2500.0 ; cp:intensityMax 8000.0 ;
    cp:colorTemperatureMin 3000.0 ; cp:colorTemperatureMax 6500.0 ;
    cp:numBinsIntensity 3 ; cp:numBinsColorTemperature 2 ;
    cp:minEpisodesPerCell 2 .
```

The evaluation ranges split into 3 × 2 = 6 cells, each about 1833 wide in
intensity and 1750 K high in color temperature, each needing 2 episodes.

## 5. Read the report

**Required**

Run `verify` with the filled-in spec
([Data Verification §4](data_verification.md#4-run-the-check-and-read-the-report)):

```bash
uv run python scripts/store.py verify --store stores/<repo_name> \
    --spec my_specs/image_augmentation.ttl
```

| Argument | Meaning |
|---|---|
| `--store` | Store folder of the dataset, `stores/<repo_name>` |
| `--spec` | The filled spec to check |

The command runs in the general environment.

Example output, for the spec of [section 4.1](#41-choosing-the-values) and a
dataset of 20 episodes: 12 recorded with the default ranges, then 8 after the
ranges were widened to the evaluation's:

```
Store   : stores/<repo_name>
Spec    : image_augmentation  (/…/my_specs/image_augmentation.ttl)
Episodes: 20

[rect_light] intensity=2500..8000  temperature=3000..6500 K  covered 3/6 cells (>= 2 each)

VIOLATION 'rect_light' cell (i=0, j=0) intensity=[2500, 4333.33] temperature=[3000, 4750] K has 0 episode(s); needs at least 2  [spec/image_augmentation/rect_light_requirement/cell/0]
VIOLATION 'rect_light' cell (i=2, j=0) intensity=[6166.67, 8000] temperature=[3000, 4750] K has 1 episode(s); needs at least 2  [spec/image_augmentation/rect_light_requirement/cell/2]
VIOLATION 'rect_light' cell (i=0, j=1) intensity=[2500, 4333.33] temperature=[4750, 6500] K has 1 episode(s); needs at least 2  [spec/image_augmentation/rect_light_requirement/cell/3]

RESULT: coverage spec NOT met

Findings written to stores/<repo_name>/findings/
Figures written to stores/<repo_name>/figures/ (1):
  stores/<repo_name>/figures/image_augmentation__rect_light.png
```

How to read it:

- **Figure** `image_augmentation__rect_light.png`: a heatmap of the grid with
  intensity on the x axis and color temperature (K) on the y axis, so the
  **coolest (highest kelvin) bin is at the top**. Each cell shows its episode
  count (the numbers are left out for grids larger than 12 × 12); red cells
  have fewer episodes than the minimum, green ones have enough. The black
  frame is the range from the spec. Pass `--no-figures` to skip the figures. The next `verify` replaces
  them, also when it checks only another axis.
- All 12 default-range episodes fall into one cell (middle intensity, high
  temperature, 14 episodes in total): the default ranges lie inside it. Only the 8 later episodes
  spread over the other cells.
- **VIOLATION**: one line per cell below the minimum, with its exact bounds.
  These are the light conditions to record next. The node name at the end is
  the cell's row-major index `k = j × numBinsIntensity + i`.
- A value on an inner cell edge belongs to the upper cell; the upper end of a
  range belongs to the last cell.

The cells and their counts are stored as `cp:LightingCell` nodes with
`cp:episodeCount`, so they can be queried
([Data Verification §6](data_verification.md#6-query-the-store)), for
example:

```bash
uv run python scripts/store.py query --store stores/<repo_name> --query-text \
  'SELECT ?i ?j ?n WHERE { ?c a cp:LightingCell ; cp:cellI ?i ; cp:cellJ ?j ; cp:episodeCount ?n } ORDER BY ?j ?i'
```

| Argument | Meaning |
|---|---|
| `--store` | Store folder of the dataset, `stores/<repo_name>` |
| `--query-text` | A SPARQL query; this one lists each cell's column `i`, row `j` and episode count `n` |

## 6. Compare with a rollout

**Required**

The rollout comparison works the same for every axis
([Data Verification §5](data_verification.md#5-compare-the-dataset-with-a-rollout)).
For lighting:

- `scripts/rollout.py` logs each run's light right after the reset, under
  `"factors": {"lighting": {"rect_light": [intensity, color_temperature]}}`,
  read in the same way as the dataset column. The light comes from the eval
  file's `randomize_light_conditions`, so with the shipped eval files every
  run falls inside 2500–8000 and 3000–6500 K.
- Each run is placed into a cell of the spec's grid by its light. A run is
  **covered** on this axis when its cell has at least `minEpisodesPerCell`
  dataset episodes. A run outside the spec's ranges is never covered.
- **In distribution** is defined in
  [Data Verification §5](data_verification.md#5-compare-the-dataset-with-a-rollout);
  with this spec alone, it equals covered.

Example output, for the dataset of [section 5](#5-read-the-report) and a
synthetic rollout log of 20 runs (made up for illustration, not from a
trained policy):

```
Store   : stores/<repo_name>
Spec    : image_augmentation
Rollout : 20261120-101500  (eval/cutlery_arrangement_eval.py, seed 42)
Runs    : 20 finished, 7 succeeded

[rect_light] intensity=2500..8000  temperature=3000..6500 K, 3x2 cells, >= 2 episodes each
  covered: 12/20 runs (60%)
  success: 7/12 covered vs 0/8 not covered

In distribution (covered on every axis: image_augmentation): 12/20 runs (60%)
  success 7/12 in distribution vs 0/8 outside

Lighting: where to record more episodes:
  rect_light: record more episodes with intensity 2500..4333.33 and 3000..4750 K (has 0, needs 2; 4 run(s), 4 failed)
  rect_light: record more episodes with intensity 2500..4333.33 and 4750..6500 K (has 1, needs 2; 3 run(s), 3 failed)
  rect_light: record more episodes with intensity 6166.67..8000 and 3000..4750 K (has 1, needs 2; 1 run(s), 1 failed)

WARNING   'rect_light': 8 run(s) started with lighting the dataset does not cover (8 failed)  [rect_light/cell/(0, 0), rect_light/cell/(0, 1), rect_light/cell/(2, 0)]

RESULT: the dataset does NOT cover every rollout run; see above
```

| Part | Meaning | In the example above |
|---|---|---|
| **Figure** `<spec>__<rollout>__<light>.png` | Two heatmaps of the same grid. Left: dataset episodes per cell (red = fewer than the minimum). Right: the rollout runs that started in each cell, coloured by success rate (red = all failed, green = all succeeded, grey = no runs), with `successes/runs` written in the cell for small grids. | The dim, warm corner (`0` episodes, 4 runs, all failed) is red on the left and red on the right: missing data. The cell with 2 episodes and 4 of 4 runs succeeded is green on both. |
| **outside your ranges** | Shown only when runs fell outside the spec's ranges. | None: the spec uses the eval ranges. |
| **covered** | Runs in covered cells / all runs. | 2 + 6 + 4 = 12 of 20. |
| **success** | Success rate of the covered runs against the other runs. A much lower rate outside points to missing data rather than the model. | 7/12 against 0/8. |
| **where to record more episodes** | Cells with runs and too few episodes, the ones with most failed runs first. A run outside the ranges gives the hint to widen the ranges in the spec and in the task's light randomization. | Record dim light (intensity 2500–4333) first. |
| **WARNING** | One line per light with uncovered runs. | 8 runs, all failed. |

The light-specific error:

| Message | Cause | Fix |
|---|---|---|
| `Finished run N has no lighting for 'rect_light', which the training dataset's spec requires` | The rollout log has no `lighting` factor (written by an older `rollout.py`, or an eval file without `recorded_light_names`) | Run the rollout again with the current repository |

## 7. Findings

**Background**

Every line of the report is stored as a `cp:Finding`. Severities:
`Error` (the check could not run), `Violation` (a requirement is not met),
`Warning` (allowed, but worth knowing), `Info` (a suggested next step). The
common messages (store, changed dataset, Turtle syntax, ...) are in
[Data Verification §8](data_verification.md#8-troubleshooting).

| Severity | Message | Cause | Fix |
|---|---|---|---|
| Error | `Dataset has no 'lighting.rect_light' column, which the spec requires (...)` | Dataset recorded without `--use_lerobot_recorder`, or with a task whose `recorded_light_names` does not list the light | Record again ([section 3](#3-the-lighting-column)); check `cp:lightName` |
| Error | `Spec still has TODO values: ...` | Unfilled fields | Fill them in ([section 4](#4-write-the-specification)) |
| Error | `... must be a number without quotes, got "2500"` (or `a whole number`) | Quoted number, or a fraction in a bin count or minimum | Write `2500.0`; whole numbers for bins and minimum |
| Error | `rect_light: intensityMin must be < intensityMax, got ...` (or `colorTemperatureMin`) | Empty range | Swap the values |
| Error | `rect_light: numBinsIntensity must be at least 1, got 0` (or `numBinsColorTemperature`, `minEpisodesPerCell`) | Zero or negative value | Use a positive whole number |
| Error | `Spec declares no cp:LightingRequirement; nothing to check` | The block was deleted, but the axis line kept | Restore the block |
| Error | `Episode N: non-finite lighting for 'rect_light'` (at `ingest`) | NaN or infinity in the column | Inspect that episode; drop it or record again |
| Violation | `'rect_light' cell (i=…, j=…) intensity=[…] temperature=[…] K has N episode(s); needs at least M` | Too few episodes under that light | Widen the ranges in the task configuration and record more episodes there |
| Violation | `Episode N has no lighting for required light 'rect_light'` | The column is missing for some episodes | Record with the repository recorder |
| Violation | `More than one lighting requirement for light 'rect_light'; ...` | Two blocks for one light | Keep one |
| Violation | `Requirement is not linked from a ImageAugmentationAxis (cp:hasRequirement)` | The `spec:image_augmentation_axis cp:hasRequirement ...` line was deleted | Restore it from the shipped spec |
| Warning | `Episode N: light 'rect_light' at intensity I, T K is outside the required ranges; it does not count toward coverage` | The task's randomization range is wider than the spec's | Allowed; widen the spec if those lights matter |

## 8. What the check does not cover

**Background**

- **Only one light, two values.** The light's tint, texture, position and
  other scene lights are not recorded. Blur, contrast and other image
  augmentations of STAR-Gen are not used.
- **Each light separately.** A spec with several lights checks each one on its
  own grid; combinations are not checked.
- **Only the first frame.** The light does not change within an episode.
- **Lighting only.** Combinations with object position ("dim light *while*
  the cup is far left") are not checked; the rollout's *in distribution*
  count only requires every axis to cover the run.

Good coverage raises the chance that a policy generalizes to new lighting; it
does not guarantee it.
