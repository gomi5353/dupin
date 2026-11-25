# Object Poses: Object Position

This guide describes the Object Poses axis of the data verification: where on
the table the objects start in the dataset, measured against a requirement
written in a spec. It is one of three axes
([Data Verification §1](data_verification.md#1-the-three-generalization-axes)).
The workflow, the store, the commands and the general parts of the report are
described in [Data Verification](data_verification.md); this guide covers only
what is specific to this axis.

Every section starts with one of two labels: **Required** (something to do or
follow) or **Background** (explanation without work items).

## 1. The axis and why position

**Background**

The axes follow STAR-Gen (Gao et al., "A Taxonomy for Evaluating Generalist
Robot Manipulation Policies", arXiv:2503.01238). This axis is STAR-Gen's
**Object Poses** axis, VB-POSE (it changes both what the policy sees and how
it has to move). STAR-Gen's factor is the *manipulated object pose*. This
repository checks the initial position only, so the factor is called **object
position** ("manipulated" is omitted for brevity).

| | Name in the repository |
|---|---|
| Axis class | `cp:ObjectPoseAxis` |
| Requirement class (factor) | `cp:ObjectPositionRequirement` |
| Verifier name in reports | `object_pose` |
| Shipped spec | `specs/<task>/object_pose.ttl` |
| Dataset columns | `object_position.<name>` |
| Rollout log factor | `object_position` |

Imitation-learning policies tend to perform well only in situations that the
demonstrations represent well. If evaluation places objects where the dataset
has few or no episodes, the policy may fail because of missing data rather than
because of the model.

Object position is the main thing that changes between episodes: every reset
places the objects at a random position, while the robot, cameras, and task
stay the same. Object positions can also be read directly from the simulator,
without labeling or estimation.

**Why position and not the full pose.** The check considers an object's
position only, rather than its full pose (position and orientation), for the
following reasons:

- **Position is the primary variable of interest.** The tasks randomize only
  the position. Each object starts with a fixed rotation, so the orientation is
  the same in every episode and says nothing about coverage.
- **Keeping orientation fixed improves episode reliability.** Randomizing
  initial orientations could cause objects such as cups or cutlery to start in
  unstable configurations, potentially leading them to tip over or fall off the
  table. Such episodes may become unusable, requiring environment resets and
  additional simulation attempts, thereby wasting computational resources on
  episodes that are never recorded.
- **Position allows straightforward spatial discretization.** Coverage is
  measured on the table plane using the (x, y) coordinates. Positions in metres
  can be directly mapped onto a spatial grid, whereas orientations represented
  by quaternions require more complex discretization and have representational
  ambiguities (e.g., two quaternions can represent the same rotation).

The check answers one question: **where on the table does the dataset actually
have episodes?** The spec states, per object, a rectangle of the table, a
grid that divides it into cells, and a minimum number of episodes per cell.
The check counts the episodes whose object *started* in each cell and reports
the cells below the minimum.

## 2. The object-position columns

**Background**

The check needs the `object_position.*` columns. Only the repository's
`LeRobotRecorderManager` writes them
([Data Collection §5](data_collection.md#5-the-recorder-and-the-dataset)).
Each column has this format:

| Key | dtype | Shape | Content |
|---|---|---|---|
| `object_position.<name>` | float32 | `(3,)` | `[x, y, z]` |

- **Position**: metres, relative to the environment origin. This is the frame
  of `init_state.pos` in the task configurations, so those numbers and the
  numbers in a spec are directly comparable.
- **No orientation**: the start rotation is fixed in every task
  ([section 1](#1-the-axis-and-why-position)).
- **Every frame** is recorded. The check uses the first recorded frame of each
  episode, which is where the objects started.
- **Name**: the object's scene name (`blue_cup`, `fork`, ...).

The recorded objects are those in `tracked_object_names` of the task
configuration: the objects whose start position is randomized. Objects that
start at the same place in every episode are listed in `fixed_object_names`
instead. They get no column and are not verified, because their position never
varies; rollout logs their pose once, in the log's header, for reference only
([section 5](#5-compare-with-a-rollout)).

| Task | `tracked_object_names` (recorded, verified) | `fixed_object_names` (not recorded) |
|---|---|---|
| Cup stacking | `blue_cup`, `pink_cup` | none |
| Cutlery arrangement | `fork`, `knife` | `plate` |
| Toy blocks collection | `green_block`, `blue_block`, `red_block` | `storage_box` |

Details of how the recorder captures the positions:

- It skips the first 5 steps of every episode, so the first recorded frame is
  about step 6. Objects spawned slightly above the table may have settled by
  then; in tests they moved by a few millimetres, which does not matter for
  coverage.
- The position in frame *t* is read after that frame's `env.step()`, while the
  observation is read before it, so the position is one simulation step (1/60 s)
  later. Objects at rest do not move in that time.
- Only successful episodes are saved, so the check counts the starting
  positions of successful demonstrations.

A quick test after recording a few episodes (general environment):

```bash
uv run python -c "import json; f=json.load(open('datasets/${HF_USER}/<repo_name>/meta/info.json'))['features']; print([k for k in f if k.startswith('object_position.')])"
```

The command reads `meta/info.json` of the dataset folder
`datasets/${HF_USER}/<repo_name>` and prints the feature keys that start with
`object_position.`. The output lists one key per tracked object. An empty list
means that the dataset was recorded without `--use_lerobot_recorder` and has
to be recorded again.

## 3. The spec: `object_pose.ttl`

**Required**

Copy `specs/<task>/object_pose.ttl` (for example into `my_specs/`) and
replace every `"TODO"`
([Data Verification §3](data_verification.md#3-specs)). For each object, the
spec says: *"the initial positions in the dataset should cover this rectangle,
split into this grid, with at least this many episodes per cell."*

### 3.1 One object block

The spec declares one Object Poses axis at the top
(`spec:object_pose_axis a cp:ObjectPoseAxis .`). The shipped file has one
block per tracked object of the task, between `# --- BEGIN OBJECT ---` and
`# --- END OBJECT ---`. Each block adds one object position requirement to the
axis and describes the object, the area, and the grid:

```turtle
spec:object_pose_axis cp:hasRequirement spec:blue_cup_requirement .

spec:blue_cup_requirement a cp:ObjectPositionRequirement ;
    cp:hasTaskObject spec:blue_cup ;
    cp:hasAreaConfiguration spec:blue_cup_area ;
    cp:minEpisodesPerCell "TODO" .

spec:blue_cup a cp:TaskObject ;
    cp:name "blue_cup" .

spec:blue_cup_area a cp:AreaConfiguration ;
    cp:hasCoverageGrid spec:blue_cup_grid ;
    cp:areaXMin "TODO" ; cp:areaXMax "TODO" ;
    cp:areaYMin "TODO" ; cp:areaYMax "TODO" .

spec:blue_cup_grid a cp:CoverageGrid ;
    cp:numCellW "TODO" ; cp:numCellH "TODO" .
```

| Field | Meaning | Type |
|---|---|---|
| `cp:name` | Object to check; matches `object_position.<name>` in the dataset | Text in quotes (already filled) |
| `cp:areaXMin`, `cp:areaXMax`, `cp:areaYMin`, `cp:areaYMax` | Area (a rectangle), in metres and in the environment frame, where initial positions should be covered | Number, **no quotes** (`8.80`) |
| `cp:numCellW`, `cp:numCellH` | Number of cells along x (width) and along y (height) | Whole number ≥ 1, no quotes (`3`) |
| `cp:minEpisodesPerCell` | Episodes needed in each cell | Whole number ≥ 1, no quotes (`2`) |

- Deleting a block stops checking that object; copying a block (with new
  names for its nodes) checks another one. Objects without a block are
  recorded but never checked.
- The links (`cp:hasRequirement`, `cp:hasTaskObject`, `cp:hasAreaConfiguration`,
  `cp:hasCoverageGrid`) stay as shipped.
- `verify` rejects the spec with an `ERROR` line when a `TODO` is left, a
  number is quoted or not whole where it must be, an area is empty
  (`areaXMin >= areaXMax` or `areaYMin >= areaYMax`), or a grid size or minimum
  is below 1 ([section 6](#6-troubleshooting)).

### 3.2 Choosing the values

The values can be chosen freely. The following questions help.

**Area (`areaXMin` .. `areaYMax`)**

- Where could the object plausibly start at test time? The public eval file
  shows each object's nominal position (`init_state.pos`) and its randomization
  range (`pose_range`); the private test uses different poses. The table, the
  robot's reach, and the camera views also limit the area.
- Can the robot reach and grasp the object everywhere in the area?
  Covering unreachable corners wastes collection time.
- The area is set per object; objects of the same task can have different
  areas.
- The area describes where coverage is *wanted*. Copying the range the dataset
  has now would hide exactly the gaps the check is meant to find.

**Grid (`numCellW`, `numCellH`)**

- Cell width = `(areaXMax - areaXMin) / numCellW`, cell height =
  `(areaYMax - areaYMin) / numCellH`. A useful scale is how far the object can
  move before the grasp has to change: a few centimetres for small objects,
  more for large or forgiving ones.
- More cells show gaps in more detail but need more episodes: the minimum
  dataset size is `numCellW × numCellH × minEpisodesPerCell` per object.

**Minimum per cell (`minEpisodesPerCell`)**

- `1` only says that at least one example exists there. Higher values require
  repeated examples in every region.
- Budget: 3 × 3 cells × 3 episodes = 27 episodes per object if perfectly
  spread. Random object placement needs more than that to fill every cell.

**Example** (illustrative values, not a recommendation). The cup-stacking eval
file places `blue_cup` at x 8.90, y 6.48 and randomizes it by ±0.1 m, so x
8.8–9.0, y 6.38–6.58:

```turtle
    cp:minEpisodesPerCell 3 .                    # in the requirement
    cp:areaXMin 8.8 ; cp:areaXMax 9.0 ;          # in the area
    cp:areaYMin 6.4 ; cp:areaYMax 6.6 .
    cp:numCellW 2 ; cp:numCellH 2 .              # in the grid
```

This is a 20 cm × 20 cm square split into four 10 cm cells, each needing 3
episodes. Its `areaYMin` of 6.4 misses the bottom 2 cm of the eval range on
purpose; [section 5](#5-compare-with-a-rollout) shows how the rollout
comparison finds that.

## 4. Read the report

**Required**

Run `verify` with the filled spec
([Data Verification §4](data_verification.md#4-run-the-check-and-read-the-report)):

```bash
uv run python scripts/store.py verify --store stores/<repo_name> --spec my_specs/object_pose.ttl
```

| Argument | Meaning |
|---|---|
| `--store` | Store folder of the dataset, `stores/<repo_name>` |
| `--spec` | The filled spec to check (general environment) |

The Object Poses part of the report, for the spec of
[section 3.2](#32-choosing-the-values) (`pink_cup` with x 9.0–9.2,
y 6.38–6.58, also 2 × 2 cells and 3 episodes each) and a dataset of 12
episodes:

```
Store   : stores/<repo_name>
Spec    : object_pose  (/home/<user>/<repo>/my_specs/object_pose.ttl)
Episodes: 12

[blue_cup] x=8.800..9.000  y=6.400..6.600  covered 3/4 cells (>= 3 each)

[pink_cup] x=9.000..9.200  y=6.380..6.580  covered 4/4 cells (>= 3 each)

VIOLATION 'blue_cup' cell (i=1, j=1) x=[8.9, 9] y=[6.5, 6.6] has 1 episode(s); needs at least 3  [spec/object_pose/blue_cup_grid/cell/3]

RESULT: coverage spec NOT met

Findings written to stores/<repo_name>/findings/
Figures written to stores/<repo_name>/figures/ (2):
  stores/<repo_name>/figures/object_pose__blue_cup.png
  stores/<repo_name>/figures/object_pose__pink_cup.png
```

How to read it:

- **Header** `[blue_cup] x=… y=… covered N/M cells`: the object's area from
  the spec, and how many of its cells have at least `minEpisodesPerCell`
  episodes.
- **Figure** `object_pose__<object>.png`: a heatmap of the object's grid with x
  and y in metres (**+y at the top**). Each cell shows its episode count (the
  numbers are left out for grids larger than 12 × 12); red cells have fewer
  episodes than the minimum, green ones have enough. The black frame is the
  area from the spec. Pass `--no-figures` to skip the figures. The next `verify` replaces
  them, also when it checks only another axis.
- **VIOLATION**: one line per thin or empty cell, with its exact bounds; these
  are the places to collect next. The node name at the end
  (`spec/object_pose/blue_cup_grid/cell/3`) is the cell's row-major index
  `k = j × numCellW + i`.
- **WARNING** `Episode N: initial position of '…' at (x, y) is outside the
  required area; it does not count toward coverage`: allowed but worth
  knowing; the episode counts toward no cell.

Each cell's episode count is stored as `cp:episodeCount` with the findings, so
the counts can be queried later (`queries/cell_counts.rq`,
`queries/uncovered_cells.rq`;
[Data Verification §6](data_verification.md#6-query-the-store)).

## 5. Compare with a rollout

**Required**

The general steps (put the rollout log into the store, run `verify`) are in
[Data Verification §5](data_verification.md#5-compare-the-dataset-with-a-rollout).
For this axis, the comparison asks: **did the dataset cover the positions the
rollout actually used?** It reads where each rollout episode (run) started,
from the `object_position` factor of the rollout log, and whether it
succeeded. The comparison does not read the eval file: the runs' own positions
show where the rollout placed the objects. The robot base and the fixed
objects' poses in the log's header are stored (`cp:hasFixedObject`) but not
compared.

How the comparison works, per object:

- **Cell of a run.** Each run is placed into a cell of **the spec's grid**
  for that object, by its initial position, with the same rule as the dataset
  check. A run that started outside the area is placed into a cell of the
  same size outside it, continuing the grid as a lattice (for example column
  `i = -1` left of the area).
- **Covered.** A run is covered when it started inside the area, in a cell
  with at least `minEpisodesPerCell` dataset episodes. Only cells with at
  least one run are judged.
- **Covered on this axis.** A run is covered on the Object Poses axis when it
  is covered for every object requirement of the spec at once.

**Example.** The cup-stacking eval file randomizes `blue_cup` over x 8.8–9.0,
y 6.38–6.58 and `pink_cup` over x 9.0–9.2, y 6.38–6.58. With the spec and
dataset of [section 4](#4-read-the-report), a rollout with 20 episodes gives:

```
Store   : stores/<repo_name>
Spec    : object_pose
Rollout : 20261103-142530  (eval/cup_stacking_eval.py, seed 42)
Runs    : 20 finished, 8 succeeded

[blue_cup] your area x=8.800..9.000 y=6.400..6.600, cell 0.1 x 0.1, 2x2 cells, >= 3 episodes each
  10 run(s) started outside your area, in 2 cell(s) of the same size
  covered: 1/4 sampled cells, 5/20 runs (25%)
  success: 4/5 covered vs 4/15 not covered

[pink_cup] your area x=9.000..9.200 y=6.380..6.580, cell 0.1 x 0.1, 2x2 cells, >= 3 episodes each
  covered: 4/4 sampled cells, 20/20 runs (100%)
  success: 8/20 covered vs 0/0 not covered

In distribution (covered on every axis: object_pose): 5/20 runs (25%)
  success 4/5 in distribution vs 4/15 outside

Record more episodes here (2 more at least; cells with failed rollout runs first):
  blue_cup cell (i=1, j=1) x=[8.900, 9.000] y=[6.500, 6.600] centre (8.950, 6.550): has 1, needs 2 more; 5 rollout run(s) started here, 4 failed

Rollout runs started outside your area. Grow it by whole cells (same cell size), then record there:
  blue_cup: area x=[8.800, 9.000] y=[6.400, 6.600] -> areaXMin 8.8 ; areaXMax 9 ; areaYMin 6.3 ; areaYMax 6.6 ; numCellW 2 ; numCellH 3

WARNING   'blue_cup': 3 of 4 sampled cell(s) are not covered by the dataset (15 run(s), 11 failed)  [blue_cup/cell/(1, 1), blue_cup/cell/(0, -1), blue_cup/cell/(1, -1)]

RESULT: the dataset does NOT cover every rollout run; see above
```

How to read it:

| Part | Meaning | In the example above |
|---|---|---|
| **Header** `[blue_cup] your area … cell … NxM cells` | The area and grid for the object, from the spec. | Cell size (9.0 − 8.8) / 2 = 0.1 by (6.6 − 6.4) / 2 = 0.1. |
| **Figure** `<spec>__<rollout>__<object>.png` | Two heatmaps with +y at the top. Left: dataset episodes per cell (red = fewer than `minEpisodesPerCell`, so runs there are not covered). Right: the rollout runs that started in each cell, coloured by success rate (red = all failed, green = all succeeded, grey = no runs). Cells outside the area that runs started in are drawn too. | `blue_cup`, cell (x1, y1): red on the left (1 episode, minimum 3), red on the right (1 of 5 runs succeeded): the place to record next. Cell (x0, y1): green on the left, grey on the right: enough episodes, no runs, so not judged. |
| **outside your area** | Runs that started outside the area, and how many lattice cells of the grid's size they fall into. They are not covered. | 10 runs started at y 6.38–6.4, below `areaYMin` 6.4: in cells (0, −1) and (1, −1). |
| **covered** | Covered cells / cells with runs (outside cells included), and the runs in covered cells / all runs. | `blue_cup`: only (x0, y0) of the 4 sampled cells is covered, so 1/4 cells and 5/20 runs. `pink_cup`: 4/4 cells, 20/20 runs. |
| **success** | Success rate of the covered runs compared with the other runs. | `blue_cup`: 4/5 vs 1 + 1 + 2 = 4 of 15. `pink_cup`: all 20 runs covered, so 8/20 vs 0/0. |
| **In distribution** | Runs covered on every axis of the spec ([Data Verification §5](data_verification.md#5-compare-the-dataset-with-a-rollout)). With only this axis in the spec, a run counts when every object started in a covered cell. | All `pink_cup` runs are covered, so this equals `blue_cup`'s 5/20; success 4/5 (80 %) inside vs 4/15 (27 %) outside. |
| **Record more episodes here** — *where is data missing?* | Cells of the spec's grid with runs and fewer episodes than the minimum. Sorted by failed runs, then by missing episodes. `N more at least` is the sum of `needs … more`. Recording there needs no change to the spec. | Cell (i=1, j=1), x 8.9–9.0, y 6.5–6.6, has 1 episode; it needs 2 more. 5 runs started there, 4 failed. |
| **Grow your area** — *is the area too small?* | Runs started outside the spec's area. The suggested area is enlarged by whole cells, so the cell size and the cells already recorded stay the same. After updating the spec, run `verify` again to find the new cells that need episodes. | The runs at y 6.38–6.4 fall into row j = −1 (y 6.3–6.4): `areaYMin` 6.4 − 0.1 = 6.3 and `numCellH` 2 + 1 = 3. |
| **WARNING** | One line per object with uncovered cells, named `(i, j)` as in the grid. | `blue_cup`: (1, 1), (0, −1) and (1, −1); 5 + 4 + 6 = 15 runs, 11 failed. |

The two hint blocks (**Record more episodes here**, **Grow your area**) are
suggestions computed from this axis, one way to read the comparison.
`--no-hints` leaves them out of the printed report; they are still stored as
`Info` findings. Each run's placement (`cp:RunPlacement` with `cp:cellI`,
`cp:cellJ`, `cp:covered`) is stored too, so other questions can be answered from it
(`queries/run_outcomes_by_cell.rq`, `queries/failed_runs_not_covered.rq`).

The comparison uses the public eval file only. The private test uses a
separate configuration with different positions
([Rollout §1](rollout.md#1-what-rollout-does)), so covering the public range
may not be enough.

## 6. Troubleshooting

**Background**

Messages of this axis. General messages (store, dataset, rollout log, `TODO`,
quoted numbers) are in
[Data Verification §8](data_verification.md#8-troubleshooting).

| Message | Cause | Fix |
|---|---|---|
| `meta/info.json declares no 'object_position.<name>' features; …` (at `ingest`) | Dataset recorded without `--use_lerobot_recorder` | Record again with the option ([Data Collection §5](data_collection.md#5-the-recorder-and-the-dataset)) |
| `Dataset has no 'object_position.<name>' column, which the spec requires` | Block for an object that was not recorded, or a typo in `cp:name` | Fix the name or delete the block |
| `<object>: coverage area is empty: areaXMin must be < areaXMax and areaYMin < areaYMax, got …` | Min and max swapped or equal | Swap the values |
| `<object>: numCellW must be at least 1, got 0` (or `numCellH`, `minEpisodesPerCell`) | Zero or negative grid size or minimum | Use a positive whole number |
| `Spec declares no cp:ObjectPositionRequirement; nothing to check` | Every object block was deleted | Keep at least one block, or leave `object_pose.ttl` out of `--spec` |
| `Requirement is not linked from an ObjectPoseAxis (cp:hasRequirement)` | The `spec:object_pose_axis cp:hasRequirement ...` line of a block was deleted or names another requirement | Restore the line from the shipped spec |
| `Requirement is linked from more than one axis (cp:hasRequirement); …` | Two axes list the same requirement | Keep one `cp:hasRequirement` line per requirement |
| `Area must belong to exactly one requirement …` (or `Grid must belong to exactly one area …`) | Two blocks share one area or grid, or an area is not used by any block | Give each block its own area and grid, as in the shipped spec |
| `Episode N has no initial position for required object '…'` | The episode has no value for that object | Inspect that episode; record with the repository recorder |
| `Episode N: non-finite initial position for '…'` (at `ingest`) | NaN or infinity written as a position (an object fell out of the scene) | Inspect that episode; drop it or fix the reset |
| `Finished run N has no initial position for '…', which the training dataset's spec requires` | The spec names an object the rollout log does not record | Check the object name in the spec |
| `Spec has more than one requirement for '…'; compare needs one per object` | Two blocks for the same object | Keep one block per object |
| `'…' starts at different positions across the rollout's runs, but the spec has no requirement for it; it is not compared` (warning) | No block for that object in the spec | Add a block ([section 3.1](#31-one-object-block)) |
| Every episode lands in one cell | Objects never moved between episodes | Check that the task's reset randomization is active |

## 7. What the check does not cover

**Background**

- **Only x and y position.** Orientation is not recorded; height (z) is
  recorded and stored in the store (`cp:positionZ`) but not checked.
- **Each object separately.** Combinations ("blue cup left *while* pink cup
  right") and distances between objects are not checked.
- **Only the first frame.** Trajectory quality, grasp success, and timing are
  not checked.
- **Rectangles only.** Other shapes need several requirements for the same
  object, or extra SHACL rules passed with `verify --extra-shapes`
  ([Data Verification §7](data_verification.md#7-how-it-works-rdf-owl-shacl)).
- **Fixed objects** (`fixed_object_names`) are not checked.

Good coverage raises the chance that a policy generalizes to new positions; it
does not guarantee it.

## 8. How it works

**Background**

This section is not needed to use the check. The general parts (store,
`Verification`, `Comparison`, `Finding`) are in
[Data Verification §7](data_verification.md#7-how-it-works-rdf-owl-shacl).
The axis lives in
`packages/data_verification/src/data_verification/axes/object_pose/`:

| File | Content |
|---|---|
| `ontology.ttl` | `cp:ObjectPoseAxis` (`cp:starGenAxis "VB-POSE"`), `cp:ObjectPositionRequirement` (`cp:starGenFactor "manipulated object pose (position only)"`), area, grid, cells, run placements |
| `shapes.ttl` | SHACL rules on the dataset: cell coverage, outside-area warning, spec well-formed |
| `rollout_shapes.ttl` | SHACL rule on a rollout: every finished run has the required positions |
| `spec.py` | Checks that stop `verify` for a spec (`TODO` left, values not numbers, empty area) |
| `coverage.py` | Grid cells (added when the spec enters the store), per-cell episode counts and tables |
| `compare.py` | Run placements, coverage of rollout runs, hints |

The axis's part of the vocabulary (dotted links ┄ are added by `verify`; the
rest is written in the spec or recorded by `ingest`):

```text
Dataset
├┄ hasGeneralizationAxis
│   ▶ ObjectPoseAxis                         (a GeneralizationAxis)
│      └─ hasRequirement
│          ▶ ObjectPositionRequirement       (an AxisRequirement; minEpisodesPerCell)
│             ├─ hasTaskObject ▶ TaskObject (name)
│             └─ hasAreaConfiguration
│                 ▶ AreaConfiguration (areaXMin … areaYMax)
│                    └─ hasCoverageGrid
│                        ▶ CoverageGrid (numCellW, numCellH)
│                           └┄ hasCell ▶ Cell (index k, cellI, cellJ, cellXMin … cellYMax; episodeCount)
└─ hasEpisode
    ▶ Episode (index = episode_index, frameCount)
       └─ hasTaskObject ▶ TaskObject (name)
              └─ hasPosition ▶ Position (positionX, positionY, positionZ)

Comparison
└─ hasPlacement
    ▶ RunPlacement (cellI, cellJ, covered)        one per finished run and requirement
       ├─ ofRun ▶ Run
       ├─ ofRequirement ▶ ObjectPositionRequirement
       └─ inCell ▶ Cell                           only inside the area
```

- A requirement's object and the episodes' objects are all `TaskObject`s; the
  check matches them by `name`. Only the episodes' and runs' objects have a
  `Position`.
- A cell is identified by its index `k`, row-major: column `i = k mod numCellW`,
  row `j = ⌊k / numCellW⌋`. When the spec enters the store, `verify` stores
  each cell's column, row and bounds for queries; the SHACL rule computes the
  bounds again from the area and the grid. A position on an inner cell edge
  belongs to the upper cell; the area's upper edges belong to the last column
  and row.
- Every `Position` has `positionX`, `positionY` and `positionZ`, stored as
  decimals rounded to 6 places (micrometres). The coverage rule uses x and y
  only; z is there for inspection and for custom rules.

Additional rules (for example "the object must start in front of the robot";
in cup stacking the robot base is at y 6.0 and the cups start at y 6.38–6.58)
are written as SHACL shapes in a separate `.ttl` file and passed with
`verify --extra-shapes`. The packaged shapes in `axes/object_pose/shapes.ttl`
are commented and are a good starting point. A minimal example:

```turtle
@prefix cp:  <http://taica.course/cp/ontology#> .
@prefix sh:  <http://www.w3.org/ns/shacl#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

cp:InFrontOfRobotShape a sh:NodeShape ;
    sh:targetClass cp:Position ;
    sh:property [ sh:path cp:positionY ; sh:minInclusive "6.0"^^xsd:decimal ;
                  sh:message "Object starts behind the robot" ] .
```
