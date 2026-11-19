# Object Position Coverage

This guide describes the object-position coverage check, the data-verification
tool of the course project. It extends steps 3 and 6 of
[Getting Started](getting_started.md): every dataset is checked before training,
and the results are used again when a trained policy fails. After a rollout, the
dataset is also compared with the positions the rollout actually used
([section 8](#8-compare-the-dataset-with-a-rollout)).

All of it goes through one **verification store** per dataset: a directory of
RDF (Turtle) files that records the dataset, the group's specifications, the
rollouts, and every verification result, and that can be queried with SPARQL
([section 9](#9-query-the-store)).

Every section starts with one of two labels: **Required** (something students
must do or follow) or **Background** (explanation without work items).

## 1. What the check does and why

**Background**

Imitation-learning policies tend to perform well only in situations that the
demonstrations represent well. If evaluation places objects where the dataset
has few or no episodes, the policy may fail because of missing data rather than
because of the model.

Object position is the factor checked here because it is the main thing that
changes between episodes: every reset places the objects at a random position,
while the scene, robot, cameras, and task stay the same. Object positions can also
be read directly from the simulator, without labeling or estimation.

**Why position and not the full pose.** The coverage check considers an object's 
position only, rather than its full pose (position and orientation) for the following reasons:

- **Position is the primary variable of interest.** The tasks randomize only the position. 
  Each object starts with a fixed rotation, so the orientation is the same in every episode 
  and says nothing about coverage.
- **Keeping orientation fixed improves episode reliability.** Randomizing initial orientations 
  could cause objects such as cups or cutlery to start in unstable configurations, potentially 
  leading them to tip over or fall off the table. Such episodes may become unusable, requiring 
  environment resets and additional simulation attempts, thereby wasting computational resources 
  on episodes that are never recorded.
- **Position allows straightforward spatial discretization.** Coverage is measured on the table 
  plane using the (x, y) coordinates. Positions in metres can be directly mapped onto a spatial 
  grid, whereas orientations represented by quaternions require more complex discretization and 
  have representational ambiguities (e.g., two quaternions can represent the same rotation).

Groups that randomize object orientations in their own experiments can incorporate 
orientation as an additional coverage factor ([Adding a Coverage Factor](adding_coverage_factor.md)).

The check answers one question: **where on the table does the dataset actually
have episodes, measured against a requirement that the group writes?** The
group states, per object, a rectangle of the table, a grid that divides it into
cells, and a minimum number of episodes per cell. The check counts the episodes
whose object *started* in each cell and reports the cells below the minimum.

```text
record ──▶ ingest ──▶ init spec ──▶ fill in ──▶ attach ──▶ verify ──▶ query / collect where cells are thin ──┐
  ▲  (object_position.* columns)                                                                             │
  └──────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

| Step | Runs in | Command (`uv run python scripts/store.py ...`) |
|---|---|---|
| Record, with object positions | Simulator environment | The group's collection script with the repository recorder ([How to Record Episodes §5.2](how_to_record.md#52-the-repositorys-recorder-and-object-positions)) |
| Put the dataset into a store | Host `uv` environment | `ingest --dataset` |
| Create a specification skeleton | Host `uv` environment | `init` |
| Fill in the requirement | Any editor | [Section 3](#3-write-the-specification) |
| Attach the specification to the store | Host `uv` environment | `attach` |
| Check | Host `uv` environment | `verify` |
| Look at the results | Host `uv` environment | `explore`, `query` ([section 9](#9-query-the-store)) |

The store reads only `meta/info.json`, `meta/tasks.parquet` and
`data/**/*.parquet` of the dataset. It never starts the simulator and never
decodes video, so every step runs in seconds. The LeRobot dataset itself remains unchanged, and can still be used for training and uploaded to the Hugging Face Hub.

The values of the requirement (area, grid, minimum) are the group's decision.
Running the check is required for every dataset; the report is evidence for the
group's own analysis, not a grade.

## 2. The object-position columns

**Background**

The check needs the `object_position.*` columns. Only the repository's
`LeRobotRecorderManager` writes them; [How to Record Episodes
§5.2](how_to_record.md#52-the-repositorys-recorder-and-object-positions) explains
the recorder. Each column has this format:

| Key | dtype | Shape | Content |
|---|---|---|---|
| `object_position.<name>` | float32 | `(3,)` | `[x, y, z]` |

- **Position**: metres, relative to the environment origin. This is the frame
  of `init_state.pos` in the task configurations, so those numbers and the
  numbers in a specification are directly comparable.
- **No orientation**: the start rotation is fixed in every task
  ([section 1](#1-what-the-check-does-and-why)).
- **Every frame** is recorded. The check uses the first recorded frame of each
  episode, which is where the objects started.
- **Name**: the object's scene name (`blue_cup`, `fork`, ...).

The recorded objects are those in `tracked_object_names` of the task
configuration: the objects whose start position is randomized. Objects that
start at the same place in every episode are listed in `fixed_object_names`
instead. They get no column and are not verified, because their position never
varies; rollout logs their pose once, in the log's header, for reference only
([section 8](#8-compare-the-dataset-with-a-rollout)).

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

A quick test after recording a few episodes:

```bash
uv run python -c "import json; f=json.load(open('datasets/<HF_USER>/<dataset_repo>/meta/info.json'))['features']; print([k for k in f if k.startswith('object_position.')])"
```

The output lists one key per tracked object. An empty list means that the
dataset was recorded with LeIsaac's recorder and has to be recorded again.

## 3. Write the specification

**Required**

A specification is a small text file in [Turtle](https://www.w3.org/TR/turtle/)
format. For each object, it says: *"the initial positions in the dataset should
cover this rectangle, split into this grid, with at least this many episodes
per cell."*

### 3.1 Put the dataset into a store

```bash
uv run python scripts/store.py ingest \
    --dataset datasets/${HF_USER}/<dataset_repo> \
    --task <task_id>
```

This creates `stores/<dataset_repo>/dataset.ttl`: the dataset, each episode
with its frame count, and each tracked object's initial position, plus the
dataset's path, a fingerprint of its files, and the time of the ingest.
`--task` (optional) records the gym task the dataset was recorded with. The
store is named after the dataset directory; `--name` chooses another name.

Run `ingest --dataset` again after recording more episodes. It rewrites
`dataset.ttl` and removes the previous verification results, which described
the old episodes; attached specifications and ingested rollouts stay.

### 3.2 Create the skeleton

Object names are taken from the dataset's `object_position.*` columns:

```bash
uv run python scripts/store.py init \
    --store stores/<dataset_repo> \
    --out specs/my_coverage.ttl
```
Alternatively, specify object names explicitly by using `--objects blue_cup,pink_cup` instead. `init` never
overwrites an existing file. The raw template is
`packages/data_verification/src/data_verification/axes/object_position/template.ttl`;
it can also be copied by hand, replacing `OBJECT_NAME`.

### 3.3 One object block

The specification declares one object-position axis at the top
(`spec:object_position_axis a cp:ObjectPositionAxis .`). Each object block adds
one requirement to it and describes the object, the area, and the grid:

```turtle
# Already in my_dataset: 40 episode(s), initial x 8.930..9.050, y 6.400..6.560. For reference only.
spec:object_position_axis cp:hasRequirement spec:blue_cup_requirement .

spec:blue_cup_requirement a cp:ObjectPositionAxisRequirement ;
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
| `cp:numCellW`, `cp:numCellH` | Number of cells along x (width) and along y (height) | Whole number, no quotes (`3`) |
| `cp:minEpisodesPerCell` | Episodes needed in each cell | Whole number ≥ 1, no quotes (`2`) |

- Deleting a block stops checking that object; copying a block checks another
  one.
- The links (`cp:hasRequirement`, `cp:hasTaskObject`, `cp:hasAreaConfiguration`,
  `cp:hasCoverageGrid`) stay as generated.
- The `# Already in ...` comment shows the range that the dataset has now. It
  is information only: the area describes where coverage is *wanted*, so
  copying the current range would hide exactly the gaps the check is meant to
  find.
- A store can hold several specifications, for example a coarse and a fine
  grid, or one per evaluation condition. Each is attached under its own name
  (the file name without `.ttl`) and verified separately.

### 3.4 Choosing the values

These values are the group's decision. The following questions help.

**Area (`areaXMin` .. `areaYMax`)**

- Where could the object plausibly start at test time? The public eval file
  shows each object's nominal position (`init_state.pos`) and its randomization
  range (`pose_range`); the private test uses different poses. The table, the
  robot's reach, and the camera views also limit the area.
- Can the robot reach and grasp the object everywhere in the area?
  Covering unreachable corners wastes collection time.
- The area is set per object; objects of the same task can have different
  areas.

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

**Example** (illustrative values, not a recommendation):

```turtle
    cp:minEpisodesPerCell 2 .                    # in the requirement
    cp:areaXMin 0.40 ; cp:areaXMax 0.70 ;        # in the area
    cp:areaYMin -0.15 ; cp:areaYMax 0.15 .
    cp:numCellW 3 ; cp:numCellH 3 .              # in the grid
```

This is a 30 cm × 30 cm square split into nine 10 cm cells, each needing 2
episodes.

## 4. Run the check and read the report

**Required**

Attach the filled-in specification, then verify:

```bash
uv run python scripts/store.py attach --store stores/<dataset_repo> --spec specs/my_coverage.ttl
uv run python scripts/store.py verify --store stores/<dataset_repo>
```

`attach` checks the specification first (no `TODO` left, numbers without
quotes, non-empty areas, a registered axis) and refuses it with an `ERROR` line
otherwise. It then copies the specification into
`stores/<dataset_repo>/specs/my_coverage.ttl`, renaming its nodes
(`spec:blue_cup_grid` becomes `…/spec/my_coverage/blue_cup_grid`) so that
several specifications can share the store, and adds the grid cells with their
bounds. **After editing the specification, attach it again**; `verify` refuses
a specification whose file changed since it was attached.

`verify` checks the dataset against every attached specification, prints one
report per specification, and writes the results into
`stores/<dataset_repo>/findings/`. Example output:

```
Store   : stores/my_dataset
Spec    : my_coverage  (specs/my_coverage.ttl)
Episodes: 31

[blue_cup] x=8.800..9.100  y=6.290..6.590  covered 6/9 cells (>= 2 each)
  y2   0!  0!  0!
  y1    5   3   5
  y0    5   5   7
        x0  x1  x2

VIOLATION 'blue_cup' cell (i=0, j=2) x=[8.8, 8.9] y=[6.49, 6.59] has 0 episode(s); needs at least 2  [spec/my_coverage/blue_cup_grid/cell/6]
WARNING   Episode 30: initial position of 'blue_cup' at (9.5, 6.4) is outside the required area; it does not count toward coverage  [my_dataset]

RESULT: coverage spec NOT met
```

How to read it:

- **Grid**: each number is the count of episodes whose object started in that
  cell; `!` marks a count below the minimum. Columns are x (`x0` is the
  `areaXMin` side), rows are y, with **+y at the top**.
- **VIOLATION**: a requirement is not met. Each thin or empty cell is one line
  with its exact bounds; these are the places to collect next. The node name
  at the end (`spec/my_coverage/blue_cup_grid/cell/6`) is the cell's row-major
  index `k = j × numCellW + i`.
- **WARNING**: allowed but worth knowing. An initial position outside the
  area does not count toward any cell; the message names the episode.
- **ERROR**: the check could not run (missing column, the dataset or the
  specification changed after it entered the store, ...). See
  [section 5](#5-troubleshooting).

Every line of the report is also stored as a `cp:Finding` in
`findings/my_coverage.ttl`, together with each cell's episode count
(`cp:episodeCount`), so the same results can be queried later
([section 9](#9-query-the-store)). `explore` prints a summary of the store:

```bash
uv run python scripts/store.py explore --store stores/<dataset_repo>
```

| Option of `verify` | Effect |
|---|---|
| `--spec my_coverage` | Verifies only this attached specification; repeatable (default: all) |
| `--no-rollouts` | Checks the dataset only, without comparing rollouts ([section 8](#8-compare-the-dataset-with-a-rollout)) |
| `--json report.json` | Also writes the full result (counts per cell, all findings) as JSON |
| `--strict` | Treats warnings as failures |
| `--extra-shapes my_rules.ttl` | Also runs the group's own SHACL rules ([section 7](#7-how-it-works-rdf-owl-shacl)); repeatable |

Exit code: `0` when every specification is met, `1` when one is not or when an
error occurs. A collection script can use it, for example, to stop collecting
once coverage is reached (run `ingest --dataset` before each `verify`).

| Other command | Effect |
|---|---|
| `detach --store … --spec my_coverage` | Removes the specification and its results from the store |

## 5. Troubleshooting

**Background**

| Message | Cause | Fix |
|---|---|---|
| `… is not a store (no dataset.ttl)` | `--store` points to a directory without an ingested dataset | Run `ingest --dataset` first, or fix the path |
| `The dataset … changed after it entered the store; run: ingest --dataset …` | Episodes were recorded or files changed after the last ingest | Run the printed `ingest` command, then `verify` |
| `The spec … changed after it entered the store; run: attach --spec …` | The specification file was edited after `attach` | Run the printed `attach` command, then `verify` |
| `Spec declares no generalization axis with a registered verifier` | The axis line (`spec:object_position_axis a cp:ObjectPositionAxis .`) is missing or names another class | Restore it from the template |
| `meta/info.json declares no 'object_position.<name>' features` | Dataset recorded with LeIsaac's recorder instead of the repository's | Record again with the repository recorder ([How to Record Episodes §5.2](how_to_record.md#52-the-repositorys-recorder-and-object-positions)) |
| `Dataset has no 'object_position.<name>' column, which the spec requires` | Block for an object that was not recorded, or a typo in `cp:name` | Fix the name or delete the block |
| `Spec still has TODO values: ...` | Unfilled fields | Fill them in ([section 3.3](#33-one-object-block)) |
| `... must be a number without quotes, got "8.8"` | Quoted number | Write `8.8`, not `"8.8"` |
| `... must be a whole number without quotes, got 2.5` | Fraction in `numCellW`, `numCellH`, or `minEpisodesPerCell` | Use a whole number |
| `Spec is not valid Turtle: ...` | Syntax error, usually a missing `;` or `.` | Compare with the template: `;` between the fields of one block, `.` at the end of a block |
| `coverage area is empty` | `areaXMin >= areaXMax` or `areaYMin >= areaYMax` | Swap the values |
| `numCellW must be at least 1` (or `numCellH`, `minEpisodesPerCell`) | Zero or negative grid size or minimum | Use a positive whole number |
| `Requirement is not linked from an ObjectPositionAxis` | The `spec:object_position_axis cp:hasRequirement ...` line of a block was deleted or names another requirement | Restore the line from the template |
| `Requirement is linked from more than one axis` | Two axes list the same requirement | Keep one `cp:hasRequirement` line per requirement |
| `Area must belong to exactly one requirement` (or `Grid must belong to exactly one area`) | Two blocks share one area or grid, or an area is not used by any block | Give each block its own area and grid, as in the template |
| `Dataset contains no episodes` | Empty dataset | Record episodes first |
| `episode(s) do not start at frame_index 0` | Dataset edited or written by another tool | Record with the repository recorder |
| `non-finite initial position` | NaN or infinity written as a position (an object fell out of the scene) | Inspect that episode; drop it or fix the reset |
| Every episode lands in one cell | Objects never moved between episodes | Check that the task's reset randomization is active |

## 6. What the check does not cover

**Background**

- **Only x and y position.** Orientation is not recorded; height (z) is
  recorded and stored in the store (`cp:positionZ`) but not checked.
- **Each object separately.** Combinations ("blue cup left *while* pink cup
  right") and distances between objects are not checked.
- **Only the first frame.** Trajectory quality, grasp success, and timing are
  not checked.
- **Rectangles only.** Other shapes need several requirements for the same
  object, or extra rules ([section 7](#7-how-it-works-rdf-owl-shacl)).

Good coverage raises the chance that a policy generalizes to new positions; it
does not guarantee it. Other factors can be added by following
[Adding a New Dataset Coverage Factor](adding_coverage_factor.md).

## 7. How it works: RDF, OWL, SHACL

**Background**

This section is not needed to use the tool.

| Piece | In one sentence | File (under `packages/data_verification/src/data_verification/`) |
|---|---|---|
| **RDF** | Data stored as `subject → predicate → object` facts, e.g. `episode/3/object/blue_cup/position → cp:positionX → 8.91` | Written into the store by `store.py` and `graph.py` |
| **OWL** | The vocabulary: kinds of things and the properties linking them (below) | `ontology/core.ttl` (store, dataset, rollout, results) and `axes/object_position/ontology.ttl` (the object-position axis) |
| **Specification** | The group's requirement, written as RDF in that vocabulary | The group's `.ttl` file, attached to the store |
| **SHACL** | Rules: each rule selects nodes (every episode, every cell, ...) and states what must hold. The coverage rule is a SPARQL query that counts the initial positions inside each cell | `shapes/dataset.ttl`, `shapes/rollout.ttl` (any axis) and `axes/object_position/shapes.ttl`, `rollout_shapes.ttl` |
| **Verifier** | One per generalization axis: its ontology, its SHACL rules, its specification template and optional Python steps (counts, tables, run placements), registered in `registry.py` | `verifier.py`, `registry.py`, `axes/object_position/` |

The store is one directory per dataset. Each file is rewritten as a whole, so
an interrupted command never leaves half a file:

```text
stores/<dataset_repo>/
├── dataset.ttl                       ingest --dataset
├── specs/<spec>.ttl                  attach
├── rollouts/<timestamp>.ttl          ingest --rollout-log  (section 8)
├── findings/<spec>.ttl               verify: the dataset against the spec
├── findings/<spec>__<timestamp>.ttl  verify: the rollout against the spec (section 8)
└── queries/                          the group's own SPARQL queries (section 9)
```

What enters the store is never derived: `ingest` and `attach` record facts and
the group's declarations, each with its source path, a hash and a time
(`cp:sourcePath`, `cp:sourceSha256`, `cp:ingestedAt`). `verify` adds result
nodes that point at those facts, and replaces them on the next run.

The vocabulary, as a tree (dotted links ┄ are added by `attach` or `verify`;
the rest is written in the specification or recorded by `ingest`).
Verification starts from the `Dataset`:

```text
Dataset (name, recordedObject, task, robotType, fps, sourcePath, sourceSha256)
├┄ hasGeneralizationAxis                          (attach)
│   ▶ ObjectPositionAxis                 (a GeneralizationAxis)
│      └─ hasRequirement
│          ▶ ObjectPositionAxisRequirement   (an AxisRequirement; minEpisodesPerCell)
│             ├─ hasTaskObject ▶ TaskObject (name)
│             └─ hasAreaConfiguration
│                 ▶ AreaConfiguration (areaXMin … areaYMax)
│                    └─ hasCoverageGrid
│                        ▶ CoverageGrid (numCellW, numCellH)
│                           └┄ hasCell ▶ Cell (index k, cellI, cellJ, cellXMin … cellYMax; episodeCount from verify)
└─ hasEpisode
    ▶ Episode (index = episode_index, frameCount)
       └─ hasTaskObject ▶ TaskObject (name)
              └─ hasPosition ▶ Position (positionX, positionY, positionZ)

Spec (name, sourcePath, sourceSha256)
└─ declaresAxis ▶ ObjectPositionAxis …

Verification (verifiedAt, passed)                 (verify)
├─ ofSpec ▶ Spec
├─ onDataset ▶ Dataset
└─ hasFinding ▶ Finding (severity, message, verifier, focusNode)
```

- The dataset *has* the axes of its attached specifications; their
  requirements apply to its episodes only. Each specification is verified
  with only its own axes.
- A requirement's object and the episodes' objects are all `TaskObject`s; the
  check matches them by `name`. Only the episodes' objects have a
  `Position`.
- A cell is identified by its index `k`, row-major: column `i = k mod numCellW`,
  row `j = ⌊k / numCellW⌋`. `attach` stores its column, row and bounds for
  queries; the SHACL rule computes the bounds again from the area and the
  grid. A position on an inner cell edge belongs to the upper cell; the
  area's upper edges belong to the last column and row.
- Every `Position` has `positionX`, `positionY` and `positionZ`, stored as
  decimals rounded to 6 places (micrometres). The coverage rule uses x and y
  only; z is there for inspection and for the group's own rules.
- A `Finding` has a severity: `Error` (the check could not be made),
  `Violation` (a requirement is not met), `Warning`, or `Info` (a suggested
  next step, [section 8](#8-compare-the-dataset-with-a-rollout)).

What `verify` does, per attached specification:

1. Loads `dataset.ttl` and the specification from the store, and stops with an
   `ERROR` when the dataset or the specification file changed since it entered
   the store.
2. Runs the SHACL rules of the store and of every axis of the specification
   over both with [`pyshacl`](https://github.com/RDFLib/pySHACL).
3. Runs each axis's Python step: for object position, the episode count of
   every cell and the count grid.
4. Prints each failed rule as one VIOLATION or WARNING line, plus the count
   grid, and writes the `Verification` with its findings and the cell counts
   to `findings/<spec>.ttl`.

Additional rules (for example "the object must start in front of the robot") are
written as SHACL shapes in a separate `.ttl` file and passed with
`verify --extra-shapes`. The packaged shapes in `axes/object_position/shapes.ttl`
are commented and are a good starting point. A minimal example:

```turtle
@prefix cp:  <http://taica.course/cp/ontology#> .
@prefix sh:  <http://www.w3.org/ns/shacl#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

cp:InFrontOfRobotShape a sh:NodeShape ;
    sh:targetClass cp:Position ;
    sh:property [ sh:path cp:positionY ; sh:minInclusive "0.0"^^xsd:decimal ;
                  sh:message "Object starts behind the robot" ] .
```

## 8. Compare the dataset with a rollout

**Required**

The dataset check compares the dataset with the group's own requirement. After
a policy has been evaluated, the comparison asks a second question: **did the
dataset cover the positions the rollout actually used?** It reads where each
rollout episode (run) started and whether it succeeded, and shows the runs that
started where the dataset is thin, together with the cells to record next.
Repeating record → train → rollout → verify closes those gaps one round at a
time.

### 8.1 Run the rollout

Every rollout ([Rollout §3](rollout.md#3-run-rollout)) writes
`outputs/rollout/<timestamp>.jsonl`, with no flag needed: one line per episode with each
randomized object's position right after the reset (env-local frame, the same frame as
the dataset columns) and the outcome. The header line before them records the
robot base and fixed object poses once (`robot_base`, `fixed_objects`); the
store keeps the fixed objects' positions (`cp:hasFixedObject`) but does not
compare them. Episodes stopped by a manual reset are marked `aborted`: they
are kept but not counted.

The log is written line by line while the rollout runs, so a rollout stopped
with Ctrl+C still leaves a readable log. It enters the store only after the
rollout has ended.

### 8.2 Compare

Put the log into the store of the dataset the evaluated policy was trained on,
then verify:

```bash
uv run python scripts/store.py ingest \
    --store stores/<dataset_repo> \
    --rollout-log outputs/rollout/<timestamp>.jsonl
uv run python scripts/store.py verify --store stores/<dataset_repo>
```

`ingest --rollout-log` writes `stores/<dataset_repo>/rollouts/<timestamp>.ttl`:
the rollout with its eval file, seed and checkpoint, and every run with its
status and initial positions. When the checkpoint has a `train_config.json`
naming a different dataset, the comparison warns.

`verify` checks the dataset (section 4) and then compares every ingested
rollout with every attached specification, writing each comparison to
`findings/<spec>__<timestamp>.ttl`. `--rollout <timestamp>` compares only that
rollout, `--no-rollouts` none.

How the comparison works, per object:

- **Cell of a run.** Each run is placed into a cell of **the group's own grid**
  for that object, by its initial position, with the same rule as the dataset check.
  A run that started outside the area is placed into a cell of the same size
  outside it, continuing the grid (for example column `i = -1` left of the
  area).
- **Covered.** A run is covered when it started inside the area, in a cell
  with at least `minEpisodesPerCell` dataset episodes. Only cells with at
  least one run are judged.
- **In distribution.** A run is in distribution when it is covered for every
  requirement of the specification at once.

The comparison does not read the eval file: the runs' own positions show
where the rollout placed the objects.

**Example.** The cup-stacking eval file randomizes `blue_cup` over x 8.8–9.0,
y 6.38–6.58 and `pink_cup` over x 9.0–9.2, y 6.38–6.58. The group's
specification and dataset are:

| Object | Area in the specification | Grid | Minimum | Episodes per cell of the group's grid |
|---|---|---|---|---|
| `blue_cup` | x 8.8–9.0, y 6.4–6.6 | 2 × 2 (cells of 0.1 × 0.1 m) | 3 | (i=0, j=0): 5, (i=1, j=0): 3, (i=0, j=1): 3, **(i=1, j=1): 1** |
| `pink_cup` | x 9.0–9.2, y 6.38–6.58 | 2 × 2 (cells of 0.1 × 0.1 m) | 3 | 3 in every cell |

`blue_cup`'s area starts at y 6.4, so it misses the bottom 2 cm of the eval
range, and its cell (i=1, j=1) has fewer than 3 episodes. A rollout with 20
episodes then gives:

```
Store   : stores/my_dataset
Spec    : my_coverage
Rollout : 20261103-142530  (eval/cup_stacking_eval.py, seed 42)
Runs    : 20 finished, 8 succeeded

[blue_cup] your area x=8.800..9.000 y=6.400..6.600, cell 0.1 x 0.1, 2x2 cells, >= 3 episodes each
  dataset episodes : rollout runs per cell (! = runs started where the dataset has fewer than 3 episodes)
  y1    3:0  1:5!
  y0    5:5   3:0
          x0    x1
  rollout successes / runs per cell (- = no runs)
  y1      -   1/5
  y0    4/5     -
          x0    x1
  10 run(s) started outside your area, in 2 cell(s) of the same size
  covered: 1/4 sampled cells, 5/20 runs (25%)
  success: 4/5 covered vs 4/15 not covered

[pink_cup] your area x=9.000..9.200 y=6.380..6.580, cell 0.1 x 0.1, 2x2 cells, >= 3 episodes each
  dataset episodes : rollout runs per cell (! = runs started where the dataset has fewer than 3 episodes)
  y1    3:5   3:5
  y0    3:5   3:5
          x0    x1
  rollout successes / runs per cell (- = no runs)
  y1    2/5   1/5
  y0    4/5   1/5
          x0    x1
  covered: 4/4 sampled cells, 20/20 runs (100%)
  success: 8/20 covered vs 0/0 not covered

In distribution (covered on every axis: object_position): 5/20 runs (25%)
  success 4/5 in distribution vs 4/15 outside

Record more episodes here (2 more at least; cells with failed rollout runs first):
  blue_cup cell (i=1, j=1) x=[8.900, 9.000] y=[6.500, 6.600] centre (8.950, 6.550): has 1, needs 2 more; 5 rollout run(s) started here, 4 failed

Rollout runs started outside your area. Grow it by whole cells (same cell size), then record there:
  blue_cup: area x=[8.800, 9.000] y=[6.400, 6.600] -> areaXMin 8.8 ; areaXMax 9 ; areaYMin 6.3 ; areaYMax 6.6 ; numCellW 2 ; numCellH 3

WARNING   'blue_cup': 3 of 4 sampled cell(s) are not covered by the dataset (15 run(s), 11 failed)  [blue_cup/cell/(1, 1), blue_cup/cell/(0, -1), blue_cup/cell/(1, -1)]

RESULT: the dataset does NOT cover every rollout run; see above
```

The last two blocks answer two questions:

- **Record more episodes here**: *Where is data missing?* Cells of the group's
  own grid to record more episodes in, without changing the specification.
- **Grow your area**: *Is the area too small?* The rollout placed objects
  outside the area of the specification; the area has to be enlarged first.

These two blocks are suggestions computed from the object-position axis, one
way to read the comparison. `--no-hints` leaves them out of the printed report;
they are still stored as `Info` findings. The group can draw its own
conclusions from the stored placements instead
([section 9](#9-query-the-store)).

How to read it:

| Part | Meaning | In the example above |
|---|---|---|
| **Runs** | Episodes of the rollout that ended with success or failure. Episodes stopped by a manual reset (`aborted`) are not counted. | The log has 20 episodes, none aborted; 8 succeeded. |
| **Header** `[blue_cup] your area … cell … NxM cells` | The group's area and grid for the object, from the specification. | Cell size (9.0 − 8.8) / 2 = 0.1 by (6.6 − 6.4) / 2 = 0.1. |
| **Dataset grid** `episodes:runs` | Per cell of the group's grid, +y at the top as in the dataset check: the dataset episodes that started there, then the rollout runs that started there. `!` = the cell has runs but fewer than `minEpisodesPerCell` episodes, so those runs are not covered. Outcomes are not in this grid. | `5:5` (x0, y0): 5 episodes, 5 runs: covered. `1:5!` (x1, y1): 1 episode, minimum 3, 5 runs: not covered. `3:0` (x0, y1): enough episodes, no runs, so not judged. |
| **Outcome grid** `successes/runs` | Per cell: how many of the runs that started there succeeded. `-` = no runs. Says nothing about the dataset; read it next to the dataset grid. | `4/5` (x0, y0): 4 of the 5 runs succeeded. `1/5` (x1, y1): 1 of 5 succeeded, in the cell the dataset grid marks thin. |
| **outside your area** | Runs that started outside the area, and how many cells of the grid's size they fall into. They are not covered. | 10 runs started at y 6.38–6.4, below `areaYMin` 6.4: in cells (0, −1) and (1, −1). |
| **covered** | Covered cells / cells with runs (outside cells included), and the runs in covered cells / all runs. | `blue_cup`: only (x0, y0) of the 4 sampled cells is covered, so 1/4 cells and 5/20 runs. `pink_cup`: 4/4 cells, 20/20 runs. |
| **success** | Success rate of the covered runs compared with the other runs. | `blue_cup`: 4/5 vs 1 + 1 + 2 = 4 of 15. `pink_cup`: all 20 runs covered, so 8/20 vs 0/0. |
| **In distribution** | Runs covered for **every** requirement at once, with their success rate against the rest. A much lower rate outside points to missing data rather than the model. | A run counts only if both cups started in covered cells. All `pink_cup` runs are covered, so this equals `blue_cup`'s 5/20; success 4/5 (80 %) inside vs 4/15 (27 %) outside. |
| **Record more episodes here** — *where is data missing?* | Cells of the group's grid with runs and fewer episodes than the minimum. Sorted by failed runs, then by missing episodes. `N more at least` is the sum of `needs … more`. Recording there needs no change to the specification. | Cell (i=1, j=1), x 8.9–9.0, y 6.5–6.6, has 1 episode; it needs 2 more. 5 runs started there, 4 failed. |
| **Grow your area** — *is the area too small?* | Runs started outside the group's area. The suggested area is enlarged by whole cells, so the cell size and the cells already recorded stay the same. After updating the specification, run `attach` and `verify` again to find the new cells that need episodes. | The runs at y 6.38–6.4 fall into row j = −1 (y 6.3–6.4): `areaYMin` 6.4 − 0.1 = 6.3 and `numCellH` 2 + 1 = 3. |
| **WARNING** | One line per object with uncovered cells, named `(i, j)` as in the grid. | `blue_cup`: (1, 1), (0, −1) and (1, −1); 5 + 4 + 6 = 15 runs, 11 failed. |
| **ERROR** | The comparison could not run (malformed log, a source changed since it entered the store, ...); see the table below. | None. |
| **RESULT** | Whether every run started in a covered cell. | 15 runs did not, so NOT. |

| Option of `verify` | Effect |
|---|---|
| `--rollout <timestamp>` | Compares only this ingested rollout; repeatable (default: all) |
| `--no-hints` | Leaves the suggestions out of the printed report |
| `--json result.json` | Also writes the full comparison (all cells, runs, suggestions) as JSON |
| `--strict` | Exits `1` when the dataset does not cover every run |

Exit code: a comparison alone does not fail `verify` (warnings list the
uncovered runs); an error such as a changed log does, and so do warnings with
`--strict`.

| Message | Cause | Fix |
|---|---|---|
| `... no finished run (all aborted or none run)` | Every episode of the log was aborted | Run the rollout again |
| `The rollout … changed after it entered the store; run: ingest --rollout-log …` | The log was written to again after `ingest`, for example the rollout was still running | Run the printed `ingest` command after the rollout has ended |
| `ingest --rollout-log needs --store` | No store given for the log | Pass the store of the training dataset |
| `Finished run N has no initial position for '...'` | The specification names an object the rollout log does not record | Check the object name in the specification |
| `... checkpoint was trained on '...', not on ...` (warning) | The log was put into the store of another dataset than the one the policy learned from | Ingest the log into the training dataset's store |
| `'...' starts at different positions across the rollout's runs, but the spec has no requirement for it` (warning) | No block for that object in the specification | Add a block ([section 3.3](#33-one-object-block)) |

The comparison uses the public eval file only. The final ranking uses a
separate configuration with different positions
([Rollout §1](rollout.md#1-what-rollout-does)), so covering the public range
is necessary but not sufficient. As in the dataset check, only x and y are
compared, and each object separately.

### 8.3 How it works

**Background**

The comparison starts from a `Rollout`, which has the dataset its policy was
trained on (dotted links ┄ are added by `verify`):

```text
Rollout (name, evalFile, evalSha256, seed, checkpoint, trainedOnRepo, successEpisodes, totalEpisodes)
├─ hasDataset
│   ▶ Dataset                     the store's dataset, as in section 7
├─ hasFixedObject ▶ TaskObject (name) ─ hasPosition ▶ Position
└─ hasRun
    ▶ Run (index, status)         one rollout episode: succeeded, failed or aborted
       └─ hasTaskObject
          ▶ TaskObject (name)
              └─ hasPosition ▶ Position (positionX, positionY, positionZ)

Comparison (verifiedAt, runsInDistribution)       (verify)
├─ ofSpec ▶ Spec
├─ ofRollout ▶ Rollout
├─ inDistributionRun ▶ Run                        every axis covers it
├─ hasPlacement
│   ▶ RunPlacement (cellI, cellJ, covered)        one per finished run and requirement
│      ├─ ofRun ▶ Run
│      ├─ ofRequirement ▶ ObjectPositionAxisRequirement
│      └─ inCell ▶ Cell                           only inside the area
└─ hasFinding ▶ Finding
```

- `ingest --rollout-log` adds one `Rollout` with all its runs.
  `totalEpisodes` counts the finished runs and `successEpisodes` the
  succeeded ones; aborted runs are kept but counted in neither.
- `Run` and `Episode` are different classes, so a run never counts as dataset
  coverage.
- SHACL rules in `shapes/rollout.ttl` check the rollout side (one training
  dataset, one status per run, counts that match the runs) and
  `axes/object_position/rollout_shapes.ttl` checks that every finished run has
  a position for every required object. Placing runs into cells and the
  suggestions are computed in `axes/object_position/compare.py`.

## 9. Query the store

**Background**

Every fact and result in the store can be asked for with
[SPARQL](https://www.w3.org/TR/sparql11-query/), locally with `rdflib` (no
server):

```bash
uv run python scripts/store.py query --store stores/<dataset_repo> \
    --query-file queries/uncovered_cells.rq
uv run python scripts/store.py query --store stores/<dataset_repo> \
    --query-text 'SELECT (COUNT(?e) AS ?n) WHERE { ?e a cp:Episode }'
```

`SELECT` prints a table (`--format csv` or `--format json` for files), `ASK`
prints `true` or `false`, and `CONSTRUCT`/`DESCRIBE` print Turtle. The
prefixes `cp:`, `rdf:`, `rdfs:`, `owl:` and `xsd:` are predefined.
`--with-ontology` also loads the classes and their labels.

The repository's `queries/` directory has examples to start from:

| File | Question |
|---|---|
| `episodes.rq` | Where did each object start in each recorded episode? |
| `cell_counts.rq` | How many dataset episodes does each cell have? |
| `uncovered_cells.rq` | Which cells have fewer episodes than required, and how many more each needs? |
| `run_outcomes_by_cell.rq` | How many rollout runs started in each cell, and how many succeeded? |
| `failed_runs_not_covered.rq` | Which failed runs started where the dataset does not cover? |
| `findings.rq` | Every stored finding, by specification, rollout and severity |

For example, the runs per cell of a rollout, with their outcomes:

```sparql
SELECT ?object ?i ?j ?covered (COUNT(?run) AS ?runs) (SUM(?ok) AS ?successes) WHERE {
    ?c a cp:Comparison ; cp:hasPlacement ?pl .
    ?pl cp:ofRun ?run ; cp:ofRequirement ?req ; cp:cellI ?i ; cp:cellJ ?j ; cp:covered ?covered .
    ?req cp:hasTaskObject ?o .
    ?o cp:name ?object .
    ?run cp:status ?status .
    BIND (IF(?status = "succeeded", 1, 0) AS ?ok)
}
GROUP BY ?object ?i ?j ?covered
```

Numbers are compared with `FILTER`: `?e cp:index ?i . FILTER (?i = 2)` finds
episode 2, while `?e cp:index 2` finds nothing, because indices are stored as
`xsd:nonNegativeInteger` and `2` is an `xsd:integer`. Keep the group's own
queries in `stores/<dataset_repo>/queries/`.
