# Adding a New Dataset Coverage Factor

This guide describes how to extend data verification to another factor, for
example the camera viewpoint, the object orientation, the table appearance, or
the motion speed. It extends step 6 of [Getting Started](getting_started.md).

It builds on [Data Verification](data_verification.md) and on the three
verifiers that already exist: they are complete worked examples, and this
guide points to them instead of repeating them.

| STAR-Gen axis | Verifier (`axes/<axis>/`) | Factor | Requirement class |
|---|---|---|---|
| Object Poses (VB-POSE) | `object_pose` | object position | `cp:ObjectPositionRequirement` |
| Image Augmentations (V-AUG) | `image_augmentation` | lighting | `cp:LightingRequirement` |
| Object Properties (S-PROP) | `object_property` | referencing objects by color | `cp:ColorReferenceRequirement` |

`image_augmentation` is the shortest of the three and the closest model for a
new numeric factor; `object_property` is the model for a categorical one.

Every section starts with one of two labels: **Required** (steps of an
extension) or **Background** (explanation without work items).

## 1. Decide between a rule, a factor, and an axis

**Background**

| Question | Extension |
|---|---|
| Is every value valid or within a safety limit? | A SHACL rule |
| Does each episode contain the required metadata? | A SHACL rule |
| Are there enough episodes across several ranges or categories? | A coverage factor |

A validation rule checks individual facts. A coverage factor defines bins or
categories, counts episodes in each one, and compares those counts with a
requirement.

For example, "light intensity must be finite" is a rule. "Each of three
lighting ranges must contain at least five episodes" is a coverage requirement.
A rule alone needs no code: it is a SHACL file passed with
`scripts/store.py verify --extra-shapes`
([Data Verification §7](data_verification.md#7-how-it-works-rdf-owl-shacl)).
The rest of this guide is about coverage factors.

The repository follows the STAR-Gen taxonomy (Gao et al., "A Taxonomy for
Evaluating Generalist Robot Manipulation Policies", arXiv:2503.01238): one
**verifier** per STAR-Gen **axis**, and one **requirement class** per
**factor** of that axis. A new factor therefore takes one of two paths:

| The factor belongs to | Extension |
|---|---|
| An axis that has no verifier yet (for example a camera viewpoint) | A new verifier under `axes/<axis>/`, with its own requirement class (this guide) |
| An axis that already has a verifier (for example blur, a second factor of Image Augmentations) | A new requirement class in that verifier's `ontology.ttl`, `shapes.ttl` and `coverage.py`; sections 2, 4, 5, 8, 9 and 11 apply in the same way |

The axis class is named after the STAR-Gen axis, the requirement class after
the factor, and the dataset columns and rollout-log keys use the factor name.

## 2. Write a factor contract first

**Required**

The factor is defined in writing before any code. The running example of
this guide is a camera viewpoint factor:

```text
Factor: camera height
Dataset column: camera_height.front
dtype and shape: float32, (1,)
unit: metres, env-local frame
sampling: every recorded frame
episode value: first recorded frame
required domain: 1.10 to 1.25 m
bins: 3 equal-width bins
minimum: 5 episodes per bin
```

The contract specifies:

- the exact physical or semantic meaning;
- unit and coordinate frame, if applicable;
- dtype and shape;
- when the value is sampled;
- how frame values become one episode-level value;
- numeric bins or categorical values;
- what counts as sufficient coverage.

The episode aggregation is a deliberate choice. Initial scene factors normally
use the first frame, as all three existing verifiers do. Motion factors may
need a minimum, maximum, mean, or trajectory summary.

The column is named `<factor>.<name>`, like `lighting.rect_light` and
`object_color.fork`. A key starting with `observation.` is used only when the
factor is intentionally a policy input, because LeRobot turns every
`observation.*` key into a policy input.

## 3. The extension path

**Background**

The store, the commands (`ingest`, `verify`, `query`, `explore`) and the core
vocabulary stay as they are; the new verifier plugs into them:

```text
recorder ── writes <factor>.<name> per frame ──▶ LeRobot v3 dataset  ─┐
rollout.py ── logs factors.<factor> per episode ──▶ rollout JSONL log ┤
                                                                      │ ingest  (+ every verifier's ingest steps)
                                                                      ▼
                                      stores/<repo_name>/  (RDF: facts, specs, results)
specs (copies of specs/<task>/*.ttl) ── verify --spec ──────▶│ verify  (core SHACL + the SHACL rules and
                                                                      │          Python steps of each spec's verifiers)
                                                                      ▼
                                       findings/  ──▶ report, explore, SPARQL query
```

A verifier lives in its own package and is registered in one list:

```text
packages/data_verification/src/data_verification/
├── verifier.py                  the Verifier base class (every step optional)
├── registry.py                  VERIFIERS = [ObjectPoseVerifier(), ImageAugmentationVerifier(), ...]
├── dataset.py                   column readers (read_first_frame_vectors)
├── grid.py                      equal-width bins shared by the grid axes
├── spec_template.py             helpers for spec_errors() (TODO left, values not numbers)
└── axes/
    ├── object_pose/             Object Poses: object position
    ├── image_augmentation/      Image Augmentations: lighting
    ├── object_property/         Object Properties: referencing objects by color
    └── camera_viewpoint/        the new axis
        ├── __init__.py          class CameraViewpointVerifier(Verifier)
        ├── ontology.ttl         axis class, requirement class, properties     (section 6)
        ├── shapes.ttl           SHACL rules on the dataset                   (section 9)
        ├── rollout_shapes.ttl   SHACL rules on a rollout                     (section 11)
        └── coverage.py          counts per bin (check) and run placements (compare)
specs/<task>/camera_viewpoint.ttl   the shipped spec, one per task           (section 8)
```

| `Verifier` member | Used by | Needed for |
|---|---|---|
| `name`, `axis_class`, `package` | every command | always; `name` is also the file name of the shipped spec |
| `ontology.ttl` | `verify`, `query --with-ontology` | always: the axis class must subclass `cp:GeneralizationAxis` |
| `ingest_dataset()`, `ingest_rollout()` | `ingest` | adding the factor's values to episodes and runs (sections 7 and 11) |
| `recorded_names()` | the verifier's own `dataset_errors()` | the names the dataset has a column for (default: the recorded objects) |
| `spec_errors()`, `derive_spec()` | `verify`, when it loads a spec | refusing unfilled specs; adding derived nodes to the spec (section 8) |
| `shapes.ttl`, `dataset_errors()`, `check()` | `verify` (dataset) | the coverage rule; counts and tables (section 9) |
| `rollout_shapes.ttl`, `compare()` | `verify` (rollouts) | the rollout comparison (section 11) |

`verify` runs a verifier only for the specs that declare its axis class. A
verifier that only needs SHACL rules leaves every Python step at its default.
OWL describes what the facts mean, while SHACL validates them. Neither can
recover a factor that was never recorded in the dataset.

## 4. Record the factor in LeRobot v3

**Required**

The repository's `LeRobotRecorderManager` in
[`lerobot_recorder.py`](../packages/simulator/src/simulator/utils/lerobot_recorder.py)
is the base for every verification factor: it already writes the
`object_position.*`, `lighting.*` and `object_color.*` columns and each
episode's task text
([Data Collection §5](data_collection.md#5-the-recorder-and-the-dataset)).
A new factor is added next to them; the existing columns stay.

1. **Say what to record in the task configuration.** Like
   `recorded_light_names`, a list in
   [`single_arm_franka_cfg.py`](../packages/simulator/src/simulator/tasks/template/single_arm_franka_cfg.py)
   names what the recorder reads, for example
   `recorded_camera_names: list[str] = []`, set to `["front"]` in the task.
2. **Read the value and declare the feature.** Values read from the scene go
   in [`scene_factors.py`](../packages/simulator/src/simulator/utils/scene_factors.py),
   next to `light_values()` and `object_colors()`, with a feature builder like
   `lighting_features()`:

   ```python
   CAMERA_HEIGHT_KEY_PREFIX = "camera_height."

   def camera_height_features(camera_names: list[str]) -> dict[str, dict]:
       return {
           f"{CAMERA_HEIGHT_KEY_PREFIX}{name}": {"dtype": "float32", "shape": (1,), "names": ["z"]}
           for name in camera_names
       }

   def camera_heights(scene, name: str, env_ids) -> list[list[float]]:
       """[z] of the camera ``name`` for each env in ``env_ids``, relative to the env origin."""
       ...
   ```

3. **Write it per frame.** In `lerobot_recorder.py`, add the feature in
   `build_with_factors` (where the comment says "Append the feature specs of
   further verification factors here") and the value in `scene_frame()`, which
   is called for every frame of environment 0. A value that comes from the
   recorded step instead of the scene goes in `build_lerobot_frame()` of the
   template, like the object positions.

Before any RDF work, one short episode is recorded and its raw output inspected:

```bash
uv run python -c "import json; print(json.load(open('datasets/${HF_USER}/<repo_name>/meta/info.json'))['features'].get('camera_height.front'))"

uv run python - <<'PY'
import pyarrow.parquet as pq

table = pq.read_table(
    "datasets/${HF_USER}/<repo_name>/data/chunk-000/file-000.parquet",
    columns=["episode_index", "frame_index", "camera_height.front"],
)
print(table.slice(0, 5))
PY
```

The first command prints the declared feature from `meta/info.json`. The second
prints the first five rows of the column (`columns` limits what is read from
the Parquet file). This separates recording failures from verifier failures.
Only environment 0 is recorded, so the recording runs with one environment.

## 5. Read and aggregate the column

**Required**

[`dataset.py`](../packages/data_verification/src/data_verification/dataset.py)
already reads first-frame vectors: `read_first_frame_vectors(root, prefix,
width)` returns, for every `<prefix><name>` column, the value of each
episode's first frame, checks the declared shape, and returns nothing when the
dataset has no such column. The lighting and object-color verifiers use it
with `LIGHTING_KEY_PREFIX` and `OBJECT_COLOR_KEY_PREFIX`; a new factor adds
its prefix next to them (and keeps it in sync with `scene_factors.py`).

A factor with another aggregation adds its own reader. It:

1. Verifies the feature declaration in `meta/info.json`.
2. Verifies that the column exists in every Parquet shard.
3. Groups rows by `episode_index`.
4. Applies the aggregation from the contract.
5. Reports missing, null, `NaN`, or infinite values with episode numbers.

Parquet row order is not guaranteed. The readers sort by `episode_index` and
`frame_index` before selecting the initial value.

## 6. Describe the axis with RDF and OWL

**Required**

The axis's vocabulary goes in `axes/<axis>/ontology.ttl`. The axis class
subclasses `cp:GeneralizationAxis` and the requirement class subclasses
`cp:AxisRequirement`, so they reuse `cp:hasRequirement` and
`cp:hasGeneralizationAxis` from `ontology/core.ttl`: when `verify` loads a
spec, the dataset gets every axis the spec declares. Episodes (with their
`cp:index`), runs, the dataset, specs and findings are shared with the other
axes. The annotations `cp:starGenAxis` and `cp:starGenFactor` record where the
axis and the factor come from. Each class is described by what its members
have (`rdfs:subClassOf` an `owl:Restriction`), as in
`axes/image_augmentation/ontology.ttl`. A minimal vocabulary:

```turtle
@prefix cp:   <http://taica.course/cp/ontology#> .
@prefix owl:  <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd:  <http://www.w3.org/2001/XMLSchema#> .

cp:CameraViewpointAxis a owl:Class ;
    rdfs:subClassOf cp:GeneralizationAxis ,
        [ a owl:Restriction ; owl:onProperty cp:hasRequirement ; owl:allValuesFrom cp:CameraHeightRequirement ] ;
    rdfs:label "Camera Viewpoints"@en ;
    cp:starGenAxis "<axis code from STAR-Gen>" .

cp:CameraHeightRequirement a owl:Class ;
    rdfs:subClassOf cp:AxisRequirement ,
        [ a owl:Restriction ; owl:onProperty cp:cameraName ; owl:qualifiedCardinality "1"^^xsd:nonNegativeInteger ; owl:onDataRange xsd:string ] ,
        [ a owl:Restriction ; owl:onProperty cp:heightMin ; owl:qualifiedCardinality "1"^^xsd:nonNegativeInteger ; owl:onDataRange xsd:decimal ] ,
        [ a owl:Restriction ; owl:onProperty cp:heightMax ; owl:qualifiedCardinality "1"^^xsd:nonNegativeInteger ; owl:onDataRange xsd:decimal ] ,
        [ a owl:Restriction ; owl:onProperty cp:numBinsHeight ; owl:qualifiedCardinality "1"^^xsd:nonNegativeInteger ; owl:onDataRange xsd:positiveInteger ] ,
        [ a owl:Restriction ; owl:onProperty cp:minEpisodesPerBin ; owl:qualifiedCardinality "1"^^xsd:nonNegativeInteger ; owl:onDataRange xsd:positiveInteger ] ;
    rdfs:label "Camera Height Requirement"@en ;
    cp:starGenFactor "camera height" .

# One value per episode (dataset) and per finished run (rollout).
cp:cameraHeight a owl:DatatypeProperty ; rdfs:range xsd:decimal .
cp:recordedCamera a owl:DatatypeProperty ; rdfs:range xsd:string .
cp:cameraName a owl:DatatypeProperty ; rdfs:range xsd:string .
cp:heightMin a owl:DatatypeProperty ; rdfs:range xsd:decimal .
cp:heightMax a owl:DatatypeProperty ; rdfs:range xsd:decimal .
cp:numBinsHeight a owl:DatatypeProperty ; rdfs:range xsd:positiveInteger .
cp:minEpisodesPerBin a owl:DatatypeProperty ; rdfs:range xsd:positiveInteger .
```

A property name that another axis already defines (such as
`cp:minEpisodesPerCell`) is reused, not defined again. OWL defines the
concepts and relationships; executable constraints belong in SHACL.

## 7. Add the values to the store at ingest

**Required**

`ingest --dataset` writes the dataset, its episodes, the object positions and
each episode's task text, then calls `ingest_dataset(root, graph, dataset)` of
every registered verifier. The verifier reads its column (section 5) and adds
the value to each episode, whose IRI is `<dataset>/episode/<episode_index>`:

```python
class CameraViewpointVerifier(Verifier):
    name = "camera_viewpoint"
    axis_class = CP.CameraViewpointAxis
    package = __name__

    def recorded_names(self, graph, dataset):
        return sorted(str(n) for n in graph.objects(dataset, CP.recordedCamera))

    def ingest_dataset(self, root, graph, dataset):
        values = read_first_frame_vectors(root, CAMERA_HEIGHT_KEY_PREFIX, 1)   # camera -> episode -> (z,)
        errors = []
        for camera, episodes in sorted(values.items()):
            graph.add((dataset, CP.recordedCamera, Literal(camera)))
            for episode_index, (z,) in sorted(episodes.items()):
                if not math.isfinite(z):
                    errors.append(f"Episode {episode_index}: non-finite camera height for '{camera}'")
                    continue
                add_camera_height(graph, URIRef(f"{dataset}/episode/{episode_index}"), camera, z)
        return errors                                                         # non-empty: ingest stops
```

`ImageAugmentationVerifier.ingest_dataset()` is the same code for the
lighting; `add_light_condition()` in `graph.py` is the model for a helper such
as `add_camera_height()`, which adds one node per camera to the episode.
`ingest_rollout(log, graph, rollout)` does the same for each run of a rollout
log (section 11). Extraction stays in Python and validation in RDF/SHACL, so
the store can be inspected and queried without starting Isaac Sim. Ingest only
records facts: no counting, no judgment. A dataset recorded before the factor
existed still ingests; a spec that requires the factor then fails with the
error from `dataset_errors()` (section 9).

## 8. Ship a spec file

**Required**

There is no spec generator: the shipped specs are copied from
`specs/<task>/` and every `TODO` is replaced
([Data Verification §3](data_verification.md#3-specs)). A new verifier
therefore ships one spec per task, named after the verifier:
`specs/<task>/camera_viewpoint.ttl`. The existing spec files are the model:
a header comment that explains the factor and the `verify` command, then one
commented block per requirement, with every value `"TODO"`:

```turtle
@prefix cp:   <http://taica.course/cp/ontology#> .
@prefix spec: <http://taica.course/cp/spec/mine#> .

spec:camera_viewpoint_axis a cp:CameraViewpointAxis ;
    cp:hasRequirement spec:front_camera_height .

spec:front_camera_height a cp:CameraHeightRequirement ;
    # Must match the dataset column camera_height.<name>.
    cp:cameraName "front" ;
    # Range of the camera height to cover, metres. Numbers without quotes.
    cp:heightMin "TODO" ;
    cp:heightMax "TODO" ;
    cp:numBinsHeight "TODO" ;
    cp:minEpisodesPerBin "TODO" .
```

Requirement values come from the expected evaluation conditions, the robot's
limitations, and the collection budget. Copying the minimum and maximum
already found in the dataset would hide the missing region that the check is
meant to detect; the comments in the spec say so.

When `verify` loads a spec it renames every `spec:` node under the store's IRI
for that spec (the file name without `.ttl`), so copies of the same shipped
file never clash. Before that, `spec_errors()` returns the problems that stop
the spec (a `TODO` left, a quoted number, an empty range); the spec is then
reported as an `Error` and the other specs are still checked. The helpers
`todo_fields()`, `value_type_errors()` and `number()` in
[`spec_template.py`](../packages/data_verification/src/data_verification/spec_template.py)
do most of the work; `ImageAugmentationVerifier.spec_errors()` is the example.
`spec_errors()` returns nothing for a spec that does not declare its axis,
because one file may declare several axes. `derive_spec()` may add derived
nodes to the stored spec; `object_pose` adds its grid cells there.

`make test` checks every file under `specs/<task>/`: its name must be a
verifier name, and `spec_errors()` must report only the `TODO` values.
`test_every_task_ships_a_spec_per_axis` expects one file per registered
verifier for each predefined task.

## 9. Add bins and SHACL constraints

**Required**

Bins come from the requirement:

- numeric factor: intervals, with the edge rule of
  [`grid.py`](../packages/data_verification/src/data_verification/grid.py)
  (`cell_edge()`, `lattice_index()`: `[min, max)`, the upper edge in the last
  bin, so that no episode is counted twice);
- two-dimensional factor: cells, as for the object position and the lighting;
- categorical factor: one bin per required category, as for the color
  references.

Shapes go in `axes/<axis>/shapes.ttl`; `verify` runs them together with the
core shapes (`shapes/dataset.ttl`) and every finding names the verifier whose
shape produced it. They check:

- required specification fields and datatypes;
- exactly one valid episode-level value where the spec requires one
  (`cp:LightingDatasetShape` in `axes/image_augmentation/shapes.ttl` is the
  example);
- values outside the required domain (a `Warning`).

Counting is done in Python. `check(graph, dataset)` returns an `AxisCheck`
(a subclass with `lines()` for the table to print), its findings (one
`Violation` per bin below the minimum), and the derived triples to store,
such as one cell node with its bounds and `cp:episodeCount`.
`check_lighting()` in `axes/image_augmentation/coverage.py` is the example.
`dataset_errors(graph, dataset)` reports, as an `Error`, a spec that names
something the dataset has no column for.

Conventions for what a verifier reports:

- an `Error` when the check cannot run (a missing column, a bad spec);
- a `Violation` when a requirement is not met, so `verify` exits `1`;
- a `Warning` for values outside the required domain; it fails only with
  `--strict`;
- counts per bin in `AxisCheck.lines()`, not only pass or fail;
- malformed and out-of-range episodes named in the findings;
- derived results stored as triples, so that they can be queried;
- the input dataset never modified.

## 10. Register the verifier

**Required**

Append an instance to `VERIFIERS` in
[`registry.py`](../packages/data_verification/src/data_verification/registry.py):

```python
from .axes.camera_viewpoint import CameraViewpointVerifier
from .axes.image_augmentation import ImageAugmentationVerifier
from .axes.object_property import ObjectPropertyVerifier
from .axes.object_pose import ObjectPoseVerifier

VERIFIERS: list[Verifier] = [
    ObjectPoseVerifier(),
    ImageAugmentationVerifier(),
    ObjectPropertyVerifier(),
    CameraViewpointVerifier(),
]
```

No new command is needed: the same commands now cover the factor.

```bash
uv run python scripts/store.py ingest --dataset datasets/${HF_USER}/<repo_name>   # adds the camera height to each episode
cp specs/<task>/camera_viewpoint.ttl my_specs/                                     # replace every "TODO"
uv run python scripts/store.py verify --store stores/<repo_name> \
    --spec my_specs/object_pose.ttl my_specs/camera_viewpoint.ttl
```

| Argument | Meaning |
|---|---|
| `ingest --dataset <path>` | The dataset directory; the new verifier adds its column to each episode |
| `--store <path>` | The store to verify |
| `--spec <files>` | One spec file per axis to check; the options of `verify` are in [Data Verification §4](data_verification.md#4-run-the-check-and-read-the-report) |

`verify` refuses a spec that declares no axis class of a registered verifier.

If two factors later share substantial behavior, common code is refactored
after both are tested, as `grid.py` was for the object position and the
lighting. Different factors often have different units, aggregations, and
boundary semantics.

## 11. Compare the factor with a rollout

**Required**

After a rollout is ingested, `verify` checks whether the dataset covers the
conditions the rollout actually ran
([Data Verification §5](data_verification.md#5-compare-the-dataset-with-a-rollout)).
The comparison starts from the `cp:Rollout`: it has the dataset the policy was
trained on (`cp:hasDataset`), whose axes give the requirements, and its runs
(`cp:hasRun`), each with a `cp:status` and the values it started with. A run
counts as *in distribution* only when **every** axis of the spec covers it.
No eval file is read: the runs' own values show what the rollout used.

1. **Rollout log.** `start_factors()` in
   [`rollout_log.py`](../packages/simulator/src/simulator/utils/rollout_log.py)
   builds each episode's `factors` right after the reset. The new factor is
   logged under its own key, next to `object_position` and `lighting`, with a
   `*_FACTOR` constant:

   ```json
   {"type": "episode", "episode": 3, "status": "failed",
    "factors": {"object_position": {"blue_cup": [8.91, 6.47, 1.0]},
                "lighting": {"rect_light": [6120.0, 4100.0]},
                "camera_height": {"front": [1.16]}},
    "instruction": "pick up the blue cup and place it on the pink cup."}
   ```

   The reader in
   [`data_verification/rollout_log.py`](../packages/data_verification/src/data_verification/rollout_log.py)
   gets the same constant; an entry in `_FACTOR_VALUES` makes it check the
   shape and finiteness of every value, and a property on `RolloutRun` (like
   `lighting`) returns the values.
2. **`ingest_rollout(log, graph, rollout)`** adds the value to each `cp:Run`
   (`<rollout>/run/<index>`), and `axes/<axis>/rollout_shapes.ttl` requires it
   on every finished run in a shape targeting `cp:Rollout`
   (`cp:LightingRolloutShape` is the example). A violation there stops the
   comparison.
3. **`compare(graph, rollout, dataset, comparison)`** places each finished
   run into a bin of the spec's own binning (a value outside the domain falls
   into a bin of the same width outside it, which `lattice_index()` computes),
   marks the run covered when its bin is inside the domain and has at least
   the minimum number of dataset episodes, and returns an `AxisComparison`
   with `run_covered` (finished run index to covered). Placements are stored
   as triples hanging off `comparison`; suggestions for the next round go in
   `hint_lines()` and are stored as `Info` findings. `compare_lighting()` in
   `axes/image_augmentation/coverage.py` is the example. A verifier without
   `compare()` does not decide which runs are in distribution; `verify` says
   so in an `Info` finding.

Runs have no minimum: they only record where the rollout ran and whether it
succeeded. The minimum belongs to the dataset's bins.

## 12. Update the documentation

**Required**

- The vocabulary trees in
  [Data Verification §7](data_verification.md#7-how-it-works-rdf-owl-shacl)
  show every class and property the store holds. The new axis, requirement
  class, recorded facts, and derived nodes are added to them, with the
  command that writes each one.
- The axis table in
  [Data Verification §1](data_verification.md#1-the-three-generalization-axes)
  gets a row, and the new axis gets its own page modelled on
  [Image Augmentations: Lighting](image_augmentation.md): the factor, its
  columns, the spec fields and how to choose them, the report, and the rollout
  comparison.
- [Data Collection §5](data_collection.md#5-the-recorder-and-the-dataset)
  lists the new column, and [Rollout §3](rollout.md#3-run-rollout) the new
  log key.

## 13. Test the complete path

**Required**

Tests use synthetic LeRobot v3 fixtures and the store functions
(`ingest_dataset`, `ingest_rollout`, `verify_store`) as in
[`test_image_augmentation_and_object_property.py`](../tests/test_image_augmentation_and_object_property.py);
`test_every_axis_must_cover_a_run` in
[`test_rollout_comparison.py`](../tests/test_rollout_comparison.py) registers a
second verifier for a test. New test files are added to the `test` target of
the `Makefile`. They cover at least:

1. Every bin meets the minimum.
2. One bin is empty.
3. Values lie exactly on boundaries.
4. A value is outside the required domain.
5. The feature declaration or Parquet column is missing.
6. The shape or datatype is wrong.
7. A value is null, `NaN`, or infinite.
8. Episodes and frames are out of order.
9. The Turtle spec contains `TODO`, missing, or invalid values.
10. `verify` exit codes and JSON output are correct.
11. The stored results answer a SPARQL query about the factor.
12. A rollout log with and without the factor, and a run outside the domain.

One small simulator collection is also run. Synthetic data proves the checker,
but not that the recorder reads the intended simulator quantity.

## 14. Definition of done

**Required**

The extension is complete when it is possible to:

- understand the factor, unit, sampling, and aggregation;
- collect it in a LeRobot v3 dataset;
- inspect the raw Parquet values;
- copy the shipped spec from `specs/<task>/` and fill it in;
- run the same `ingest` and `verify` commands;
- see actionable coverage gaps, and query them from the store;
- compare the dataset with a rollout on the factor;
- reproduce passing and failing tests;
- train normally without the verification metadata becoming a policy input.

Coverage is evidence, not proof. A gap can support a hypothesis about a policy
failure, but complete coverage does not guarantee policy success.
