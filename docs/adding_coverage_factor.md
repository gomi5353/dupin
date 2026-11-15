# Adding a New Dataset Coverage Factor

This guide describes how to extend the dataset verifier for other factors that a group wants to investigate. 
Examples are lighting, camera viewpoint, object orientation, appearance, background, and motion speed. It
extends step 6 of [Getting Started](getting_started.md).

It assumes familiarity with the complete worked example in
[Object Position Coverage](object_position_coverage.md).

Every section starts with one of two labels: **Required** (something that an
extension must do) or **Background** (explanation without work items).

## 1. Decide between a rule and a coverage factor

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
([Object Position Coverage §7](object_position_coverage.md#7-how-it-works-rdf-owl-shacl)).
The rest of this guide is about coverage factors.

## 2. Write a factor contract first

**Required**

The factor is defined in writing before any code:

```text
Factor: scene light intensity
Dataset key: factor.light_intensity
dtype and shape: float32, (1,)
unit: lux
sampling: every recorded frame
episode value: first recorded frame
required domain: 300 to 900 lux
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
use the first frame. Motion factors may need a minimum, maximum, mean, or
trajectory summary.

Verification-only metadata uses a key such as `factor.<name>`. A key starting
with `observation.` is used only when the factor is intentionally a policy
input, because LeRobot exposes `observation.*` features to the policy.

## 3. The extension path

**Background**

A coverage factor is a new **generalization axis** with its own **verifier**.
The store, the commands (`ingest`, `attach`, `verify`, `query`) and the core
vocabulary stay as they are; the factor plugs into them:

```text
collection script ── records factor.<name> ──▶ LeRobot v3 dataset ─┐
rollout.py ── logs factors.<name> per run ──▶ rollout JSONL log ───┤
                                                                   │ ingest  (+ the verifier's ingest steps)
group's spec (.ttl, from the verifier's template) ── attach ───────┤
                                                                   ▼
                                   stores/<dataset_repo>/  (RDF: facts, specs, results)
                                                                   │ verify  (core SHACL + every verifier's
                                                                   │          SHACL rules and Python steps)
                                                                   ▼
                                   findings/  ──▶ report, explore, SPARQL query
```

A verifier lives in its own package and is registered in one list:

```text
packages/data_verification/src/data_verification/
├── verifier.py                  the Verifier base class (every step optional)
├── registry.py                  VERIFIERS = [ObjectPositionVerifier(), LightingVerifier(), ...]
└── axes/
    ├── object_position/         the complete worked example
    └── lighting/                the new factor
        ├── __init__.py          class LightingVerifier(Verifier)
        ├── ontology.ttl         axis and requirement classes, properties   (section 6)
        ├── template.ttl         spec skeleton for `init --axis lighting`   (section 8)
        ├── shapes.ttl           SHACL rules on the dataset                 (section 9)
        └── rollout_shapes.ttl   SHACL rules on a rollout                   (section 11)
```

| `Verifier` member | Used by | Needed for |
|---|---|---|
| `name`, `axis_class`, `package` | every command | always |
| `ontology.ttl` | `attach`, `verify`, `query --with-ontology` | always: the axis class must subclass `cp:GeneralizationAxis` |
| `ingest_dataset()`, `ingest_rollout()` | `ingest` | adding the factor's values to episodes and runs (section 7) |
| `template.ttl`, `render_spec()` | `init` | a spec skeleton |
| `spec_errors()`, `derive_spec()` | `attach` | refusing unfilled specs; generating bins (section 9) |
| `shapes.ttl`, `dataset_errors()`, `check()` | `verify` (dataset) | the coverage rule; counts and tables |
| `rollout_shapes.ttl`, `compare()` | `verify` (rollouts) | the rollout comparison (section 11) |

A verifier that only needs SHACL rules leaves every Python step at its
default. OWL describes what the facts mean, while SHACL validates them.
Neither can recover a factor that was never recorded in the dataset.

## 4. Record the factor in LeRobot v3

**Required**

The repository's `LeRobotRecorderManager` in
[`lerobot_recorder.py`](../packages/simulator/src/simulator/utils/lerobot_recorder.py)
is the base for every verification factor: it already adds the
`object_position.*` columns, and a new factor is appended next to them
([How to Record Episodes §5.2](how_to_record.md#52-the-repositorys-recorder-and-object-positions)).
Recording continues with that class, so the object positions stay in the dataset
next to the new factor.

1. Declare the feature in the recorder's feature dictionary
   (`build_with_factors` in `lerobot_recorder.py`):

   ```python
   features["factor.light_intensity"] = {
       "dtype": "float32",
       "shape": (1,),
       "names": ["lux"],
   }
   ```

2. Read the value from the environment.
3. Add it in `build_lerobot_frame()` in
   [`single_arm_franka_cfg.py`](../packages/simulator/src/simulator/tasks/template/single_arm_franka_cfg.py),
   next to the object positions, at the same simulation step as the other frame
   values:

   ```python
   frame["factor.light_intensity"] = np.asarray([light_lux], dtype=np.float32)
   ```

One short episode is recorded and its raw output inspected before any RDF work:

```bash
uv run python -c "import json; print(json.load(open('datasets/${HF_USER}/<dataset_repo>/meta/info.json'))['features'].get('factor.light_intensity'))"

uv run python - <<'PY'
import pyarrow.parquet as pq

table = pq.read_table(
    "datasets/<HF_USER>/<dataset_repo>/data/chunk-000/file-000.parquet",
    columns=["episode_index", "frame_index", "factor.light_intensity"],
)
print(table.slice(0, 5))
PY
```

This separates recording failures from verifier failures.

## 5. Read and aggregate the column

**Required**

[`dataset.py`](../packages/data_verification/src/data_verification/dataset.py)
is the example. The extension adds a reader for the factor, inside its axis
package, called from the verifier's `ingest_dataset()` (section 7). The reader:

1. Verifies the feature declaration in `meta/info.json`.
2. Verifies that the column exists in every Parquet shard.
3. Groups rows by `episode_index`.
4. Applies the aggregation from the contract.
5. Reports missing, null, `NaN`, or infinite values with episode numbers.

Parquet row order is not guaranteed. The object-position reader sorts by
`episode_index` and `frame_index` before selecting the initial value.

## 6. Describe the factor with RDF and OWL

**Required**

The axis's vocabulary goes in `axes/<axis>/ontology.ttl`. A new factor is a
new generalization axis: its axis class subclasses `cp:GeneralizationAxis` and
its requirement class subclasses `cp:AxisRequirement`, so it reuses
`cp:hasRequirement` and `cp:hasGeneralizationAxis` from `ontology/core.ttl`:
`attach` links the dataset to every axis of the specification. Episodes (with
their `cp:index`), runs, the dataset, specs and findings are shared with the
object-position axis. Each class is described by what its members have
(`rdfs:subClassOf` an `owl:Restriction`), as in
`axes/object_position/ontology.ttl`. A minimal vocabulary:

```turtle
@prefix cp:   <http://taica.course/cp/ontology#> .
@prefix owl:  <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd:  <http://www.w3.org/2001/XMLSchema#> .

cp:LightingAxis a owl:Class ;
    rdfs:subClassOf cp:GeneralizationAxis ,
        [ a owl:Restriction ; owl:onProperty cp:hasRequirement ; owl:allValuesFrom cp:LightingAxisRequirement ] .
cp:LightingAxisRequirement a owl:Class ;
    rdfs:subClassOf cp:AxisRequirement ,
        [ a owl:Restriction ; owl:onProperty cp:minLux ; owl:qualifiedCardinality "1"^^xsd:nonNegativeInteger ; owl:onDataRange xsd:decimal ] ,
        [ a owl:Restriction ; owl:onProperty cp:maxLux ; owl:qualifiedCardinality "1"^^xsd:nonNegativeInteger ; owl:onDataRange xsd:decimal ] ,
        [ a owl:Restriction ; owl:onProperty cp:numberOfBins ; owl:qualifiedCardinality "1"^^xsd:nonNegativeInteger ; owl:onDataRange xsd:positiveInteger ] ,
        [ a owl:Restriction ; owl:onProperty cp:minEpisodesPerBin ; owl:qualifiedCardinality "1"^^xsd:nonNegativeInteger ; owl:onDataRange xsd:positiveInteger ] .

# One value per episode (dataset) and per finished run (rollout).
cp:lightIntensityLux a owl:DatatypeProperty ; rdfs:range xsd:decimal .
cp:minLux a owl:DatatypeProperty ; rdfs:range xsd:decimal .
cp:maxLux a owl:DatatypeProperty ; rdfs:range xsd:decimal .
cp:numberOfBins a owl:DatatypeProperty ; rdfs:range xsd:positiveInteger .
cp:minEpisodesPerBin a owl:DatatypeProperty ; rdfs:range xsd:positiveInteger .
```

OWL defines the concepts and relationships; executable constraints belong in
SHACL.

## 7. Add the values to the store at ingest

**Required**

`ingest --dataset` writes the dataset and its episodes, then calls
`ingest_dataset(root, graph, dataset)` of every registered verifier. The
verifier reads its column (section 5) and adds the value to each episode,
whose IRI is `<dataset>/episode/<episode_index>`:

```python
class LightingVerifier(Verifier):
    name = "lighting"
    axis_class = CP.LightingAxis
    package = __name__

    def ingest_dataset(self, root, graph, dataset):
        values, errors = read_light_intensity(root)          # section 5: {episode_index: lux}
        for episode, lux in values.items():
            graph.add((URIRef(f"{dataset}/episode/{episode}"), CP.lightIntensityLux, decimal_literal(lux)))
        return errors                                          # non-empty: ingest stops
```

The result in `stores/<dataset_repo>/dataset.ttl`:

```turtle
ds:episode/7 a cp:Episode ;
    cp:index 7 ;
    cp:lightIntensityLux 612.0 .
```

`ingest_rollout(log, graph, rollout)` does the same for each run of a rollout
log (section 11). Extraction stays in Python and validation in RDF/SHACL, so
the store can be inspected and queried without starting Isaac Sim. Ingest only
records facts: no counting, no judgment.

## 8. Create a requirement template

**Required**

The template is `axes/<axis>/template.ttl`; `init --axis lighting` writes a
copy (override `render_spec()` to fill in more, as the object-position
verifier does):

```turtle
@prefix cp:   <http://taica.course/cp/ontology#> .
@prefix spec: <http://taica.course/cp/spec/lighting#> .

spec:lighting_axis a cp:LightingAxis ;
    cp:hasRequirement spec:lighting .

spec:lighting a cp:LightingAxisRequirement ;
    cp:minLux 300.0 ;
    cp:maxLux 900.0 ;
    cp:numberOfBins 3 ;
    cp:minEpisodesPerBin 5 .
```

Requirement values come from the expected evaluation conditions, the robot's
limitations, and the collection budget. Copying the minimum and maximum
already found in the dataset would hide the missing region that the check is
meant to detect.

A store can hold this specification next to the object-position one, or both
axes can be written in one file. `attach` renames every `spec:` node under the
store, so templates never clash. `spec_errors()` returns the problems that
should stop `attach` (a `TODO` left, a quoted number); `axes/object_position/spec.py`
is the example.

## 9. Add bins and SHACL constraints

**Required**

`expand_cells()` in
[`axes/object_position/coverage.py`](../packages/data_verification/src/data_verification/axes/object_position/coverage.py)
is called from the verifier's `derive_spec()` at `attach`. It adds
`numCellW × numCellH` cells to each grid, with their row-major `cp:index` and
their bounds; the SHACL rule computes the bounds again from the area. Only the
episodes of the dataset that has the requirement's axis count
(`?dataset cp:hasGeneralizationAxis ?axis ; cp:hasEpisode ?episode`).
The extension adds an equivalent expansion in its own `derive_spec()`:

- numeric factor: intervals;
- two-dimensional factor: cells;
- categorical factor: one bin per required category.

Shapes go in `axes/<axis>/shapes.ttl`; `verify` runs them together with the
core shapes (`shapes/dataset.ttl`) and every finding names the verifier whose
shape produced it. They check:

- required specification fields and datatypes;
- valid minimum, maximum, bin count, and episode minimum;
- exactly one valid episode-level value where appropriate;
- at least the required number of episodes in every bin;
- values outside the required domain.

[`axes/object_position/shapes.ttl`](../packages/data_verification/src/data_verification/axes/object_position/shapes.ttl)
is the SPARQL counting example. Numeric bins use `[min, max)` and include the
upper edge only in the final bin; otherwise one episode may be counted twice.

Counting in SPARQL can be slow on large datasets. `check()` may count in Python
instead and return an `AxisCheck` with its findings, the lines to print, and
derived triples to store (the object-position verifier stores
`cp:episodeCount` on each cell); `check_coverage()` is the example.

## 10. Register the verifier

**Required**

Append an instance to `VERIFIERS` in
[`registry.py`](../packages/data_verification/src/data_verification/registry.py):

```python
from .axes.lighting import LightingVerifier
from .axes.object_position import ObjectPositionVerifier

VERIFIERS: list[Verifier] = [
    ObjectPositionVerifier(),
    LightingVerifier(),
]
```

No new command is needed: the same commands now cover the factor.

```bash
uv run python scripts/store.py ingest --dataset datasets/${HF_USER}/<dataset_repo>   # adds lux to each episode
uv run python scripts/store.py init --store stores/<dataset_repo> --axis lighting --out specs/lighting.ttl
uv run python scripts/store.py attach --store stores/<dataset_repo> --spec specs/lighting.ttl
uv run python scripts/store.py verify --store stores/<dataset_repo>
```

`attach` refuses a specification whose axis class no registered verifier
checks. Conventions for what a verifier reports:

- a `Violation` when a requirement is not met, so `verify` exits `1`;
- counts per bin in `AxisCheck.lines()`, not only pass or fail;
- malformed and out-of-range episodes named in the findings;
- derived results stored as triples, so that they can be queried;
- the input dataset never modified.

If two factors later share substantial behavior, common binning code is
refactored after both are tested. Different factors often have different
units, aggregations, and boundary semantics.

## 11. Compare the factor with a rollout

**Required**

After a rollout is ingested, `verify` checks whether the dataset covers the
conditions the rollout actually ran
([Object Position Coverage §8](object_position_coverage.md#8-compare-the-dataset-with-a-rollout)).
The comparison starts from the `cp:Rollout`: it has the dataset the policy was
trained on (`cp:hasDataset`), whose axes give the requirements, and its runs
(`cp:hasRun`), each with a `cp:status` and the values it started with. A run
counts as *in distribution* only when it is covered on **every** axis, so a new
factor joins the comparison with two additions, next to the object-position
ones. No eval file is read: the runs' own values show what the rollout used.

1. **Rollout log.** In `scripts/rollout.py`, log each run's sampled value
   under the factor's own key, next to `object_position`:

   ```json
   {"type": "episode", "episode": 3, "status": "failed",
    "factors": {"object_position": {"blue_cup": [8.91, 6.47, 1.0]}, "lighting": {"intensity": 6120.0}}}
   ```

   The format is described in
   [`rollout_log.py`](../packages/simulator/src/simulator/utils/rollout_log.py);
   the verifier's `ingest_rollout()` adds the value to each `cp:Run`
   (`<rollout>/run/<index>`), and `axes/<axis>/rollout_shapes.ttl` requires it
   on every finished run in a shape targeting `cp:Rollout`.
2. **`compare(graph, rollout, dataset, comparison)`.** It places each finished
   run into a bin of the student's own binning (a value outside the student's
   domain falls into a bin of the same width outside it), marks the run covered
   when its bin is inside the domain and has at least the minimum number of
   dataset episodes, and returns an `AxisComparison` with, for each run,
   whether it is covered. Placements are stored as triples hanging off
   `comparison`; suggestions for the next round go in `hint_lines()` and are
   stored as `Info` findings. `compare_object_position()` in
   `axes/object_position/compare.py` is the example.

Runs have no minimum: they only record where the rollout ran and whether it
succeeded. The minimum belongs to the dataset's bins.

## 12. Test the complete path

**Required**

Tests use synthetic LeRobot v3 fixtures and the store functions
(`ingest_dataset`, `attach_spec`, `ingest_rollout`, `verify_store`) as in
[`test_object_position_coverage.py`](../tests/test_object_position_coverage.py);
`test_every_axis_must_cover_a_run` in
[`test_rollout_comparison.py`](../tests/test_rollout_comparison.py) registers a
second verifier for a test. They cover at least:

1. Every bin meets the minimum.
2. One bin is empty.
3. Values lie exactly on boundaries.
4. A value is outside the required domain.
5. The feature declaration or Parquet column is missing.
6. The shape or datatype is wrong.
7. A value is null, `NaN`, or infinite.
8. Episodes and frames are out of order.
9. The Turtle requirement contains missing or invalid values.
10. `verify` exit codes and JSON output are correct.
11. The stored results answer a SPARQL query about the factor.

One small simulator collection is also run. Synthetic data proves the checker,
but not that the recorder reads the intended simulator quantity.

## 13. Definition of done

**Required**

Another student can:

- understand the factor, unit, sampling, and aggregation;
- collect it in a LeRobot v3 dataset;
- inspect the raw Parquet values;
- write a coverage requirement;
- run the same `ingest`, `attach` and `verify` commands;
- see actionable coverage gaps, and query them from the store;
- compare the dataset with a rollout on the factor;
- reproduce passing and failing tests;
- train normally without the verification metadata becoming a policy input.

Coverage is evidence, not proof. A gap can support a hypothesis about a policy
failure, but complete coverage does not guarantee policy success.
