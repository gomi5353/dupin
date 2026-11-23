# Object Properties: Referencing Objects by Color

This guide describes the color-reference check, one of the three
generalization axes of the data-verification tool
([Data Verification §1](data_verification.md#1-the-three-generalization-axes)).
It explains what changes between episodes (only the words of the task text),
where the task sets those words, what the dataset records, how to fill in the
specification, and how to read the report for the dataset and for a rollout.
The workflow, the store, the `verify` command and the queries are common to
every axis and are described in [Data Verification](data_verification.md).

Every section starts with one of two labels: **Required** (something to do or
follow) or **Background** (explanation without work items).

## 1. What the check does and why

**Background**

The axis follows STAR-Gen (Gao et al., *A Taxonomy for Evaluating Generalist
Robot Manipulation Policies*, arXiv:2503.01238). STAR-Gen's axis **Object
Properties (S-PROP)** is a semantic change: the instruction refers to a
task-relevant object by one of its physical properties instead of its name.
The paper's example turns "put carrot on plate" into "put the orange object
on the plate". The repository uses one such factor, **referencing objects
based on color**:

| STAR-Gen axis | Factor | Axis class / verifier / spec file | Requirement class | Data |
|---|---|---|---|---|
| Object Properties, S-PROP (Semantic) | referencing objects based on color | `cp:ObjectPropertyAxis` / `object_property` / `object_property.ttl` | `cp:ColorReferenceRequirement` | Each episode's task text; `object_color.<obj>` |

**Only the instruction changes.** The objects, the scene and the required
motion stay exactly the same: in the cutlery task the fork is pink and goes to
the left whether the text says "fork" or "pink object". The objects are
**never recolored**. Changing an object's color as well would be a different
STAR-Gen axis, VS-PROP (Visual + Semantic), which the repository does not
use.

Why it matters: a policy that reads the task text learns which words go with
which object. If training only ever says "fork", the words "pink object" are
new at evaluation, although the scene is the same. The evaluation files use
the color form in about half of the episodes
([section 2](#2-where-the-wording-is-set)); the recording defaults never do.

The check answers two questions:

1. **Coverage**: how many dataset episodes refer to each object by its name,
   and how many by its color, measured against a minimum set in the spec? A
   reference with enough episodes is **seen**; one with fewer is
   **unseen**.
2. **Unchanged object**: does each object look the same in every episode? It
   should, since S-PROP changes only the words.

## 2. Where the wording is set

**Required**

The task configuration
(`packages/simulator/src/simulator/tasks/<task>/<task>_env_cfg.py`) builds
each episode's task text from three settings:

```python
object_references: dict[str, dict[str, str]] = {
    "fork": {"name": "fork", "color": "pink object"},
    "knife": {"name": "knife", "color": "orange object"},
}
reference_weights: dict[str, float] = {"name": 1.0, "color": 0.0}
task_description_template: str = "place the {fork} on the left and the {knife} on the right of the plate."
```

| Setting | Meaning |
|---|---|
| `object_references` | Per object (its scene name), one phrase per **form**: `"name"` refers to it by name, `"color"` by its color only. |
| `reference_weights` | How often each form is used. Recording default `{"name": 1.0, "color": 0.0}` (name only); evaluation files `{"name": 0.5, "color": 0.5}`. |
| `task_description_template` | The task text, with one `{object}` field per object. |

At every reset, the event `choose_object_reference_forms(self.reference_weights)`
draws **one form per episode, used for every object**, from a random
generator seeded with the environment's seed. `task_text(env, env_id)` then
fills the template with that form's phrases. The cutlery task therefore has
two possible texts:

- name form: "place the fork on the left and the knife on the right of the plate."
- color form: "place the pink object on the left and the orange object on the right of the plate."

Forms are never mixed within one episode, so in one dataset every object of
the task has the same number of name-form and color-form episodes.

To cover the color form, raise its weight in the task configuration, for
example `{"name": 0.5, "color": 0.5}`, and record again. A form with a weight
above 0 needs a phrase for every object, and the weights must be ≥ 0 with a
positive sum; otherwise the environment refuses to start with a `ValueError`
naming the problem. The phrases themselves are best left unchanged: the
evaluation files use the same phrases, and the check compares the dataset with
exactly those words.

### 2.1 The three tasks

| Task | Template | Name form | Color form | How strongly S-PROP is tested |
|---|---|---|---|---|
| Cup stacking | `pick up the {blue_cup} and place it on the {pink_cup}.` | blue cup / pink cup | blue object / pink object | **Weak.** "blue cup" already contains the color word, so the forms differ only in "cup" vs "object". |
| Cutlery arrangement | `place the {fork} on the left and the {knife} on the right of the plate.` | fork / knife | pink object / orange object | **Strongest.** The two forms share no word; the policy has to link "pink" to the fork and "orange" to the knife. |
| Toy blocks collection | `pick up the {green_block}, the {blue_block} and the {red_block} and place them into the storage box.` | bridge / cylinder / triangle | green object / blue object / red object | **Weak.** All three blocks go into the box, so the reference does not change what the robot has to do. |

## 3. What the dataset records

**Background**

The repository's `LeRobotRecorderManager`
([Data Collection §5](data_collection.md#5-the-recorder-and-the-dataset))
records two things for this axis.

**The episode's own task text.** Every frame's `task` is `task_text(env, 0)`,
the text of that episode in its form. LeRobot stores each distinct text once
in `meta/tasks.parquet` and gives every frame a `task_index` into it. `ingest`
reads the `task_index` of each episode's first frame and stores the text as
the episode's `cp:taskDescription`. This text is what the coverage check
reads.

**The object colors**, one column per object in `object_references`:

| Key | dtype | Shape | Content |
|---|---|---|---|
| `object_color.<name>` | float32 | `(3,)` | `[r, g, b]`, the diffuse color of the object's material (OmniPBR `inputs:diffuse_color_constant`; values above 1 are possible) |

The color column is **not** a factor that varies. It exists only to check
that the object never changes ([section 1](#1-what-the-check-does-and-why)).
The check uses the first recorded frame of each episode. Only environment 0
is recorded (a LeIsaac limitation).

A quick test after recording a few episodes (general environment):

```bash
uv run python -c "import json; f=json.load(open('datasets/${HF_USER}/<repo_name>/meta/info.json'))['features']; print([k for k in f if k.startswith('object_color.')])"
uv run python -c "import pyarrow.parquet as pq; print(pq.read_table('datasets/${HF_USER}/<repo_name>/meta/tasks.parquet').to_pandas())"
```

Both commands read from the dataset folder `datasets/${HF_USER}/<repo_name>`.
The first prints one key per object of `object_references`; the second
lists the distinct task texts. With the recording default, only the name-form
text appears.

## 4. Write the specification

**Required**

Copy `specs/<task>/object_property.ttl` into `my_specs/` and fill in every
`TODO` ([Data Verification §3](data_verification.md#3-specs)). The file has
one block per object of the task. The cutlery block for the fork:

```turtle
spec:object_property_axis a cp:ObjectPropertyAxis .

spec:object_property_axis cp:hasRequirement spec:fork_reference_requirement .

spec:fork_reference_requirement a cp:ColorReferenceRequirement ;
    cp:hasTaskObject spec:fork_reference_object ;
    cp:hasReference spec:fork_by_name , spec:fork_by_color ;
    cp:minEpisodesPerReference "TODO" .

spec:fork_reference_object a cp:TaskObject ;
    cp:name "fork" .

spec:fork_by_name a cp:NameReference ;
    cp:phrase "TODO" .

spec:fork_by_color a cp:ColorReference ;
    cp:phrase "TODO" .
```

| Field | Meaning | Type |
|---|---|---|
| `cp:name` | Object to check; matches the `object_references` key and `object_color.<name>` | Text in quotes (already filled) |
| `cp:phrase` of the `cp:NameReference` | The words the task text uses for the object's name form: `object_references["fork"]["name"]` | Text in quotes (`"fork"`) |
| `cp:phrase` of the `cp:ColorReference` | The words for its color form: `object_references["fork"]["color"]` | Text in quotes (`"pink object"`) |
| `cp:minEpisodesPerReference` | Episodes needed for **each** reference of the object, name and color alike, to count as seen | Whole number ≥ 1, no quotes (`10`) |

**Phrases.** Copy them from `object_references` of the eval file
(`eval/<task>_eval.py`); they are the same as in the task configuration. The
phrases describe the wording the policy will meet, so they come from
evaluation, not from a free choice. A phrase matches when the text
contains its words as whole words, case-insensitive ("fork" does not match
"forklift"). When several phrases of one object match, the longest wins.

**Minimum.** It can be chosen freely. With the default form choice, the
name and color episodes come from one random draw per episode, so with
weights 0.5/0.5 about half of the episodes use each form. A minimum of 10 per
reference then needs about 20 episodes. Raising the color weight above 0.5
fills the color reference faster.

**Own minimum per reference (optional).** To set a different number for
one reference, add `cp:minEpisodesPerReference` to that reference; it
replaces the requirement's value for it:

```turtle
spec:fork_by_color a cp:ColorReference ;
    cp:phrase "pink object" ;
    cp:minEpisodesPerReference 15 .
```

Deleting a block stops checking that object. Since one form is used for every
object of an episode, one block per task already checks the coverage of each
form; the other blocks add each object's own color check.

## 5. Read the report

**Required**

Run `verify` with the filled-in spec
([Data Verification §4](data_verification.md#4-run-the-check-and-read-the-report)):

```bash
uv run python scripts/store.py verify --store stores/<repo_name> \
    --spec my_specs/object_property.ttl
```

| Argument | Meaning |
|---|---|
| `--store` | Store folder of the dataset, `stores/<repo_name>` |
| `--spec` | The filled spec to check |

The command runs in the general environment.

Example output, for a spec with only the fork block
(`minEpisodesPerReference 5`) and 10 episodes recorded with the default
`reference_weights`:

```
Store   : stores/<repo_name>
Spec    : object_property  (/…/my_specs/object_property.ttl)
Episodes: 10

[fork] 1/2 references seen (episodes >= needed); color (1.87, 0.4, 1.88)
  "fork" (name)                   10 / 5   
  "pink object" (color)            0 / 5     unseen

VIOLATION 'fork' reference "pink object" (color) is used by 0 episode(s); needs at least 5  [spec/object_property/fork_by_color]

RESULT: coverage spec NOT met
```

How to read it:

- **Header**: references with at least their minimum, and the object's usual
  color (the most common one across episodes).
- **One line per reference**: its phrase and form, the episodes whose text
  uses it, the minimum, and `unseen` when below the minimum.
- **VIOLATION**: the color form is missing. The default weights never use it;
  raise its weight and record again.

A second example: 13 episodes, 6 by name, 6 by color, 1 with a text that
uses neither phrase, and one color-form episode in which the fork's color is
different:

```
[fork] 2/2 references seen (episodes >= needed); color (1.87, 0.4, 1.88)
  "fork" (name)                    6 / 5   
  "pink object" (color)            6 / 5   
  (unrecognized)                   1  not counted

VIOLATION 'fork' does not look the same in every episode (color differs from (1.87, 0.4, 1.88)); Object Properties (S-PROP) changes only the task text, never the object  [episode/12]
WARNING   'fork': the task text uses none of the spec's phrases (fork, pink object); these episodes do not count  [episode/11]
```

- **(unrecognized)**: episodes whose text uses none of the object's phrases.
  They count toward no reference. Usually the task configuration's phrases
  were changed, or the spec has a typo.
- **does not look the same**: an episode's color differs from the usual
  color by more than 0.001 in a channel. Something recolored the object;
  remove that change, because this axis keeps the scene the same.

The reference counts are stored on the references (`cp:episodeCount`), so
they can be queried
([Data Verification §6](data_verification.md#6-query-the-store)):

```bash
uv run python scripts/store.py query --store stores/<repo_name> --query-text \
  'SELECT ?phrase ?n WHERE { ?ref cp:phrase ?phrase ; cp:episodeCount ?n }'
```

| Argument | Meaning |
|---|---|
| `--store` | Store folder of the dataset, `stores/<repo_name>` |
| `--query-text` | A SPARQL query; this one lists each reference's phrase and episode count |

## 6. Compare with a rollout

**Required**

The rollout comparison works the same for every axis
([Data Verification §5](data_verification.md#5-compare-the-dataset-with-a-rollout)).
For this axis, `scripts/rollout.py` logs each run's `instruction`, the text
the policy was given, and the comparison sorts the runs by the reference
their instruction uses. With the eval files, each run's text is drawn with
weights 0.5/0.5, so about half of the runs use each form.

Each object's runs fall into three groups:

| Group | Which runs | Covered? | Why it exists |
|---|---|---|---|
| **seen** | The instruction uses a reference with at least its minimum of dataset episodes | Yes | The policy was trained on this wording. |
| **unseen** | The instruction uses a reference with fewer episodes than the minimum (none included) | No | The wording is in the spec but too rare in the dataset: the S-PROP gap. |
| **(unrecognized)** | The instruction uses none of the spec's phrases | No | The spec does not describe the evaluation's wording. With the shipped eval files and phrases copied from them, this group is empty. |

**In distribution** is defined in
[Data Verification §5](data_verification.md#5-compare-the-dataset-with-a-rollout);
with this spec alone, a run counts when its reference is seen.

Example output, for the first dataset of [section 5](#5-read-the-report)
(10 episodes, all by name) and a rollout of 13 runs: 6 by name, 6 by color,
and 1 whose instruction says "utensil" (not in the eval file, added to show
the third group). The log is synthetic, made up for illustration, not from a
trained policy:

```
Store   : stores/<repo_name>
Spec    : object_property
Rollout : 20261120-113000  (eval/cutlery_arrangement_eval.py, seed 42)
Runs    : 13 finished, 6 succeeded

[fork] per reference: dataset episodes / needed, seen or unseen, rollout successes / runs
  "fork" (name)                   10 / 5    seen    5/6
  "pink object" (color)            0 / 5    unseen  1/6
  (unrecognized)                            -       0/1
  covered: 6/13 runs (46%)
  success: 5/6 seen vs 1/7 unseen or unrecognized references

In distribution (covered on every axis: object_property): 6/13 runs (46%)
  success 5/6 in distribution vs 1/7 outside

Object Properties: what to record next:
  fork: record more episodes whose task text says "pink object" (color) (has 0, needs 5; raise its form in reference_weights)
  fork: 1 run(s) referred to it in a way your spec does not list; add that phrase to the spec

WARNING   'fork': 1 run(s) referred to it in a way the spec does not list (1 failed)  [fork]
WARNING   'fork': 6 run(s) used an unseen reference, with fewer dataset episodes than needed ("pink object" (color): 0/5)  [fork]

RESULT: the dataset does NOT cover every rollout run; see above
```

What the report shows:

- **Coverage of the evaluation's wording.** `covered` is the share of runs
  whose wording the dataset has enough of. Here only the name-form runs are
  covered: 6 of 13.
- **The S-PROP gap.** `success: 5/6 seen vs 1/7 unseen` compares the success
  rate under trained wording with the rate under new wording. The scene was
  the same in both groups, so the difference comes from the words alone. A
  policy that does not use the task text behaves the same under both forms;
  its gap is then close to zero.
- **What to change.**
  - An unseen reference: raise that form's weight in `reference_weights` of
    the task configuration and record more episodes (the first hint).
  - Unrecognized runs: the spec's phrases do not match the eval wording. Copy
    the phrases from `object_references` of the eval file (the second hint).
    Rollout with `--policy_language_instruction` gives every run that one
    text, which also lands here when it uses neither phrase.

The axis-specific error:

| Message | Cause | Fix |
|---|---|---|
| `Finished run N has no instruction, which the Object Properties axis compares (rollout log written by an older rollout.py?)` | The log's episode lines have no `instruction` | Run the rollout again with the current repository |

The runs' texts are stored with the rollout (`cp:taskDescription` of each
`cp:Run`), so the success rate per wording can be queried directly:

```bash
uv run python scripts/store.py query --store stores/<repo_name> --query-text \
  'SELECT ?text (COUNT(?run) AS ?runs) (SUM(?ok) AS ?successes) WHERE { ?run a cp:Run ; cp:taskDescription ?text ; cp:status ?s . BIND (IF(?s = "succeeded", 1, 0) AS ?ok) } GROUP BY ?text'
```

## 7. Findings

**Background**

Every line of the report is stored as a `cp:Finding`. Severities:
`Error` (the check could not run), `Violation` (a requirement is not met),
`Warning` (allowed, but worth knowing), `Info` (a suggested next step). The
common messages (store, changed dataset, Turtle syntax, ...) are in
[Data Verification §8](data_verification.md#8-troubleshooting).

| Severity | Message | Cause | Fix |
|---|---|---|---|
| Error | `Dataset has no 'object_color.<name>' column, which the spec requires to check that the object never changes (...)` | Dataset recorded without `--use_lerobot_recorder`, or the object is not in `object_references`; or a typo in `cp:name` | Record again ([section 3](#3-what-the-dataset-records)); check the name |
| Error | `Spec still has TODO values: ...` | Unfilled phrases or minimum | Fill them in ([section 4](#4-write-the-specification)) |
| Error | `...: minEpisodesPerReference must be a whole number without quotes, got ...` | Quoted number or fraction | Write `10`, not `"10"` |
| Error | `fork: minEpisodesPerReference must be at least 1, got 0` | Zero or negative minimum | Use a positive whole number |
| Error | `fork: no reference (cp:hasReference)` | The `cp:hasReference` line was deleted | Restore it from the shipped spec |
| Error | `Spec declares no cp:ColorReferenceRequirement; nothing to check` | Every block was deleted, but the axis line kept | Restore a block |
| Error | `Episode N: non-finite color for '<name>'` (at `ingest`) | NaN or infinity in the color column (no material with a diffuse color bound) | Inspect the object's material |
| Violation | `'fork' reference "pink object" (color) is used by N episode(s); needs at least M` | Too few episodes with that wording | Raise that form's weight in `reference_weights` and record more |
| Violation | `'fork' does not look the same in every episode (...)` | The object's color changed between episodes | Remove whatever recolors the object |
| Violation | `Episode N has no color for required object 'fork'` | The column is missing for some episodes | Record with the repository recorder |
| Violation | `Two references both use the phrase '...'; give each reference its own phrase` | Name and color phrase are the same (ignoring case) | Copy both phrases from `object_references` |
| Violation | `A reference must be a cp:NameReference or a cp:ColorReference` | A reference's class was changed | Restore it from the shipped spec |
| Violation | `Requirement is not linked from an ObjectPropertyAxis (cp:hasRequirement)` | The `spec:object_property_axis cp:hasRequirement ...` line was deleted | Restore it |
| Warning | `'fork': the task text uses none of the spec's phrases (...); these episodes do not count` | The dataset's wording differs from the spec's phrases | Keep the task configuration's phrases equal to the eval file's; check the spec for typos |
| Warning | `Episode N has no task text; it uses no reference` | The dataset has no `task_index` or `meta/tasks.parquet` | Record with the repository recorder |
| Warning | `Requirement has no cp:ColorReference; it only checks how often the object is referred to by name` | The color reference was deleted | Restore it: the color form is the point of this axis |

## 8. What the check does not cover

**Background**

- **Only two forms, only color.** Other properties (shape, size, material)
  and other wordings (synonyms, paraphrases) are not checked; a text with
  another wording is *unrecognized*.
- **Phrases, not meaning.** The check matches words. It does not know whether
  "pink object" really names the fork; the phrases in `object_references`
  define that.
- **Whether the policy reads the text.** The check counts wordings in the
  data. Whether the policy actually uses the task text shows only in the
  rollout's seen-vs-unseen success.
- **The object's color is checked, not varied.** A recolored object is an
  error here, not coverage (that would be VS-PROP).
