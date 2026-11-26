# Example SPARQL queries

Run any of them over a store:

```bash
uv run python scripts/store.py query --store stores/<repo_name> --query-file queries/<file>.rq
```

The prefixes `cp:`, `rdf:`, `rdfs:`, `owl:` and `xsd:` are predefined; `ds:` is
the store's dataset IRI followed by `/`. Add `--format csv` or `--format json`
to save a table, and `--with-ontology` to also query the classes and labels.

| File | Axis | Question |
|---|---|---|
| `episodes.rq` | Object Poses | Where did each object start in each recorded episode? (raw facts, no spec needed) |
| `cell_counts.rq` | Object Poses | How many dataset episodes does each cell of each object-position requirement have? (after `verify`) |
| `uncovered_cells.rq` | Object Poses | Which cells have fewer episodes than the requirement asks for? (after `verify`) |
| `run_outcomes_by_cell.rq` | Object Poses | How many rollout runs started in each cell, and how many succeeded? (after `compare`) |
| `failed_runs_not_covered.rq` | Object Poses | Which failed runs started where the dataset does not cover? (after `compare`) |
| `findings.rq` | all | Every stored finding, by spec, rollout and severity (after `verify` or `compare`) |

The vocabulary of every axis, for writing queries about lighting cells or
color references, is described in
[Data Verification §7](../docs/data_verification.md#7-how-it-works-rdf-owl-shacl)
and [§6](../docs/data_verification.md#6-query-the-store).

Compare numbers with `FILTER`, not by writing them in a triple pattern:
`?e cp:index ?i . FILTER (?i = 2)` finds episode 2, but `?e cp:index 2` finds
nothing, because indices are stored as `xsd:nonNegativeInteger` and `2` is an
`xsd:integer`.

Keep your own queries in `stores/<repo_name>/queries/`.
