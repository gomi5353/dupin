# Example SPARQL queries

Run any of them over a store:

```bash
uv run python scripts/store.py query --store stores/<dataset_repo> --query-file queries/<file>.rq
```

The prefixes `cp:`, `rdf:`, `rdfs:`, `owl:` and `xsd:` are predefined; `ds:` is
the store's dataset IRI followed by `/`. Add `--format csv` or `--format json`
to save a table, and `--with-ontology` to also query the classes and labels.

| File | Question |
|---|---|
| `episodes.rq` | Where did each object start in each recorded episode? (raw facts, no spec needed) |
| `cell_counts.rq` | How many dataset episodes does each cell of each spec have? (after `verify`) |
| `uncovered_cells.rq` | Which cells have fewer episodes than the spec requires? (after `verify`) |
| `run_outcomes_by_cell.rq` | How many rollout runs started in each cell, and how many succeeded? (after `verify`) |
| `failed_runs_not_covered.rq` | Which failed runs started where the dataset does not cover? (after `verify`) |
| `findings.rq` | Every stored finding, by spec, rollout and severity (after `verify`) |

Compare numbers with `FILTER`, not by writing them in a triple pattern:
`?e cp:index ?i . FILTER (?i = 2)` finds episode 2, but `?e cp:index 2` finds
nothing, because indices are stored as `xsd:nonNegativeInteger` and `2` is an
`xsd:integer`.

Keep your own queries in `stores/<dataset_repo>/queries/`.
