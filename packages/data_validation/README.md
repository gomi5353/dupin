# Dataset validation

This package validates a local LeRobot v3 dataset in three layers:

1. physical files and metadata (`meta/`, `data/`, and `videos/`);
2. RDF generated from the dataset metadata;
3. OWL vocabulary and SHACL constraints.

The default HCIS contract expects `observation.state`, `action`, `task`,
`observation.images.front`, and `observation.images.wrist`. Pass a custom
contract when another task has a different state or action shape.

```bash
uv run python scripts/validate_dataset.py \
  --dataset ./datasets/<dataset-name> \
  --contract packages/data_validation/src/data_validation/ontology/hcis-capstone.ttl \
  --report ./reports/<dataset-name>.json \
  --strict
```

The command exits with status 0 only when the dataset has no validation errors
and, with `--strict`, no warnings.
