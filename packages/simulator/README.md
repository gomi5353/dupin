# simulator (physicalai)

Project-specific simulator layer on top of the upstream `leisaac` package: the
three task configurations, the shared single-arm Franka template, randomization
events, scene assets, the state machines (`simulator.datagen.state_machine`),
the keyboard device (`simulator.devices`), the LeRobot recorder used for
every dataset (`simulator.utils.lerobot_recorder`), which adds the
verification columns (`object_position.*`, `lighting.*`, `object_color.*`, read
from the scene by `simulator.utils.scene_factors`), and the rollout log
(`simulator.utils.rollout_log`).

- Repository layout and setup: [Getting Started](../../docs/getting_started.md#37-repository-layout)
- Data-collection methods, task settings, and the recorder: [Data Collection](../../docs/data_collection.md)
- Writing and registering a new task: [Adding a New Task](../../docs/adding_new_task.md)
- Recording a new verification factor: [Adding a New Dataset Coverage Factor](../../docs/adding_coverage_factor.md#4-record-the-factor-in-lerobot-v3)
