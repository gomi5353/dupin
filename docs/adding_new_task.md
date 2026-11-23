# Adding a New Task

This guide describes how a self-defined manipulation task is written: how a
task configuration is built, how it is registered under a task id, and how it
is tested before recording. It extends step 1 of
[Getting Started](getting_started.md). The three predefined tasks do not need
this guide.

The running example is the cup-stacking task:

- [`template/single_arm_franka_cfg.py`](../packages/simulator/src/simulator/tasks/template/single_arm_franka_cfg.py):
  the shared base configuration that every task subclasses.
- [`cup_stacking/cup_stacking_env_cfg.py`](../packages/simulator/src/simulator/tasks/cup_stacking/cup_stacking_env_cfg.py):
  a complete task built on the template.
- [`cup_stacking/__init__.py`](../packages/simulator/src/simulator/tasks/cup_stacking/__init__.py):
  its registration.
- [`utils/domain_randomization.py`](../packages/simulator/src/simulator/utils/domain_randomization.py):
  the reset events that randomize object positions and lighting, and pick
  how the task text refers to the objects.
- [`specs/cup_stacking/`](../specs/cup_stacking/): the task's shipped specs,
  one per generalization axis.

Every section starts with one of two labels: **Required** (something to do
or follow) or **Background** (explanation without work items).

## 1. How a task is built

**Background**

Isaac Lab describes an environment as a tree of configuration classes
(`ManagerBasedRLEnvCfg`). Each branch configures one *manager*:

| Manager | What it owns |
|---|---|
| `scene` | USD assets, the robot, sensors, lights: everything that exists in the simulation |
| `observations` | What the environment returns at every step (joint states, camera images, ...) |
| `actions` | How the action passed to `step(action)` drives the robot joints and the gripper |
| `events` | Callbacks, mainly the randomization on every reset |
| `rewards` | Per-step rewards (empty here: imitation learning uses none) |
| `terminations` | When an episode ends: `time_out` and `success` |
| `recorders` | What the recorder captures from every step |

LeIsaac ([LightwheelAI/leisaac](https://github.com/LightwheelAI/leisaac)) adds
asset helpers, randomization helpers, and the LeRobot recorder on top of Isaac
Lab. The repository's `simulator` package adds the shared single-arm Franka
template and the three tasks.

The **template** `SingleArmFrankaTaskEnvCfg` already provides:

- the Franka robot, the `front` and `wrist` cameras (640×480 RGB, refreshed
  every 1/30 s), the end-effector frames, and a dome light;
- the observations (`joint_pos`, camera images, ...);
- the action setup `use_joint_position_actions()`: seven absolute joint
  targets plus one binary gripper command, with joint stiffness 80 and damping 4
  and gravity disabled on the robot links;
- `build_lerobot_frame`, which turns every step into a dataset row
  ([Data Collection §5](data_collection.md#5-the-recorder-and-the-dataset));
- `decimation = 1` (one simulation step of 1/60 s per `env.step()`) and a
  `time_out` termination.

A new task therefore only adds its scene objects, its success check, its
randomization, a few settings, and the lists that tell the recorder what to
record for data verification (section 3.3).

## 2. Where the task lives

**Required**

A task is a standalone set of files in its own directory under
`packages/simulator/src/simulator/tasks/`:

```text
packages/simulator/src/simulator/tasks/<my_task>/
    __init__.py             # gym.register(...)
    <my_task>_env_cfg.py    # the task configuration
specs/<my_task>/            # the task's specs, one per axis (section 6)
    object_pose.ttl
    image_augmentation.ttl
    object_property.ttl
```

| Item | Pattern | Example |
|---|---|---|
| Directory | `<task_slug>` (snake_case) | `cup_stacking` |
| Configuration module | `<task_slug>_env_cfg.py` | `cup_stacking_env_cfg.py` |
| Configuration class | `<TaskName>EnvCfg` | `CupStackingEnvCfg` |
| Scene class | `<TaskName>SceneCfg` | `CupStackingSceneCfg` |
| Task id | `HCIS-<TaskName>-SingleArm-v<N>` | `HCIS-CupStacking-SingleArm-v0` |



## 3. Write the task configuration

**Required**

`<my_task>_env_cfg.py` defines three classes. Every class is a
`@configclass`; plain Python classes are not picked up by Isaac Lab.

### 3.1 Scene

The scene class subclasses `SingleArmFrankaTaskSceneCfg`, sets the background
scene, and adds one `RigidObjectCfg` per object:

```python
@configclass
class CupStackingSceneCfg(SingleArmFrankaTaskSceneCfg):
    scene: AssetBaseCfg = KITCHEN_CFG.replace(prim_path="{ENV_REGEX_NS}/Scene")
    blue_cup: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/blue_cup",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(KITCHEN_OBJECTS_ROOT / "BlueCup" / "BlueCup.usd"),
            mass_props=MassPropertiesCfg(mass=0.1),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(8.95, 6.52, 1.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )
    pink_cup: RigidObjectCfg = ...
    # The light baked into the scene USD, wrapped so it can be randomized and recorded.
    rect_light: AssetBaseCfg = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Scene/RectLight")
```

- The attribute name (`blue_cup`) becomes the object's scene name, used as
  `env.scene["blue_cup"]` and as the dataset column `object_position.blue_cup`.
- `{ENV_REGEX_NS}` expands to each environment's own prim path. Per-environment
  objects use it; only assets shared by all environments use `/World/...`.
- Asset paths are built from `simulator.ASSETS_ROOT`
  (`packages/simulator/assets/`), which works from any working directory.
- `init_state.pos` is relative to the environment origin, in metres.
  Quaternions are `(w, x, y, z)`; `(1, 0, 0, 0)` is no rotation.
- A light is a scene entity too. `rect_light` wraps a light that already
  exists in the scene USD (`AssetBaseCfg` with only a `prim_path`), so that
  the reset event can change it and the recorder can read it as
  `lighting.rect_light`. A scene without such a light adds one, for example
  with `spawn=sim_utils.SphereLightCfg(...)` or another Isaac Lab light
  configuration.

### 3.2 Success check

The success check is a function that returns a boolean tensor of shape
`(num_envs,)`, wrapped in a `DoneTerm` named `success`:

```python
def blue_cup_on_top_pink_cup(env, blue_cup_cfg, pink_cup_cfg, xy_tolerance, z_range):
    # A difference of two positions in the same env needs no env_origins.
    offset = env.scene[blue_cup_cfg.name].data.root_pos_w - env.scene[pink_cup_cfg.name].data.root_pos_w
    done = torch.all(torch.abs(offset[:, :2]) <= xy_tolerance, dim=1)
    done &= (offset[:, 2] >= z_range[0]) & (offset[:, 2] <= z_range[1])
    return done


@configclass
class TerminationsCfg(SingleArmFrankaTerminationsCfg):
    success = DoneTerm(
        func=blue_cup_on_top_pink_cup,
        params={
            "blue_cup_cfg": SceneEntityCfg("blue_cup"),
            "pink_cup_cfg": SceneEntityCfg("pink_cup"),
            "xy_tolerance": 0.02,
            "z_range": (0.08, 0.10),
        },
    )
```

- Positions are compared after subtracting `env.scene.env_origins`; raw
  world positions differ per environment.
- Conditions are combined with `&` and `|` on tensors, not with Python `and` and
  `or`.
- `params` is passed to the function as keyword arguments; `SceneEntityCfg("blue_cup")`
  resolves to the object at runtime.
- The recorder keeps an episode only if this term is true when the episode
  ends, and rollout counts an episode as a success when it becomes true. It
  therefore defines what the task means.

### 3.3 Environment configuration

The environment class subclasses `SingleArmFrankaTaskEnvCfg`, plugs in the
scene and terminations, and sets the task-specific values:

```python
@configclass
class CupStackingEnvCfg(SingleArmFrankaTaskEnvCfg):
    scene: CupStackingSceneCfg = CupStackingSceneCfg(env_spacing=15.0)
    observations: SingleArmFrankaObservationsCfg = SingleArmFrankaObservationsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    task_description: str = "pick up the blue cup and place it on the pink cup."  # set from the template
    tracked_object_names: list[str] = ["blue_cup", "pink_cup"]
    recorded_light_names: list[str] = ["rect_light"]
    object_references: dict[str, dict[str, str]] = {
        "blue_cup": {"name": "blue cup", "color": "blue object"},
        "pink_cup": {"name": "pink cup", "color": "pink object"},
    }
    reference_weights: dict[str, float] = {"name": 1.0, "color": 0.0}
    task_description_template: str = "pick up the {blue_cup} and place it on the {pink_cup}."

    def __post_init__(self) -> None:
        super().__post_init__()
        self.episode_length_s = 20                         # evaluation episode length
        self.viewer.eye = (0.8, 0.87, 0.67)                # GUI camera
        self.viewer.lookat = (0.4, -1.3, -0.2)
        self.scene.robot.init_state.pos = (9.0, 6.0, 0.9)  # robot base
        self.scene.robot.init_state.rot = (0.707, 0.0, 0.0, 0.707)
        self.scene.robot.init_state.joint_pos = {...}      # home pose
        self.scene.front.offset.pos = (...)                 # front camera aimed at the workspace
        self.scene.front.offset.rot = (...)
        parse_usd_and_create_subassets(KITCHEN_USD_PATH, self)
        domain_randomization(self, random_options=[...])   # section 3.4
```

| Setting | Purpose |
|---|---|
| `task_description` | Short sentence describing the goal. With a `task_description_template`, the template overwrites it with the text in the most likely reference form; it is used only where one fixed text is needed. |
| `tracked_object_names` | Objects whose start position is randomized. The recorder writes their positions as `object_position.<name>` (axis Object Poses) and refuses to start when the list is empty. |
| `fixed_object_names` | Objects that start at the same place in every episode, such as the plate or the storage box (optional). They get no dataset column and are not verified; rollout logs their pose once in its log header, for reference. |
| `recorded_light_names` | Scene lights recorded as `lighting.<name>` = `[intensity, color_temperature]` (axis Image Augmentations). Each is a scene entity, such as `rect_light`. |
| `object_references` | Per object, the phrases the task text may use, `{form: phrase}`: `"name"` refers to the object by its name, `"color"` by its color only (axis Object Properties). Every listed object's color is recorded as `object_color.<name>`. Only the words change; the objects are never recolored. |
| `reference_weights` | How often an episode's task text uses each form, for example `{"name": 0.5, "color": 0.5}`. One form is drawn per episode and used for every object. A form with a weight above 0 needs a phrase for every object; otherwise the environment refuses to start. |
| `task_description_template` | The task text with one `{<object>}` field per object in `object_references`, filled with the phrases of the episode's form. Each episode's text is saved with it ([Data Collection §5](data_collection.md#5-the-recorder-and-the-dataset)) and is the instruction for language-conditioned policies. |
| `episode_length_s` | Time a policy gets per evaluation episode. |
| `env_spacing` | Distance between parallel environments; larger than the scene, or neighboring scenes overlap and cameras see into them. The shipped tasks use 15.0 m (the kitchen is 12.1 m × 7.0 m). |
| `scene.robot.init_state` | Robot base pose and home joint angles. |
| `scene.front.offset` | Pose of the front camera; it should show the whole workspace and the arm. |
| `parse_usd_and_create_subassets` | Makes objects inside a packaged USD scene addressable as `{ENV_REGEX_NS}/Scene/<name>`. Needed when the background scene already contains the objects. |
| `dynamic_reset_gripper_effort_limit` | `True` adapts the gripper force to the object nearest the gripper at every step; `False` keeps it constant. |

`__post_init__` always calls `super().__post_init__()` first, so that the
template's defaults are set before they are changed.

### 3.4 Randomization

The last call in `__post_init__` registers reset events that randomize the
scene at every `env.reset()`:

```python
domain_randomization(
    self,
    random_options=[
        randomize_objects_apart(
            pose_ranges={
                "blue_cup": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (0.0, 0.0)},
                "pink_cup": {"x": (0.02, 0.05), "y": (0.02, 0.05), "z": (0.0, 0.0)},
            },
            min_distance=0.11,
        ),
        randomize_light_conditions(
            "rect_light",
            intensity_range=(4500.0, 5500.0),
            color_variation=0.0,
            textures=[],
            color_temperature_range=(5300.0, 5900.0),
        ),
        choose_object_reference_forms(self.reference_weights),
    ],
)
```

- `randomize_objects_apart` offsets every listed object from its `init_state`
  by a uniform sample from its `pose_ranges` entry (keys `x`, `y`, `z`, `roll`,
  `pitch`, `yaw`). When two objects end up closer than `min_distance` (xy
  distance of their origins), that environment is resampled, up to
  `max_retries` times (default 5). `min_distance` is at least the sum of the
  two largest object radii, so that objects never spawn inside each other.
- For a single object, LeIsaac's `randomize_object_uniform(name, pose_range)`
  does the same without the distance check.
- `randomize_light_conditions(name, ...)` sets the light's intensity, and with
  `color_temperature_range` its color temperature, to uniform samples at every
  reset. `color_variation=0.0` and `textures=[]` keep the light's color and
  leave textures alone, as in the predefined tasks.
- `choose_object_reference_forms(self.reference_weights)` draws the episode's
  reference form at every reset; the task text is then built from
  `task_description_template`. Without this event every episode uses the most
  likely form.

Without randomization, every episode starts identically and the verifier
finds all episodes in one cell.

## 4. Register the task

**Required**

Registration gives the configuration a task id, so that `gym.make`,
`parse_env_cfg`, the data-collection scripts, and rollout can find it. Two edits are
needed.

1. `<my_task>/__init__.py` registers the id:

   ```python
   import gymnasium as gym

   gym.register(
       id="HCIS-MyTask-SingleArm-v0",
       entry_point="isaaclab.envs:ManagerBasedRLEnv",
       disable_env_checker=True,
       kwargs={"env_cfg_entry_point": f"{__name__}.my_task_env_cfg:MyTaskEnvCfg"},
   )
   ```

   `entry_point` is always `isaaclab.envs:ManagerBasedRLEnv`; only
   `env_cfg_entry_point` changes.

2. [`packages/simulator/src/simulator/tasks/__init__.py`](../packages/simulator/src/simulator/tasks/__init__.py)
   imports the new directory, next to the existing tasks:

   ```python
   from . import my_task  # noqa: F401
   ```

   Tasks are not discovered automatically: without this line,
   `import simulator.tasks` does not run the `gym.register` call and the task
   id is unknown.

The version suffix `v<N>` is increased whenever a change makes old datasets
incompatible with the task (for example new objects or a different success
check), so that each dataset stays associated with the task it was recorded
in.

## 5. Verify and test the task

**Required**

All checks run in the simulator environment (`make launch-sim`, or the Isaac
Lab container), because Isaac Lab modules can only be imported after the
simulator has started.

1. **Test new task configuration:**

   ```bash
   python -c "
   from isaaclab.app import AppLauncher
   app = AppLauncher(headless=True).app
   import simulator.tasks
   from isaaclab_tasks.utils import parse_env_cfg
   cfg = parse_env_cfg('HCIS-MyTask-SingleArm-v0', num_envs=1)
   print(type(cfg).__module__, type(cfg).__qualname__)
   print(cfg.task_description, cfg.tracked_object_names)
   app.close()
   "
   ```

   This should print `simulator.tasks.my_task.my_task_env_cfg`.

2. Run keyboard teleoperation on the new task id without `--record`
   ([Data Collection §3](data_collection.md#3-record-with-keyboard-teleoperation)).
   The objects should appear, both cameras show the workspace, and the objects land
   at a different position after every reset (press **R** to reset).

3. **Make sure the success check works:** a test attempt that solves the task makes the
   `success` term true, and one that does not leaves it false.

After these checks, recording proceeds as for a predefined task, from step 2 of
[Getting Started](getting_started.md#5-step-2-record-demonstrations). A new
task has no state machine, so it is recorded with keyboard teleoperation, or
a state machine can be written, modelled on those in
`packages/simulator/src/simulator/datagen/state_machine/` and adds it to
`TASK_REGISTRY` in `scripts/datagen/generate.py`.

## 6. Write the specs

**Required**

Data verification checks a dataset against specs, one Turtle file per
generalization axis ([Data Verification §3](data_verification.md#3-specs)).
There is no generator for them: the specs of a new task are written by
copying those of a predefined task and adapting them to the new objects.

```bash
mkdir -p specs/my_task
cp specs/cup_stacking/object_pose.ttl specs/cup_stacking/image_augmentation.ttl \
   specs/cup_stacking/object_property.ttl specs/my_task/
```

In the copies:

- `object_pose.ttl`: one object block per name in `tracked_object_names`
  (copy or delete blocks); `cp:name` is the object's scene name.
- `image_augmentation.ttl`: `cp:lightName` is a name in
  `recorded_light_names`.
- `object_property.ttl`: one object block per object in `object_references`;
  the phrases correspond to its `"name"` and `"color"` entries.

The file names stay `object_pose.ttl`, `image_augmentation.ttl` and
`object_property.ttl`, and the values stay `"TODO"`, as in the shipped specs.
The values are filled in in a copy under `my_specs/`, as for a
predefined task. `make test` checks that every file under `specs/<task>/` is
named after an axis and has only its values left as `TODO`.

## 7. Evaluate on the new task

**Background**

For a separate evaluation configuration (other nominal poses, ranges, or
lighting than in recording), an evaluation file is written in the same
form as the files in `eval/`: a configuration class plus a `gym.register(...)`
call with a unique `TASK_ID`, passed to rollout as a path
([Rollout §5](rollout.md#5-evaluating-a-self-defined-task)).
The evaluation file is best kept with the task, so that the evaluation can be
reproduced.

## 8. Common pitfalls

**Background**

- **Wrong prim namespace.** Per-environment assets need `{ENV_REGEX_NS}/...`.
- **Missing `env.scene.env_origins`.** A success check that compares raw world
  positions works for environment 0 only.
- **Quaternion order.** Isaac Lab uses `(w, x, y, z)`; SciPy and ROS often use
  `(x, y, z, w)`.
- **Task not imported.** The task id is unknown because the import line in
  `simulator/tasks/__init__.py` is missing ([section 4](#4-register-the-task)).
- **New camera.** Every camera in the scene becomes a dataset column and needs
  a matching observation term; the policy contract only allows `front` and
  `wrist`.
- **Empty `tracked_object_names`.** The recorder raises a `ValueError` and
  nothing can be verified.
- **A light or object missing from the lists.** A light not in
  `recorded_light_names` or an object not in `object_references` gets no
  `lighting.*` or `object_color.*` column, and `verify` reports an error for a
  spec that names it.
- **Template field without a phrase.** Every `{<object>}` field in
  `task_description_template` needs an entry in `object_references` with a
  phrase for every form that has a weight above 0.

## 9. Checklist

**Required**

- [ ] Files at `packages/simulator/src/simulator/tasks/<my_task>/`, named by the conventions in section 2.
- [ ] Scene, terminations, and environment classes are `@configclass` subclasses of the template classes.
- [ ] Success check uses env-local positions (subtracts `env.scene.env_origins`, or compares two objects of the same env) and returns a `(num_envs,)` boolean tensor.
- [ ] `task_description`, `tracked_object_names`, and `episode_length_s` are set.
- [ ] `recorded_light_names` lists the scene light (for example `rect_light`), and `object_references`, `reference_weights` and `task_description_template` are set.
- [ ] `__post_init__` calls `super().__post_init__()` first.
- [ ] Reset events for object positions, lighting (`randomize_light_conditions`), and the reference form (`choose_object_reference_forms`) are registered.
- [ ] `specs/<my_task>/` holds `object_pose.ttl`, `image_augmentation.ttl` and `object_property.ttl`, copied from a predefined task and adapted to the new objects and light.
- [ ] `gym.register` in the task's `__init__.py`, and the import line in `simulator/tasks/__init__.py`.
- [ ] `parse_env_cfg(<task id>)` loads the class from the new file.
- [ ] The scene runs, the cameras show the workspace, and the success check behaves as intended.
- [ ] The rollout contract settings are unchanged; the template and `dependencies/IsaacLab/` are not edited.
