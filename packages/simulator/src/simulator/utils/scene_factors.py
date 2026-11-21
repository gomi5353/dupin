"""Read the lighting and object colors of a running scene, for the dataset and the rollout log.

Both are read from the USD stage, the same values the renderer uses:

* ``lighting.<name>`` = ``[intensity, color_temperature]`` of a scene light
  (USD ``inputs:intensity`` and ``inputs:colorTemperature``; when the light
  does not use its color temperature, 6500 K, the white point). The light is a
  scene entity of the task configuration, listed in ``recorded_light_names``.
* ``object_color.<name>`` = ``[r, g, b]``, the diffuse color of the material
  bound to the object (OmniPBR ``inputs:diffuse_color_constant`` or
  UsdPreviewSurface ``inputs:diffuseColor``). Recorded for every object in the
  task configuration's ``object_references``; the Object Properties axis
  (S-PROP) checks that it stays the same, since only the task text varies.

Read back by ``packages/data_verification`` (``dataset.py``, ``rollout_log.py``);
keep the key prefixes and value names in sync with it.
"""

from __future__ import annotations

LIGHTING_KEY_PREFIX = "lighting."
LIGHTING_NAMES = ["intensity", "color_temperature"]
OBJECT_COLOR_KEY_PREFIX = "object_color."
OBJECT_COLOR_NAMES = ["r", "g", "b"]

WHITE_POINT_K = 6500.0
DIFFUSE_INPUTS = ("inputs:diffuse_color_constant", "inputs:diffuseColor")


def lighting_features(light_names: list[str]) -> dict[str, dict]:
    """LeRobot feature specs for the ``lighting.<name>`` columns."""
    return {
        f"{LIGHTING_KEY_PREFIX}{name}": {"dtype": "float32", "shape": (len(LIGHTING_NAMES),), "names": list(LIGHTING_NAMES)}
        for name in light_names
    }


def object_color_features(object_names: list[str]) -> dict[str, dict]:
    """LeRobot feature specs for the ``object_color.<name>`` columns."""
    return {
        f"{OBJECT_COLOR_KEY_PREFIX}{name}": {"dtype": "float32", "shape": (len(OBJECT_COLOR_NAMES),), "names": list(OBJECT_COLOR_NAMES)}
        for name in object_names
    }


def reference_form_errors(references: dict[str, dict[str, str]], weights: dict[str, float]) -> list[str]:
    """Problems with ``object_references`` and ``reference_weights`` (a form with weight > 0 must exist for every object)."""
    errors = []
    if not weights or any(w < 0 for w in weights.values()) or sum(weights.values()) <= 0:
        errors.append(f"reference_weights {weights} must be >= 0 with a positive sum")
    for form, weight in weights.items():
        missing = [name for name, refs in references.items() if form not in refs]
        if weight > 0 and missing:
            errors.append(f"reference form '{form}' has weight {weight:g} but no phrase for {', '.join(missing)} in object_references")
    return errors


def pick_reference_form(rng, weights: dict[str, float]) -> str:
    """One form, drawn with ``rng`` (a ``random.Random``) in proportion to ``weights``; forms of weight 0 never."""
    forms = sorted(form for form, weight in weights.items() if weight > 0)
    return rng.choices(forms, weights=[weights[f] for f in forms])[0]


def _cfg_prim_path(scene, name: str) -> str:
    """Prim path of ``name`` as written in the scene configuration.

    Read from ``scene.cfg``: ``scene[name]`` of an ``AssetBaseCfg`` (a light, the
    kitchen) is an ``XformPrimView``, which has no ``cfg``.
    """
    return getattr(scene.cfg, name).prim_path


def is_env_scoped(scene, name: str) -> bool:
    """True when ``name`` has one copy per env (its prim path is under ``{ENV_REGEX_NS}``).

    The scene replaces ``{ENV_REGEX_NS}`` in its configuration by the regex
    ``/World/envs/env_.*`` when it is built, so both forms are accepted.
    """
    path = _cfg_prim_path(scene, name)
    return "{ENV_REGEX_NS}" in path or path.startswith(scene.env_regex_ns)


def env_prims(scene, name: str, env_ids) -> list:
    """The USD prim of the scene entity ``name`` in each env of ``env_ids``.

    An entity under ``{ENV_REGEX_NS}`` has one prim per env, found by its path:
    ``scene[name].prims`` of an ``AssetBaseCfg`` lists env 0's prim only, and
    the other envs' copies mirror it until written through their own path. A
    global entity (not under ``{ENV_REGEX_NS}``) has one prim, shared by all envs.
    """
    import omni.usd

    if not is_env_scoped(scene, name):
        prim = scene[name].prims[0]
        return [prim for _ in env_ids]
    stage = omni.usd.get_context().get_stage()
    return [stage.GetPrimAtPath(object_prim_path(scene, name, int(env_id))) for env_id in env_ids]


def light_values(scene, name: str, env_ids) -> list[list[float]]:
    """``[intensity, color_temperature]`` of the scene light ``name`` for each env in ``env_ids``."""
    values = []
    for prim in env_prims(scene, name, env_ids):
        intensity = prim.GetAttribute("inputs:intensity").Get()
        enabled = prim.GetAttribute("inputs:enableColorTemperature").Get()
        temperature = prim.GetAttribute("inputs:colorTemperature").Get() if enabled else None
        values.append([float(intensity or 0.0), float(temperature) if temperature is not None else WHITE_POINT_K])
    return values


def object_prim_path(scene, name: str, env_id: int) -> str:
    """Prim path of the object ``name`` in ``env_id``.

    The scene resolves ``{ENV_REGEX_NS}`` in the configuration to the regex ``/World/envs/env_.*``;
    replace either form by the path of ``env_id``.
    """
    path = _cfg_prim_path(scene, name).replace("{ENV_REGEX_NS}", scene.env_regex_ns)
    return path.replace(scene.env_regex_ns, scene.env_prim_paths[env_id])


_SHADER_CACHE: dict[tuple[int, str], list] = {}


def diffuse_shaders(stage, root_path: str) -> list:
    """Shaders of the materials bound to the meshes under ``root_path`` that have a diffuse color input.

    Cached per stage and path: material bindings do not change while a scene runs, and the recorder
    reads the colors on every frame.
    """
    key = (id(stage), root_path)
    if key not in _SHADER_CACHE:
        _SHADER_CACHE[key] = _find_diffuse_shaders(stage, root_path)
    return [prim for prim in _SHADER_CACHE[key] if prim.IsValid()]


def _find_diffuse_shaders(stage, root_path: str) -> list:
    from pxr import Usd, UsdGeom, UsdShade

    shaders, seen = [], set()
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        return shaders
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdGeom.Gprim):
            continue
        material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        if not material:
            continue
        for shader_prim in Usd.PrimRange(material.GetPrim()):
            path = str(shader_prim.GetPath())
            if path in seen or not shader_prim.IsA(UsdShade.Shader):
                continue
            if any(shader_prim.GetAttribute(a).IsValid() for a in DIFFUSE_INPUTS):
                seen.add(path)
                shaders.append(shader_prim)
    return shaders


def read_diffuse(shader_prim) -> list[float] | None:
    for attr in DIFFUSE_INPUTS:
        value = shader_prim.GetAttribute(attr).Get() if shader_prim.GetAttribute(attr).IsValid() else None
        if value is not None:
            return [float(v) for v in value]
    return None


def object_colors(scene, names: list[str], env_ids) -> dict[str, list[list[float]]]:
    """``[r, g, b]`` of each object in ``names`` for each env in ``env_ids`` (NaN when no material with a diffuse color is bound)."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    colors: dict[str, list[list[float]]] = {}
    for name in names:
        values = []
        for env_id in env_ids:
            shaders = diffuse_shaders(stage, object_prim_path(scene, name, int(env_id)))
            rgb = read_diffuse(shaders[0]) if shaders else None
            values.append(rgb if rgb is not None else [float("nan")] * 3)
        colors[name] = values
    return colors
