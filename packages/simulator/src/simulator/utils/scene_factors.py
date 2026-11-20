"""Read the lighting and object colors of a running scene, for the dataset and the rollout log.

Both are read from the USD stage, the same values the renderer uses:

* ``lighting.<name>`` = ``[intensity, color_temperature]`` of a scene light
  (USD ``inputs:intensity`` and ``inputs:colorTemperature``; when the light
  does not use its color temperature, 6500 K, the white point). The light is a
  scene entity of the task configuration, listed in ``recorded_light_names``.
* ``object_color.<name>`` = ``[r, g, b]``, the diffuse color of the material
  bound to the object (OmniPBR ``inputs:diffuse_color_constant`` or
  UsdPreviewSurface ``inputs:diffuseColor``). Recorded for every object in the
  task configuration's ``object_color_palettes``.

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


def pick_color_words(rng, palettes: dict[str, dict], distinct: bool = True) -> dict[str, str]:
    """One color word per object of ``palettes``, drawn with ``rng`` (a ``random.Random``).

    With ``distinct``, no two objects get the same word, so the task text never names two objects
    by the same color ("the red cup on the red cup"). The objects pick in a random order, each among
    the words still free; an object whose words are all taken falls back to its whole palette.
    """
    order = list(palettes)
    rng.shuffle(order)
    taken: set[str] = set()
    picked: dict[str, str] = {}
    for name in order:
        words = sorted(palettes[name])
        free = [word for word in words if word not in taken] if distinct else words
        picked[name] = rng.choice(free or words)
        taken.add(picked[name])
    return {name: picked[name] for name in palettes}


def _env_prim(prims: list, env_id: int):
    """The prim of ``env_id``; a global (not env-scoped) asset has one prim shared by all envs."""
    return prims[0] if len(prims) == 1 else prims[env_id]


def light_values(scene, name: str, env_ids) -> list[list[float]]:
    """``[intensity, color_temperature]`` of the scene light ``name`` for each env in ``env_ids``."""
    prims = scene[name].prims
    values = []
    for env_id in env_ids:
        prim = _env_prim(prims, int(env_id))
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
    path = scene[name].cfg.prim_path.replace("{ENV_REGEX_NS}", scene.env_regex_ns)
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


def forget_shaders(stage, root_path: str) -> None:
    """Drop the cached shaders of ``root_path`` (after binding a new material to it)."""
    _SHADER_CACHE.pop((id(stage), root_path), None)


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
