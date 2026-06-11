"""SceneSpec schema, parser, asset download, and provenance for BYO sim assets.

This module describes the manipulated object(s) of a Genesis pick-and-place
scene independently of the simulator. It is deliberately free of ``torch`` /
``genesis`` imports at module level so it can be unit-tested without a GPU.

A SceneSpec is a JSON document (downloaded from ``scene_spec_uri`` or
synthesized from ``assets_uri`` / ``byo_mesh_uri``) describing one or more
objects. Each object has an ``asset_source``:

- ``byo_mesh``: a customer mesh fetched from an S3/URI (.obj/.glb/.urdf/.ply…),
  loaded with ``gs.morphs.Mesh(file=<local_path>)``.
- ``genesis_builtin``: a mesh shipped inside the Genesis package
  (a path relative to ``<genesis>/assets``), also loaded via ``gs.morphs.Mesh``.
- ``primitive``: a box/sphere/cylinder built from primitive morphs. The current
  default red ``gs.morphs.Box`` cube is the primitive fallback.

Provenance: every object records ``{uri, local_path, sha256, asset_source,
loaded}`` and the scene records a single ``asset_fallback_used`` flag, so a run
can prove the requested mesh actually loaded and that nothing silently fell
back to a primitive. If a requested mesh cannot be resolved, resolution and the
build raise ``SceneSpecError`` rather than substituting a primitive.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCENE_SPEC_SCHEMA = "npa.sim2real.manip_scene_spec.v1"

ASSET_SOURCE_BYO_MESH = "byo_mesh"
ASSET_SOURCE_GENESIS_BUILTIN = "genesis_builtin"
ASSET_SOURCE_PRIMITIVE = "primitive"
# Isaac Lab / Isaac Sim stock asset (built-in lift-cube manipuland). No
# download: it is materialized inside the Isaac image at rollout time. Recorded
# in provenance as ``asset_source=isaac_stock`` (no sha256, no fallback).
ASSET_SOURCE_ISAAC_STOCK = "isaac_stock"
ASSET_SOURCES = (
    ASSET_SOURCE_BYO_MESH,
    ASSET_SOURCE_GENESIS_BUILTIN,
    ASSET_SOURCE_PRIMITIVE,
    ASSET_SOURCE_ISAAC_STOCK,
)

ROLE_MANIPULAND = "manipuland"
ROLE_TARGET = "target"
ROLE_STATIC = "static"
ROLES = (ROLE_MANIPULAND, ROLE_TARGET, ROLE_STATIC)

PRIMITIVE_BOX = "box"
PRIMITIVE_SPHERE = "sphere"
PRIMITIVE_CYLINDER = "cylinder"
PRIMITIVES = (PRIMITIVE_BOX, PRIMITIVE_SPHERE, PRIMITIVE_CYLINDER)

MESH_SUFFIXES = (".obj", ".glb", ".gltf", ".ply", ".stl", ".urdf", ".xml", ".dae")

# The exact default object that today's hardcoded scene builds: a 4cm red cube.
DEFAULT_CUBE_SIZE = 0.04
DEFAULT_CUBE_POS = (0.5, 0.0, 0.04)
DEFAULT_TARGET_POS = (0.5, 0.3, 0.04)
DEFAULT_TARGET_THRESHOLD = 0.05
DEFAULT_COLOR = (1.0, 0.0, 0.0)


class SceneSpecError(RuntimeError):
    """Raised when a SceneSpec is malformed or an asset cannot be resolved."""


@dataclass
class ObjectSpec:
    """One physical object in the scene (manipuland, target, or static)."""

    name: str
    asset_source: str
    role: str = ROLE_MANIPULAND
    # Source references
    uri: str = ""  # byo_mesh source (s3:// or local path)
    builtin_path: str = ""  # genesis_builtin path relative to <genesis>/assets
    primitive: str = PRIMITIVE_BOX
    # Geometry
    scale: float | tuple[float, float, float] = 1.0  # mesh scale
    size: tuple[float, float, float] = (
        DEFAULT_CUBE_SIZE,
        DEFAULT_CUBE_SIZE,
        DEFAULT_CUBE_SIZE,
    )  # box
    radius: float = 0.02  # sphere/cylinder
    height: float = 0.04  # cylinder
    # Pose
    pos: tuple[float, float, float] = DEFAULT_CUBE_POS
    euler: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # Appearance / physics
    color: tuple[float, float, float] = DEFAULT_COLOR
    mass: float | None = None
    friction: float | None = None
    fixed: bool = False
    # Resolved fields (populated by resolve_scene_assets)
    local_path: str = ""
    sha256: str = ""
    loaded: bool = False

    def is_mesh(self) -> bool:
        return self.asset_source in (
            ASSET_SOURCE_BYO_MESH,
            ASSET_SOURCE_GENESIS_BUILTIN,
        )

    def provenance(self) -> dict[str, Any]:
        """Per-object provenance record (mirrors #84's image_digests pattern)."""

        return {
            "name": self.name,
            "role": self.role,
            "asset_source": self.asset_source,
            "uri": self.uri,
            "builtin_path": self.builtin_path,
            "local_path": self.local_path,
            "sha256": self.sha256,
            "loaded": bool(self.loaded),
        }


@dataclass
class SceneSpec:
    """A parsed, simulator-agnostic scene description for the pick-place env."""

    objects: list[ObjectSpec] = field(default_factory=list)
    schema: str = SCENE_SPEC_SCHEMA
    goal_pos: tuple[float, float, float] = DEFAULT_TARGET_POS
    goal_threshold: float = DEFAULT_TARGET_THRESHOLD
    source_uri: str = ""
    asset_fallback_used: bool = False

    def manipuland(self) -> ObjectSpec:
        for obj in self.objects:
            if obj.role == ROLE_MANIPULAND:
                return obj
        if self.objects:
            return self.objects[0]
        raise SceneSpecError("SceneSpec has no objects")

    def provenance_block(self) -> dict[str, Any]:
        """Scene-level provenance written into report.json / consumed spec."""

        objects = [obj.provenance() for obj in self.objects]
        return {
            "schema": "npa.sim2real.asset_provenance.v1",
            "scene_spec_schema": self.schema,
            "source_uri": self.source_uri,
            "asset_fallback_used": bool(self.asset_fallback_used),
            "goal_pos": list(self.goal_pos),
            "goal_threshold": self.goal_threshold,
            "objects": objects,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "goal_pos": list(self.goal_pos),
            "goal_threshold": self.goal_threshold,
            "source_uri": self.source_uri,
            "objects": [_object_to_dict(obj) for obj in self.objects],
        }


def default_scene_spec() -> SceneSpec:
    """Return the SceneSpec that reproduces today's red-Box primitive cube."""

    return SceneSpec(
        objects=[
            ObjectSpec(
                name="cube",
                asset_source=ASSET_SOURCE_PRIMITIVE,
                role=ROLE_MANIPULAND,
                primitive=PRIMITIVE_BOX,
                size=(DEFAULT_CUBE_SIZE, DEFAULT_CUBE_SIZE, DEFAULT_CUBE_SIZE),
                pos=DEFAULT_CUBE_POS,
                color=DEFAULT_COLOR,
            )
        ],
        goal_pos=DEFAULT_TARGET_POS,
        goal_threshold=DEFAULT_TARGET_THRESHOLD,
    )


def _coerce_triple(value: Any, name: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise SceneSpecError(f"{name} must be a 3-element list, got {value!r}")
    try:
        return (float(value[0]), float(value[1]), float(value[2]))
    except (TypeError, ValueError) as exc:
        raise SceneSpecError(f"{name} must be numeric: {value!r}") from exc


def _coerce_scale(value: Any) -> float | tuple[float, float, float]:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return _coerce_triple(value, "scale")
    raise SceneSpecError(f"scale must be a number or 3-element list, got {value!r}")


def _object_from_dict(raw: dict[str, Any], index: int) -> ObjectSpec:
    if not isinstance(raw, dict):
        raise SceneSpecError(f"object[{index}] must be a JSON object, got {raw!r}")
    asset_source = str(raw.get("asset_source") or "").strip()
    if asset_source not in ASSET_SOURCES:
        raise SceneSpecError(
            f"object[{index}] asset_source must be one of {ASSET_SOURCES}, "
            f"got {asset_source!r}"
        )
    role = str(raw.get("role") or ROLE_MANIPULAND).strip()
    if role not in ROLES:
        raise SceneSpecError(
            f"object[{index}] role must be one of {ROLES}, got {role!r}"
        )
    name = str(raw.get("name") or f"object-{index:02d}").strip()
    obj = ObjectSpec(name=name, asset_source=asset_source, role=role)

    if asset_source == ASSET_SOURCE_BYO_MESH:
        uri = str(raw.get("uri") or "").strip()
        if not uri:
            raise SceneSpecError(f"object[{index}] byo_mesh requires a non-empty uri")
        obj.uri = uri
    elif asset_source == ASSET_SOURCE_GENESIS_BUILTIN:
        builtin = str(raw.get("builtin_path") or raw.get("file") or "").strip()
        if not builtin:
            raise SceneSpecError(
                f"object[{index}] genesis_builtin requires builtin_path"
            )
        obj.builtin_path = builtin
    elif asset_source == ASSET_SOURCE_ISAAC_STOCK:
        # Optional reference to the stock Isaac asset (e.g. a task id or a
        # built-in USD key); no download, materialized inside the Isaac image.
        obj.builtin_path = str(raw.get("builtin_path") or raw.get("stock_asset") or "").strip()
    else:  # primitive
        primitive = str(raw.get("primitive") or PRIMITIVE_BOX).strip()
        if primitive not in PRIMITIVES:
            raise SceneSpecError(
                f"object[{index}] primitive must be one of {PRIMITIVES}, "
                f"got {primitive!r}"
            )
        obj.primitive = primitive

    if "scale" in raw:
        obj.scale = _coerce_scale(raw["scale"])
    if "size" in raw:
        obj.size = _coerce_triple(raw["size"], "size")
    if "radius" in raw:
        obj.radius = float(raw["radius"])
    if "height" in raw:
        obj.height = float(raw["height"])
    if "pos" in raw:
        obj.pos = _coerce_triple(raw["pos"], "pos")
    if "euler" in raw:
        obj.euler = _coerce_triple(raw["euler"], "euler")
    if "color" in raw:
        obj.color = _coerce_triple(raw["color"], "color")
    if raw.get("mass") is not None:
        obj.mass = float(raw["mass"])
    if raw.get("friction") is not None:
        obj.friction = float(raw["friction"])
    obj.fixed = bool(raw.get("fixed", role == ROLE_STATIC))
    return obj


def _object_to_dict(obj: ObjectSpec) -> dict[str, Any]:
    scale = list(obj.scale) if isinstance(obj.scale, tuple) else obj.scale
    return {
        "name": obj.name,
        "asset_source": obj.asset_source,
        "role": obj.role,
        "uri": obj.uri,
        "builtin_path": obj.builtin_path,
        "primitive": obj.primitive,
        "scale": scale,
        "size": list(obj.size),
        "radius": obj.radius,
        "height": obj.height,
        "pos": list(obj.pos),
        "euler": list(obj.euler),
        "color": list(obj.color),
        "mass": obj.mass,
        "friction": obj.friction,
        "fixed": obj.fixed,
    }


def parse_scene_spec(doc: dict[str, Any], *, source_uri: str = "") -> SceneSpec:
    """Parse and validate a SceneSpec JSON document into a SceneSpec."""

    if not isinstance(doc, dict):
        raise SceneSpecError(f"SceneSpec must be a JSON object, got {type(doc)!r}")
    raw_objects = doc.get("objects")
    if not isinstance(raw_objects, list) or not raw_objects:
        raise SceneSpecError("SceneSpec must include a non-empty 'objects' list")
    objects = [_object_from_dict(raw, index) for index, raw in enumerate(raw_objects)]
    manipulands = [obj for obj in objects if obj.role == ROLE_MANIPULAND]
    if not manipulands:
        raise SceneSpecError("SceneSpec must include at least one manipuland object")
    goal_pos = (
        _coerce_triple(doc["goal_pos"], "goal_pos")
        if doc.get("goal_pos") is not None
        else DEFAULT_TARGET_POS
    )
    goal_threshold = float(doc.get("goal_threshold", DEFAULT_TARGET_THRESHOLD))
    if goal_threshold <= 0:
        raise SceneSpecError("goal_threshold must be positive")
    return SceneSpec(
        objects=objects,
        schema=str(doc.get("schema") or SCENE_SPEC_SCHEMA),
        goal_pos=goal_pos,
        goal_threshold=goal_threshold,
        source_uri=source_uri or str(doc.get("source_uri") or ""),
    )


def synthesize_scene_spec(
    *,
    assets_uri: str = "",
    byo_mesh_uri: str = "",
    scale: float | tuple[float, float, float] = 1.0,
    pos: tuple[float, float, float] = DEFAULT_CUBE_POS,
    goal_pos: tuple[float, float, float] = DEFAULT_TARGET_POS,
    goal_threshold: float = DEFAULT_TARGET_THRESHOLD,
) -> SceneSpec:
    """Build a single-manipuland SceneSpec from a bare mesh URI.

    Used when only ``assets_uri`` / ``byo_mesh_uri`` is provided (no full
    SceneSpec JSON). Keeps the Franka robot and the documented target zone.
    """

    mesh_uri = (byo_mesh_uri or assets_uri or "").strip()
    if not mesh_uri:
        raise SceneSpecError(
            "synthesize_scene_spec requires byo_mesh_uri or assets_uri"
        )
    return SceneSpec(
        objects=[
            ObjectSpec(
                name="byo_object",
                asset_source=ASSET_SOURCE_BYO_MESH,
                role=ROLE_MANIPULAND,
                uri=mesh_uri,
                scale=scale,
                pos=pos,
                color=DEFAULT_COLOR,
            )
        ],
        goal_pos=goal_pos,
        goal_threshold=goal_threshold,
        source_uri=mesh_uri,
    )


def default_isaac_stock_scene_spec(*, stock_asset: str = "lift_cube") -> SceneSpec:
    """Return the stock Isaac Lab scene (built-in lift-cube manipuland).

    Used by the Isaac held-out rollout when no BYO mesh/SceneSpec is supplied.
    Records ``asset_source=isaac_stock`` provenance (no sha256, no fallback).
    """

    return SceneSpec(
        objects=[
            ObjectSpec(
                name="isaac_stock_cube",
                asset_source=ASSET_SOURCE_ISAAC_STOCK,
                role=ROLE_MANIPULAND,
                builtin_path=stock_asset,
                size=(DEFAULT_CUBE_SIZE, DEFAULT_CUBE_SIZE, DEFAULT_CUBE_SIZE),
                pos=DEFAULT_CUBE_POS,
                color=DEFAULT_COLOR,
            )
        ],
        goal_pos=DEFAULT_TARGET_POS,
        goal_threshold=DEFAULT_TARGET_THRESHOLD,
        source_uri="isaac://stock/lift_cube",
    )


def sha256_file(path: str | Path) -> str:
    """Return the hex SHA-256 of a file (streamed)."""

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_local_asset(path: Path, *, what: str) -> None:
    if not path.is_file():
        raise SceneSpecError(f"{what} did not produce a file at {path}")
    if path.stat().st_size == 0:
        raise SceneSpecError(f"{what} produced an empty file at {path}")


def download_asset(
    uri: str,
    dest_dir: str | Path,
    *,
    client: Any = None,
    endpoint_url: str = "",
) -> Path:
    """Fetch a mesh asset (s3:// or local path) to ``dest_dir``; validate it.

    Returns the local path. Raises ``SceneSpecError`` if the asset is missing
    or empty. The S3 client follows the repo's ``StorageClient`` patterns.
    """

    uri = str(uri or "").strip()
    if not uri:
        raise SceneSpecError("download_asset requires a non-empty uri")
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(uri.split("?", 1)[0].rstrip("/")).name or "asset.bin"
    dest = dest_dir / filename

    if uri.startswith("s3://"):
        if client is None:
            from npa.clients.storage import StorageClient

            client = StorageClient.from_environment(endpoint_url=endpoint_url)
        client.download_path(uri, str(dest))
    elif uri.startswith("file://"):
        src = Path(uri[len("file://") :])
        _validate_local_asset(src, what=f"asset source {uri}")
        dest.write_bytes(src.read_bytes())
    else:
        src = Path(uri)
        _validate_local_asset(src, what=f"asset source {uri}")
        if src.resolve() != dest.resolve():
            dest.write_bytes(src.read_bytes())

    _validate_local_asset(dest, what=f"download of {uri}")
    return dest


def resolve_genesis_builtin_path(builtin_path: str) -> Path:
    """Resolve a builtin asset path relative to the installed Genesis package.

    Imported lazily so this module stays importable without genesis-world.
    """

    builtin_path = str(builtin_path or "").strip().lstrip("/")
    if not builtin_path:
        raise SceneSpecError("genesis_builtin requires a builtin_path")
    import genesis as gs  # noqa: PLC0415 - lazy, GPU image only

    assets_root = Path(gs.__file__).resolve().parent / "assets"
    candidate = assets_root / builtin_path
    if not candidate.is_file():
        raise SceneSpecError(
            f"genesis_builtin asset not found under {assets_root}: {builtin_path}"
        )
    return candidate


def resolve_scene_assets(
    spec: SceneSpec,
    *,
    dest_dir: str | Path,
    client: Any = None,
    endpoint_url: str = "",
    downloader: Any = None,
    builtin_resolver: Any = None,
) -> SceneSpec:
    """Download/resolve every object's asset, computing local_path + sha256.

    Mutates ``spec`` objects in place (sets ``local_path`` / ``sha256``) and
    returns it. Mesh assets that fail to resolve raise ``SceneSpecError`` —
    there is no silent primitive fallback.
    """

    downloader = downloader or download_asset
    builtin_resolver = builtin_resolver or resolve_genesis_builtin_path
    dest_dir = Path(dest_dir)
    for obj in spec.objects:
        if obj.asset_source == ASSET_SOURCE_BYO_MESH:
            local = downloader(
                obj.uri,
                dest_dir / _safe_name(obj.name),
                client=client,
                endpoint_url=endpoint_url,
            )
            obj.local_path = str(local)
            obj.sha256 = sha256_file(local)
        elif obj.asset_source == ASSET_SOURCE_GENESIS_BUILTIN:
            local = builtin_resolver(obj.builtin_path)
            _validate_local_asset(Path(local), what=f"genesis_builtin {obj.builtin_path}")
            obj.local_path = str(local)
            obj.sha256 = sha256_file(local)
        # primitives: nothing to download; local_path/sha256 stay empty.
    return spec


def _safe_name(value: str) -> str:
    chars = [c if c.isalnum() or c in ("-", "_", ".") else "-" for c in str(value)]
    return "".join(chars) or "object"
