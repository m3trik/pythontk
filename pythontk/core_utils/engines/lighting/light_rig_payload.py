# !/usr/bin/python
# coding=utf-8
"""The light rig payload a look match writes and a host's rig tool builds, declared once.

The host-neutral spelling: a light's type and its common settings by name
(``spot``, ``cone``, ``radius``), a host's own attributes riding in ``attrs`` by
the host's names. :meth:`LookMatch.to_rig` writes it and validates what it wrote;
mayatk's ``LightRig`` builds it, mapping every type and setting declared here
onto Maya's (its tables are checked against these, so a name added here fails
its build until it is mapped). A host's own library form (Maya node types and
attribute names) is that host's, not this contract.

Lengths are centimetres and angles degrees, whatever units the scene a rig is
saved from or applied to works in -- positions, aim points, radii and any
length an ``attrs`` or override value holds, rotations and cone angles alike --
so a rig saved in one scene lights another, in any host. A host converts at
its own boundary (mayatk's ``LightRig`` runs under ``Units.working_as()``).

Typed :class:`~pythontk.SchemaSpec` declarations: ``LightRigPayload.validate(rig)``
reports what does not fit. The library rigs are saved in is declared here too
(:meth:`LightRigPayload.library`): one per user, every host and tool's.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from pythontk.core_utils.schema_spec import SchemaSpec

if TYPE_CHECKING:
    from pythontk.core_utils.presets.store import PresetStore

_field = SchemaSpec.spec_field
_Vec = Optional[List[float]]


@dataclass
class LightRigLight(SchemaSpec):
    """One light of a rig, in the host-neutral spelling: lengths in centimetres,
    angles in degrees."""

    TYPED = True
    #: The light types a host builds.
    TYPES = ("spot", "point", "area", "aiArea", "directional", "photometric")
    #: The settings a host maps onto its own attributes, by neutral name.
    SETTINGS = (
        "intensity",
        "exposure",
        "cone",
        "penumbra",
        "dropoff",
        "radius",
        "spread",
        "normalize",
        "samples",
        "decay",
        "specular",
        "diffuse",
        "volume",
        "roundness",
        "camera",
        "color",
    )
    #: Keys a light may carry that build nothing (a look match's notes).
    NOTES = ("kelvin_note", "up", "note", "notes")

    name: str = _field(help="The light's name in the rig's group.", required=True)
    type: str = _field(help="What kind of light.", required=True, choices=TYPES)
    pos: _Vec = _field(help="World position, centimetres.")
    rot: _Vec = _field(help="World rotation, degrees (xyz).")
    aim: _Vec = _field(
        help="A world point the light faces (instead of rot), centimetres."
    )
    scale: _Vec = _field(
        help="Scale; an area light's scales its plate, 2 centimetres square."
    )
    rotate_order: Optional[int] = _field(help="The host's rotate order index.")
    visible: Optional[bool] = _field(help="False: built hidden.")
    kelvin: Optional[float] = _field(help="Colour temperature (instead of a colour).")
    gobo: Optional[str] = _field(help="A gobo (slide) image's path.")
    gobo_rotate: Optional[float] = _field(help="The gobo's rotation, degrees.")
    intensity: Optional[float] = _field(help="Light intensity.")
    exposure: Optional[float] = _field(help="Exposure, stops.")
    cone: Optional[float] = _field(help="A spot's cone angle, degrees.")
    penumbra: Optional[float] = _field(help="A spot's penumbra, degrees.")
    dropoff: Optional[float] = _field(help="A spot's dropoff.")
    radius: Optional[float] = _field(help="A soft light's radius, centimetres.")
    spread: Optional[float] = _field(help="An area light's spread, 0..1.")
    normalize: Optional[Union[bool, int]] = _field(help="Intensity per unit area.")
    samples: Optional[int] = _field(help="Light samples.")
    decay: Optional[int] = _field(help="Decay rate (0 none, 2 quadratic).")
    specular: Optional[float] = _field(help="Specular contribution scale.")
    diffuse: Optional[float] = _field(help="Diffuse contribution scale.")
    volume: Optional[float] = _field(help="Volume contribution scale.")
    roundness: Optional[float] = _field(help="An area light's roundness.")
    camera: Optional[float] = _field(help="Visibility to camera rays.")
    color: _Vec = _field(help="Linear RGB colour (in the scene's rendering space).")
    attrs: Optional[Dict[str, Any]] = _field(
        help="The host's own attributes, by its names."
    )
    textures: Optional[Dict[str, Any]] = _field(help="Attributes fed from image files.")
    user_attrs: Optional[Dict[str, Any]] = _field(
        help="Extra attributes the light carries."
    )
    filters: Optional[List[Dict[str, Any]]] = _field(
        help="Light filters (gobo, decay...)."
    )
    # Notes: free-form, they build nothing.
    kelvin_note: Any = _field(help="A note (builds nothing).")
    up: Any = _field(help="A note (builds nothing).")
    note: Any = _field(help="A note (builds nothing).")
    notes: Any = _field(help="A note (builds nothing).")


@dataclass
class LightRigFixture(SchemaSpec):
    """A light fixture in the scene, and which of three ways lights the room from it.

    One physical fixture casts one light, which a renderer can carry three ways
    -- exactly one at a time, since two would light the room twice:

    * ``glow``: the camera sees its emission and it lights nothing (its meshes
      unseen by diffuse rays, its lamps off).
    * ``emissive``: its emission lights the room too (its meshes seen by
      diffuse rays); its lamps off.
    * ``lamp``: its lamps light the room, each at its ``lamps`` intensity; its
      emission glows and lights nothing.

    A variant made in a mode other than glow says which (``MOODY_LAMP``,
    ``MOODY_EMISSIVE``): "lit" said neither. A host maps the modes onto its own
    attributes (mayatk's ``LightRig``); the emission's level is a ``materials``
    override like any other.
    """

    TYPED = True
    #: How a fixture lights the room (see the class docstring).
    MODES = ("glow", "emissive", "lamp")

    name: str = _field(help="The fixture (what a report and a variant call it).", required=True)
    mode: str = _field(help="How it lights the room.", required=True, choices=MODES)
    meshes: List[str] = _field(
        help="Its emissive meshes (a transform or its shape).",
        required=True,
        default_factory=list,
    )
    lamps: Optional[Dict[str, float]] = _field(
        help="Its lights, by name: the intensity each lights the room at in lamp mode (off in the others)."
    )


@dataclass
class LightRigPayload(SchemaSpec):
    """A light rig: its lights, and the plugs it overrides on nodes it does not
    own. Lengths are centimetres and angles degrees, whatever units the scene
    works in."""

    TYPED = True
    VERSION = 1
    #: The rig library's preset collection (:meth:`library`).
    LIBRARY = "light_rigs"
    #: The folder under the user-config root the library lived in before it was
    #: shared (mayatk's own, until 2026-10): a saved location, spelled as it was.
    _LEGACY_FOLDERS = ("mayatk",)

    version: int = _field(
        help="The payload's version (LightRigPayload.VERSION).", required=True
    )
    group: Optional[str] = _field(help="The scene group its lights live under.")
    lights: List[LightRigLight] = _field(
        help="Its lights.", required=True, nested=[LightRigLight], default_factory=list
    )
    existing: Optional[Dict[str, Dict[str, Any]]] = _field(
        help="{node: {setting or attribute: value}} on scene lights it does not own "
        "(a length in centimetres, an angle in degrees)."
    )
    materials: Optional[Dict[str, Dict[str, Any]]] = _field(
        help="{node: {attribute: value}} on materials (an emission level)."
    )
    fixtures: Optional[List[LightRigFixture]] = _field(
        help="The scene's light fixtures and how each lights the room (LightRigFixture)."
    )
    bake_gain: Optional[float] = _field(
        help="Light units per layer unit a bake multiplied its lights by."
    )
    source: Optional[Dict[str, Any]] = _field(help="Who made it, from what (carried).")
    viewer: Optional[Dict[str, Any]] = _field(help="Viewer settings (carried).")

    @classmethod
    def library(cls) -> "PresetStore":
        """The rig library: ``user_config_root()/light_rigs``, unnamespaced -- a
        rig one host or tool saves is one any other applies (a look match's,
        mayatk's ``LightRig``). Built fresh per call, as
        ``WorkspaceTemplates.store`` is (a cached store would freeze its folder
        against a sandboxed root); the older per-package folder is copied in on
        first use."""
        from pythontk.core_utils.presets.store import PresetStore

        store = PresetStore(cls.LIBRARY)
        store.adopt(
            *(store.user_dir.parent / p / cls.LIBRARY for p in cls._LEGACY_FOLDERS)
        )
        return store
