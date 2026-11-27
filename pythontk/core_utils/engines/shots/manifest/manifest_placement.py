# !/usr/bin/python
# coding=utf-8
"""ManifestPlacement -- the first pose of a helper a placing row is given where the scene has none.

A word that places something (a build sheet's ``teleport``, ``indicate``, ``pip``: its ``places``)
happens at an object, named in the row's ``where``: a viewpoint the user stands on, an arrow's
tip, a picture's camera. That object is the truth, and a person moves it; where a scene has none
yet, a tool makes it once and this says where: a plain first pose, read from bounds alone, with
no line of sight and no guess at what a part is. Declared once here, host-free; each host answers
one question (an object's bounds) and makes the object.
"""

import math
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from pythontk.geo_utils.spot_placement import SpotPlacement

Vector = Tuple[float, float, float]
Bounds = Tuple[Sequence[float], Sequence[float]]


class ManifestPlacement:
    """The plain first pose of a placing row's helper, by what it is placed for."""

    #: What a helper is, by the purpose it is placed for (``ManifestActions.PURPOSES``): a
    #: picture is taken from a camera; a viewpoint and an arrow's tip are plain locators.
    #: A held part's place (``hold``) is its own pose in its take, so none is made for it.
    KINDS: Dict[str, str] = {"stand": "locator", "point": "locator", "view": "camera"}
    #: What a helper rides, by its purpose: an arrow and a camera are children of what
    #: they are for (a door swung open carries them); a viewpoint stands under the delivery.
    RIDES: Dict[str, str] = {"stand": "delivery", "point": "target", "view": "target"}
    #: The first word of a helper's own name, by its purpose (:meth:`helper_name`).
    PREFIXES: Dict[str, str] = {"stand": "VIEW", "point": "ARROW", "view": "CAM"}

    @staticmethod
    def _parts(name: Any) -> list:
        """*name*'s path, its names in order (``|`` or ``\\`` between them)."""
        return [
            part.strip()
            for part in str(name or "").replace("\\", "|").split("|")
            if part.strip()
        ]

    @staticmethod
    def _clean(text: str) -> str:
        return re.sub(r"[^0-9A-Za-z_]+", "_", text).strip("_")

    @classmethod
    def helper_name(
        cls, purpose: str, step: str, target: str, among: Sequence[str] = ()
    ) -> str:
        """The name of the helper a placing row has when it names none (its ``where`` empty):
        ``<KIND>_<step>_<target>`` (``CAM_B02_LCKR_DOOR_GRP``), *target* by its node's own name (a
        path's last part) -- or, where another target *among* the step's ends with that name, by
        the fewest trailing names of its path no other's ends with (``PANEL|LATCH`` beside
        ``DOOR|LATCH``: ``CAM_A01_PANEL_LATCH``) -- anything but a letter, a digit or ``_`` an ``_``.

        One name per row, derived from what the step says, so every tool that reads the sheet
        names the same object (:meth:`helper_names` reads a step's rows so): two pictures in a step
        have two targets and two names; two rows of one purpose and one target in a step are told
        apart only by a ``where`` (the sheet's check says so). Rename it, or use another object, by
        writing its name in ``where``.

        Parameters:
            purpose: What the helper is placed for (:attr:`PREFIXES`).
            step: The row's step's ID.
            target: The row's Target, as the sheet writes it.
            among: The Targets of the step's other rows naming a helper of *purpose*
                (:meth:`helper_names`); empty, the target's own name.

        Raises:
            ValueError: *purpose* places no helper, or *step* or *target* is empty.
        """
        prefix = cls.PREFIXES.get(purpose)
        if prefix is None:
            raise ValueError(
                "no helper is named for %r: one of %s" % (purpose, sorted(cls.PREFIXES))
            )
        parts = cls._parts(target)
        if not step or not parts:
            raise ValueError("a helper's name needs its step and its target")
        others = [path for path in map(cls._parts, among) if path != parts]
        count = 1
        while count < len(parts) and any(
            path[-count:] == parts[-count:] for path in others
        ):
            count += 1
        return "%s_%s_%s" % (
            prefix,
            cls._clean(step),
            cls._clean("_".join(parts[-count:])),
        )

    @classmethod
    def helper_names(
        cls, step: str, rows: Sequence[Mapping[str, Any]]
    ) -> List[Optional[str]]:
        """The name of each of a step's *rows*' own helper (:meth:`helper_name`), ``None`` for a row
        that has none: one that names its object (its ``where``), places nothing a helper is named
        for, or has no Target. Each row's first Target is told apart from the Targets of the step's
        other rows of its purpose that have one.

        Parameters:
            step: The step's ID.
            rows: The step's rows, as the reader made them (``places``, ``assets``, ``settings``).
        """
        from pythontk.core_utils.engines.shots.manifest.manifest_actions import (
            ManifestActions,
        )

        def own(row: Mapping[str, Any]) -> bool:
            named = str(
                (row.get("settings") or {}).get(ManifestActions.WHERE) or ""
            ).strip()
            return (
                row.get("places") in cls.PREFIXES
                and bool(cls._parts((row.get("assets") or [""])[0]))
                and not named
            )

        placing = [row for row in rows if own(row)]
        return [
            cls.helper_name(
                row["places"],
                step,
                row["assets"][0],
                [
                    target
                    for other in placing
                    if other["places"] == row["places"]
                    for target in other["assets"]
                ],
            )
            if own(row)
            else None
            for row in rows
        ]

    #: The placement's own measures (``ptk.SpotPlacement``), named here for what a helper is made with.
    STAND_OFF = SpotPlacement.STAND_OFF
    GAP = SpotPlacement.GAP
    ARROW = SpotPlacement.ARROW
    VIEW_ANGLE = SpotPlacement.VIEW_ANGLE
    MARGIN = SpotPlacement.MARGIN
    LEAST_FRAMED = SpotPlacement.LEAST_FRAMED

    @classmethod
    def outer(cls, target: Bounds, delivery: Optional[Bounds], up: int = 1) -> Vector:
        """The level way out of the delivery at *target* (``ptk.SpotPlacement.outer``)."""
        return SpotPlacement(up=up).outer(target, delivery)

    @classmethod
    def first_pose(
        cls,
        purpose: str,
        target: Bounds,
        delivery: Optional[Bounds] = None,
        up: int = 1,
        unit: float = 1.0,
        floor: Optional[float] = None,
        rays: Any = None,
        owners: Sequence[Any] = (),
    ) -> Tuple[Vector, Vector]:
        """Where a helper placed for *purpose* first stands, and what it faces: ``ptk.SpotPlacement``, the one
        placement every tool uses (``stand``: on the floor outside the target, facing it level; ``point``: an
        arrow's tip by it, above where free; ``view``: a camera that frames it, seeing the most of it).

        Parameters:
            purpose: ``stand``, ``point`` or ``view``.
            target: What the row is about: its bounds ``(low, high)``, in the scene's units, as the scene stands at
                the row's step.
            delivery: The whole delivery's bounds: its middle says which side is outer. ``None``: the target's own.
            up: The up axis's index (1: ``Y``; 2: ``Z``).
            unit: The scene's units in a metre (100 in a centimetre scene).
            floor: The floor's height on the up axis, where a host knows it. ``None``: the delivery's lowest point.
            rays: What the host casts rays through (``ptk.ProbeRays``): what is seen decides among spots. ``None``:
                bounds alone.
            owners: The ray set's owners that are the target (a line that reaches them sees it).

        Returns:
            ``(position, aim)``: where the helper stands and the point it faces, in the scene's units.

        Raises:
            ValueError: *purpose* places no helper (:attr:`KINDS`).
        """
        if purpose not in cls.KINDS:
            raise ValueError(
                "no helper is placed for %r: one of %s" % (purpose, sorted(cls.KINDS))
            )
        place = SpotPlacement(rays=rays, up=up, unit=unit)
        if purpose == "stand":
            return place.stand(target, delivery, floor=floor, owners=owners)
        if purpose == "view":
            return place.view(target, delivery, owners=owners)
        return place.point(target, delivery, owners=owners)

    @classmethod
    def focal_length(cls, aperture_height: float) -> float:
        """The focal length that gives a camera :attr:`VIEW_ANGLE` over a film (or sensor)
        *aperture_height* high, in the aperture's own unit."""
        return aperture_height * 0.5 / math.tan(math.radians(cls.VIEW_ANGLE) * 0.5)
