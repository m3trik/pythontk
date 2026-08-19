# !/usr/bin/python
# coding=utf-8
"""Plan, assess, and apply map (texture) optimizations.

Split out of ``ImgUtils`` so the decision branches consumed by both
:meth:`MapOptimizer.optimize_map` and :meth:`MapOptimizer.assess`
live in a single planner. Prevents drift between "would change" predictions
and actual mutations.

Architecture:
    plan(image, **opts) -> [Op, ...]   # pure decisions, no IO
    apply(image, plan, ...) -> Image   # executes ops via ImgUtils helpers
    optimize_map(path, ...) -> str     # orchestrator: load + plan + apply + save
    assess(path, ...) -> dict          # wraps plan() with image read + reporting
"""
from __future__ import annotations

import os
import math

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional

try:
    from PIL import Image
except ImportError as e:
    print(f"# ImportError: {__file__}\n\t{e}")
    Image = None  # type: ignore

# From this package:
from pythontk.core_utils.help_mixin import HelpMixin
from pythontk.file_utils._file_utils import FileUtils
from pythontk.img_utils._img_utils import ImgUtils
from pythontk.core_utils.engines.textures.map_factory import MapFactory
from pythontk.core_utils.engines.textures.map_registry import MapRegistry
from pythontk.core_utils.engines.textures.output_template import (
    DeliveryBudget,
    OutputSpec,
    OutputTemplates,
)


# Modes that are a target mode's channel layout at HIGHER precision.
# Coercing them down to the 8-bit target discards precision the file
# deliberately carries — the profile writer itself authors 16-bit grayscale
# for a SUBSET of L-target map types (OutputSpec bit_depth 16 for
# Height/Displacement/Bump only — Roughness/AO/Metallic stay 8-bit), so the
# dry-run twin flagging its own 16-bit output for an L-coercion made assess
# disagree with optimize_map forever (probe-proven 2026-08-14: a staged
# I;16 height map assessed "recommended" on every subsequent run — a paired
# task/check built on the twins could then never converge).
#
# Tolerance is gated on the RESOLVED OutputSpec's bit_depth (see plan()'s
# ``high_precision_ok``), not on the target mode alone — folding it
# unconditionally into ``_MAP_TYPE_TOLERATED`` made every L-target map type
# tolerate 16-bit input, so an I;16 Roughness source under an 8-bit spec
# shipped as a 16-bit PNG with no warning (probe-proven 2026-08-14).
_HIGH_PRECISION_EQUIV: Dict[str, Tuple[str, ...]] = {
    "L": ("I", "I;16"),
}

# What each CONTAINER actually stores for a high-precision single-channel
# source — measured (save + reopen), never derived from
# ``ImgUtils._CONTAINER_MODE_FALLBACKS``. That table lists only the modes whose
# stored form DIFFERS, so every container missing from it reads as "keeps
# I;16", and gating the tolerance on the resolved bit depth alone therefore let
# a 16-bit mode reach every container at once (probe-proven 2026-08-14):
# ``optimize_map(<16-bit Height>, output_type="dds")`` raised
# ``OSError: cannot write mode I;16 as DDS`` while ``assess`` of the same call
# predicted a clean no-op, and the table's ``"tga": {"I;16": "RGB"}`` row —
# right for an unknown image — widened a linear height map to 24-bit RGB
# (+4434%) where "L" is the correct reduction for a single-channel map.
#
# A container ABSENT here gets no tolerance at all, so the plan coerces down to
# the map type's own target mode before the container fallback can ever see the
# 16-bit mode. Values are the mode the container writes, which makes PNG's
# ``"I" -> "I;16"`` a NARROWING rather than a flatten: Pillow's PNG writer
# stores mode "I" as 16-bit anyway (a path it deprecates for removal in Pillow
# 13), so naming it keeps every bit the container can hold AND makes the dry
# run predict the mode that actually lands on disk.
_HIGH_PRECISION_CONTAINERS: Dict[str, Dict[str, str]] = {
    "png": {"I": "I;16", "I;16": "I;16"},
    "tif": {"I": "I", "I;16": "I;16"},
    "tiff": {"I": "I", "I;16": "I;16"},
    # Float containers (routed to the cv2 writer): they have no 8-bit form to
    # truncate to, so the integer modes pass through at full range.
    "exr": {"I": "I", "I;16": "I;16"},
    "hdr": {"I": "I", "I;16": "I;16"},
}

# Accepted ``pot_mode`` values. Matched as ``== "down"`` or else round(), so
# every unrecognized string — "downward", "Down", a typo — silently took the
# nearest snap, which is the one behavior a budget caller passes pot_mode to
# avoid (nearest GROWS everything in the upper half of a POT band).
_POT_MODES: Tuple[str, ...] = ("nearest", "down")

# Relative aspect-ratio change below which a POT snap is NOT worth a warning.
# The snap derives the short edge as an integer, so a sub-pixel remainder is
# always present (683 for a 682.67 ideal); warning about that dust would train
# the reader straight past the case that matters — an extreme ratio whose
# derived edge hits the min-1 floor and really is reshaped.
_ASPECT_DRIFT_TOLERANCE: float = 0.01

# Map-type-driven mode coercion rules — the BASE tolerance, without the
# high-precision modes folded in. Mirrors the tolerated-mode lists in the
# original optimize_map body — defined once here so both plan() and apply()
# reference the same source. plan() adds ``_HIGH_PRECISION_EQUIV`` on top
# only when the resolved spec's bit depth supports it.
_MAP_TYPE_TOLERATED: Dict[str, Tuple[str, ...]] = {
    "RGB": ("RGB", "P", "L"),
    "RGBA": ("RGBA", "PA", "P"),
    "L": ("L", "P"),
}

# Legacy map-type-key heuristics for keys without a MapRegistry entry. Kept
# verbatim from optimize_map's pre-extraction fallback so call-site
# behavior is preserved exactly.
_LEGACY_MAP_KEY_RULES: Dict[Tuple[str, ...], Tuple[str, Tuple[str, ...]]] = {
    ("Normal", "Normal_OpenGL", "Normal_DirectX"): ("RGB", ("RGB",)),
    ("MSAO", "MaskMap"): ("RGBA", ("RGBA",)),
    ("ORM",): ("RGB", ("RGB", "RGBA")),
}
_LEGACY_GRAYSCALE_KEYS = (
    "Ambient_Occlusion",
    "Roughness",
    "Metallic",
    "Smoothness",
    "Height",
    "Bump",
)


@dataclass
class Op:
    """One operation in an optimization plan.

    ``kind`` drives which ImgUtils helper :func:`apply` dispatches to. Each
    op carries a human-readable ``description`` so :meth:`assess` can surface
    the same wording without re-deriving it.
    """

    kind: str
    description: str
    params: Dict[str, Any] = field(default_factory=dict)


class MapOptimizer(HelpMixin):
    """Plan, assess, and apply map (texture) optimizations.

    All decision logic lives in :meth:`plan` — :meth:`apply` is a thin
    dispatcher that mutates the image according to the plan's ops, never
    making its own mode/size choices. :meth:`optimize_map` orchestrates
    load + plan + apply + save; :meth:`assess` is the read-only twin.

    Single source of truth: ``apply`` does not call ``ImgUtils.set_bit_depth``
    so its idiosyncratic ``bit_depth_mapping`` middle step can't introduce
    drift between predicted (``assess``) and actual (``optimize_map``)
    outputs. ``set_bit_depth`` remains available for direct callers.
    """

    @staticmethod
    def _fit_within(width: int, height: int, max_size: int) -> Tuple[int, int]:
        """Cap the longest edge at *max_size*, deriving the other from aspect."""
        if width >= height:
            return max_size, max(1, round(height * max_size / width))
        return max(1, round(width * max_size / height)), max_size

    @staticmethod
    def _snap_pot(
        width: int,
        height: int,
        mode: str = "nearest",
        max_size: Optional[int] = None,
    ) -> Tuple[int, int]:
        """Snap the LONGEST edge to a power of two, keeping the source aspect.

        Snapping each axis independently is what a naive "force POT" does, and
        it reshapes the map rather than resizing it: 1024x768 — already POT on
        its long edge and already inside every ceiling — came out 1024x512,
        because 768 floors to 512 on its own. A POT rule states how BIG a
        texture may be, not what shape it is, so only the long edge is snapped
        and the short one is derived from the ORIGINAL ratio (nearest integer,
        floored at 1). Making BOTH edges POT is a square/POT-both target's
        requirement; no :class:`DeliveryBudget` flag expresses that today, so
        it is not assumed here — the residual non-POT edge is reported by
        ``DeliveryBudget.check`` instead of being paid for with the aspect.

        ``mode="down"`` never grows an edge — what a *budget* needs, since
        rounding to nearest inflates every dimension in the upper half of a
        band. ``max_size`` is a separate, harder rule: a stated ceiling must
        survive the snap, so a long edge that crossed it is halved back under —
        and the derived edge rides the same factor, so the shape still holds.
        Nearest still wins wherever it stays legal.

        Raises:
            ValueError: *mode* is neither of :data:`_POT_MODES`. Validated at
                the single point of use rather than at each public entry, so
                every route in (``optimize_map`` / ``assess`` / ``plan``) is
                covered by one check. None/omitted stays the documented
                "nearest" default.
        """
        if mode is not None and mode not in _POT_MODES:
            raise ValueError(
                f"Unknown pot_mode {mode!r}: expected one of {_POT_MODES} "
                f"(None/omitted = 'nearest')."
            )
        snap = math.floor if mode == "down" else round

        long_edge = max(width, height)
        snapped = max(1, 2 ** int(snap(math.log2(long_edge))))
        while max_size and snapped > max_size and snapped > 1:
            snapped //= 2

        derived = max(1, round(min(width, height) * snapped / long_edge))
        return (snapped, derived) if width >= height else (derived, snapped)

    @staticmethod
    def _aspect_shift(before: Tuple[int, int], after: Tuple[int, int]) -> float:
        """Relative change in aspect ratio between two sizes (0.0 = identical).

        Relative rather than absolute: a 0.1 drift is nothing on a 16:1 banner
        and a visible squash on a square, so only the fraction is comparable
        against one tolerance.
        """
        (bw, bh), (aw, ah) = before, after
        if min(bw, bh, aw, ah) <= 0:
            return 0.0
        return abs((aw / ah) - (bw / bh)) / (bw / bh)

    #: ``resolve_size_clamp`` sentinel: clamp to the active template's own
    #: :class:`DeliveryBudget` (``enforce_budget``) rather than to a stated
    #: pixel ceiling. Negative so it stays TRUTHY -- 0 / None already mean "no
    #: clamp" -- and can never collide with a real pixel dimension. Same
    #: convention as the Map Converter's ``CLAMP_TARGET``.
    SIZE_CLAMP_TEMPLATE = -1

    @classmethod
    def resolve_size_clamp(
        cls,
        max_size: Any,
        template: Optional[str] = None,
        logger: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Turn a user-facing "max size" mode into :meth:`assess` / :meth:`optimize_map` kwargs.

        The one place a size dial is interpreted, so every caller offering
        one -- the mayatk and blendertk scene exporters' Max Texture Size
        rows, and anything else that grows one -- answers it identically
        instead of copying the rule:

        - falsy / ``"OFF"`` -- no clamp; a template's budget stays ADVISORY
          (``assess`` reports it in ``warnings`` and nothing resamples).
        - a positive int (or its string form, as a hand-edited preset sends)
          -- hard longest-edge ceiling (``max_size``).
        - :attr:`SIZE_CLAMP_TEMPLATE` -- enforce *template*'s own
          :class:`DeliveryBudget` size ceiling. With no template active there
          is no budget to enforce, so this is a no-op. The budget's POT rule
          is deliberately NOT adopted (``force_pot=False``): a ceiling only
          ever shrinks and keeps aspect, where snapping each axis
          independently would reshape a non-square map.

        Parameters:
            max_size: The dial's value, in any of the forms above.
            template: Active output template (profile) name, or None.
            logger: Optional logger warned when *max_size* is unparseable;
                the value is ignored either way (no clamp), never guessed.

        Returns:
            Kwargs for ``assess`` / ``optimize_map`` -- ``max_size``, or
            ``enforce_budget`` + ``force_pot=False``. Empty when no clamp
            applies. A ceiling only affects a map larger than it;
            ``optimize_map`` never grows one.
        """
        if not max_size or isinstance(max_size, bool) or str(max_size).upper() == "OFF":
            return {}
        try:
            value = int(max_size)
        except (TypeError, ValueError):
            if logger is not None:
                logger.warning(
                    f"Invalid max texture size {max_size!r} — no size clamp applied."
                )
            return {}
        if value == cls.SIZE_CLAMP_TEMPLATE:
            return {"enforce_budget": True, "force_pot": False} if template else {}
        if value <= 0:
            return {}
        return {"max_size": value}

    @classmethod
    def describe_size_clamp(
        cls,
        max_size: Any,
        template: Optional[str] = None,
        logger: Optional[Any] = None,
    ) -> str:
        """Human-readable form of :meth:`resolve_size_clamp`, for log lines.

        Empty string when no clamp applies, so a caller can splice it into a
        sentence without branching.
        """
        clamp = cls.resolve_size_clamp(max_size, template, logger=logger)
        if not clamp:
            return ""
        if clamp.get("enforce_budget"):
            size = getattr(OutputTemplates.budget(template), "max_size", None)
            limit = f"{size} px" if size else "no size limit"
            return f"clamped to the template's budget ({limit})"
        return f"clamped to {clamp['max_size']} px"

    @classmethod
    def plan(
        cls,
        image: "Image.Image",
        max_size: Optional[int] = None,
        force_pot: bool = False,
        optimize_bit_depth: bool = True,
        map_type_key: Optional[str] = None,
        allow_palette: bool = False,
        pot_mode: str = "nearest",
        output_profile: Optional[str] = None,
        output_type: Optional[str] = None,
    ) -> List[Op]:
        """Return the ordered list of operations :meth:`apply` would run.

        Pure function: no file IO, no mutation. The planner tracks a
        ``logical`` mode/size as each op would change them, so downstream
        decisions (e.g. mode coercion before resize) see the post-prior-op
        state — same as if optimize_map were executing.

        Parameters:
            image: Source image (only its size/mode/info are read).
            max_size: Max edge length for the resize step. None disables.
            force_pot: Snap to a power-of-two if not already POT. The
                LONGEST edge is snapped and the other is derived from the
                source aspect (see :meth:`_snap_pot`), so the map is
                resized, never reshaped; an op whose snap still shifts the
                ratio carries a ``warning`` param (see
                :meth:`_plan_warnings`).
            optimize_bit_depth: Enable the strict-mode + wide-gamut fallback
                step (formerly delegated to set_bit_depth).
            map_type_key: Canonical map-type key from
                ``MapFactory.resolve_map_type(..., key=True)``. Drives the
                map-type mode coercion step.
            allow_palette: Preserve paletted images instead of upcasting.
            pot_mode: How ``force_pot`` snaps — ``"nearest"`` (a caller asking
                for POT wants the closest match) or ``"down"``. A *budget*
                always passes ``"down"``: rounding to nearest makes every
                dimension in the upper half of a POT band grow (1536 -> 2048,
                +78% pixels), so a delivery budget could inflate the very asset
                it exists to shrink, and then report itself as satisfied.
                Anything else raises ``ValueError`` (see :meth:`_snap_pot`) —
                an unrecognized value used to fall through to "nearest", which
                is the one snap a caller passing this is trying to avoid.
            output_profile: Workflow profile (a ``WF`` key) used only to
                resolve the map type's :class:`OutputSpec` bit depth, which
                gates the high-precision (I/I;16) mode-coercion tolerance —
                see ``stored_high_precision`` below. None (default) resolves
                against :attr:`OutputTemplates.DEFAULT`, which already
                distinguishes 16-bit Height/Displacement/Bump from every
                other 8-bit L-target map type.
            output_type: Container the run will actually write (extension, with
                or without a dot) — the *other* half of that tolerance gate: a
                16-bit spec is only worth honoring in a container that can
                store the mode (see :data:`_HIGH_PRECISION_CONTAINERS`). Both
                twins resolve it before calling; None (default) falls back to
                the resolved spec's own container, the honest reading for a
                direct caller who named neither.

        Returns:
            list[Op]: Ordered ops; empty when no changes would be applied.
        """
        ops: List[Op] = []
        width, height = image.size
        mode = image.mode

        # The high-precision (I/I;16) tolerance below must track the
        # RESOLVED spec's bit depth, not the target mode alone — Roughness/
        # AO/Metallic share the "L" target with Height/Displacement/Bump but
        # only the latter get a 16-bit OutputSpec. OutputTemplates.resolve
        # always returns a spec (falls back to DEFAULT when output_profile
        # is None/unknown), so this is safe even outside a profiled run.
        spec = (
            OutputTemplates.resolve(map_type_key, output_profile)
            if map_type_key
            else None
        )
        # ...and it must track the CONTAINER being written just as closely: a
        # 16-bit spec buys nothing in a container that cannot store the mode,
        # and honoring it there crashed the dds writer while the dry run
        # predicted a clean no-op. Empty = no tolerance, so the coercion below
        # falls back to the map type's own target mode.
        stored_high_precision: Dict[str, str] = (
            _HIGH_PRECISION_CONTAINERS.get(
                (output_type or (spec.ext if spec else "") or "").lower().lstrip("."),
                {},
            )
            if spec and spec.bit_depth and spec.bit_depth >= 16
            else {}
        )

        # --- Pre-compute resize and POT decisions (will_resize gates depalettize)
        # The resize caps the LONGEST edge and derives the other from the source
        # aspect: driving both edges from one scalar squashed every non-square
        # map to a square (harmless while only an explicit max_size reached it,
        # automatic for every budgeted profile once enforce_budget existed).
        resize_to: Optional[Tuple[int, int]] = (
            cls._fit_within(width, height, max_size)
            if max_size and max(width, height) > max_size
            else None
        )
        # max_size is passed into the snap, not pre-applied as a mode: a stated
        # ceiling must survive POT rounding whether or not the resize above
        # fired. Keying off "did a resize happen" misses the case where the
        # source already fits but the snap crosses the limit on its own (200
        # under a 250 ceiling snaps to 256).
        pot_target: Optional[Tuple[int, int]] = None
        if force_pot and width > 0 and height > 0:
            rw, rh = resize_to if resize_to else (width, height)
            pot = cls._snap_pot(rw, rh, pot_mode, max_size)
            if (rw, rh) != pot:
                pot_target = pot
        will_resize = resize_to is not None or pot_target is not None

        # --- 1. Depalettize before resize (preserves high-quality resampling)
        if will_resize and mode in ("P", "PA"):
            new_mode = (
                "RGBA"
                if (mode == "PA" or "transparency" in image.info)
                else "RGB"
            )
            ops.append(
                Op(
                    kind="depalettize",
                    description=f"Depalettize for resize: {mode} -> {new_mode}",
                    params={"target_mode": new_mode},
                )
            )
            mode = new_mode

        # --- 2. Map-type mode coercion (lines 1395+ of the original body)
        map_def = MapRegistry().get(map_type_key) if map_type_key else None
        if map_def and getattr(map_def, "mode", None):
            target_mode = map_def.mode
            equivalents = _HIGH_PRECISION_EQUIV.get(target_mode, ())
            # Only the modes this container stores AS THEMSELVES are tolerated
            # in place; the rest still need an op, just not necessarily one
            # that lands on the 8-bit target (below).
            tolerated = _MAP_TYPE_TOLERATED.get(target_mode, (target_mode,)) + tuple(
                m for m in equivalents if stored_high_precision.get(m) == m
            )
            if mode not in tolerated:
                # A high-precision source the container stores at a DIFFERENT
                # high-precision mode is narrowed to that one rather than
                # flattened to the 8-bit target: PNG writes mode "I" as 16-bit
                # regardless, so naming "I;16" keeps the precision the spec
                # paid for and makes the dry run predict the mode that really
                # lands on disk.
                coerce_to = (
                    stored_high_precision.get(mode) if mode in equivalents else None
                ) or target_mode
                ops.append(
                    Op(
                        kind="mode_coerce",
                        description=(
                            f"Mode (map_type={map_type_key}): "
                            f"{mode} -> {coerce_to}"
                        ),
                        params={"target_mode": coerce_to},
                    )
                )
                mode = coerce_to
        elif map_type_key:
            # Legacy fallback for keys not in the registry.
            for keys, (target, tolerated) in _LEGACY_MAP_KEY_RULES.items():
                if map_type_key in keys and mode not in tolerated:
                    ops.append(
                        Op(
                            kind="mode_coerce",
                            description=(
                                f"Mode (legacy {map_type_key}): "
                                f"{mode} -> {target}"
                            ),
                            params={"target_mode": target},
                        )
                    )
                    mode = target
                    break
            else:
                if map_type_key in _LEGACY_GRAYSCALE_KEYS and mode == "P":
                    ops.append(
                        Op(
                            kind="mode_coerce",
                            description=(
                                f"Mode (legacy grayscale {map_type_key}): "
                                f"P -> L"
                            ),
                            params={"target_mode": "L"},
                        )
                    )
                    mode = "L"

        # --- 3. Resize
        if resize_to is not None:
            rw, rh = resize_to
            ops.append(
                Op(
                    kind="resize",
                    description=(
                        f"Resize: {width}x{height} -> "
                        f"{rw}x{rh} (exceeds max_size={max_size})"
                    ),
                    params={"size": resize_to},
                )
            )
            width, height = rw, rh

        # --- 4. Force POT (recompute against current dims, post-resize)
        if force_pot and width > 0 and height > 0:
            pw, ph = cls._snap_pot(width, height, pot_mode, max_size)
            if (width, height) != (pw, ph):
                params: Dict[str, Any] = {"size": (pw, ph)}
                # The snap derives the short edge from the source ratio, so a
                # shift past the tolerance means the min-1 floor really did
                # reshape an extreme ratio. That is the surprising half of a
                # POT rule (a ceiling is not surprising), and the reason a
                # reshape went unnoticed before is that nothing ever said it
                # out loud — ``DeliveryBudget.check`` runs against the
                # POST-snap size, so it can never see one.
                if (
                    cls._aspect_shift((width, height), (pw, ph))
                    > _ASPECT_DRIFT_TOLERANCE
                ):
                    params["warning"] = (
                        f"POT snap changed the aspect ratio: {width}x{height} "
                        f"({width / height:.3f}) -> {pw}x{ph} ({pw / ph:.3f})"
                    )
                ops.append(
                    Op(
                        kind="force_pot",
                        description=f"Force POT: {width}x{height} -> {pw}x{ph}",
                        params=params,
                    )
                )
                width, height = pw, ph

        # --- 5. Strict-mode enforcement (post step-2 cleanups).
        # Mirrors set_bit_depth's two intentional branches: strict palette
        # upcast when a map_type target exists, and the wide-gamut fallback.
        # The bit_depth_mapping middle step in the original set_bit_depth is
        # deliberately NOT mirrored — it's the quirk that caused drift, and
        # bypassing it here makes plan the single source of truth.
        if optimize_bit_depth and (mode != "P" or not allow_palette):
            sb_target = (
                MapRegistry().get_map_modes().get(map_type_key)
                if map_type_key
                else None
            )
            sb_target_mode: Optional[str] = None
            sb_reason: Optional[str] = None

            if (
                sb_target
                and mode != sb_target
                and not (
                    stored_high_precision.get(mode) == mode
                    and mode in _HIGH_PRECISION_EQUIV.get(sb_target, ())
                )
            ):
                # Catches the strict-palette upcast for tolerated-but-non-
                # target inputs (e.g. P -> RGB when allow_palette is False).
                # A high-precision same-layout mode (I;16 where the target is
                # L) is NOT coerced only when the resolved spec is itself
                # 16-bit AND the container stores that mode: strict mode
                # exists to normalize channel layout, and flattening 16-bit
                # data to 8 would silently discard the precision a 16-bit spec
                # just paid for — but an 8-bit spec (Roughness/AO/Metallic)
                # never asked for that precision, and a container that cannot
                # hold it (dds/tga/jpg/webp/ktx2) makes honoring it a crash or
                # a 24-bit widening rather than a saving.
                sb_target_mode = sb_target
                sb_reason = f"Strict mode (map_type={map_type_key})"
            elif mode in ("HSV", "LAB", "CMYK", "YCbCr"):
                sb_target_mode = "RGBA" if mode == "CMYK" else "RGB"
                sb_reason = f"Unsupported mode {mode}"

            if sb_target_mode and sb_target_mode != mode:
                ops.append(
                    Op(
                        kind="mode_coerce",
                        description=(
                            f"{sb_reason}: {mode} -> {sb_target_mode}"
                        ),
                        params={"target_mode": sb_target_mode},
                    )
                )
                mode = sb_target_mode

        return ops

    @staticmethod
    def _plan_warnings(plan: List[Op]) -> List[str]:
        """Advisories the plan's own ops raise, in plan order.

        Distinct from an op's ``description``: that says what the run WILL do,
        this says what is surprising about doing it. The wording lives on the
        op — as with :class:`Op` descriptions and ``DeliveryBudget.check``, the
        dry-run twin and the real run must say the same thing, and two copies
        of a sentence are two chances to drift.
        """
        return [op.params["warning"] for op in plan if op.params.get("warning")]

    @staticmethod
    def project(
        plan: List[Op],
        width: int,
        height: int,
        mode: str,
    ) -> Tuple[int, int, str]:
        """Replay ``plan``'s op params to get the post-apply size and mode.

        Pure arithmetic over the plan — no pixel work — so a caller that only
        wants to *show* the outcome (a dry run, a report row) doesn't pay for
        a resize. Every op that changes size or mode carries the resulting
        value in its params, making this a replay rather than a second copy
        of :meth:`plan`'s decision logic.

        Parameters:
            plan: Ops from :meth:`plan`.
            width, height, mode: The source image's starting state.

        Returns:
            tuple: ``(width, height, mode)`` after the plan would run.
        """
        for op in plan:
            if op.kind in ("depalettize", "mode_coerce"):
                mode = op.params.get("target_mode", mode)
            elif op.kind in ("resize", "force_pot"):
                size = op.params.get("size")
                if size is not None:
                    width, height = size
        return width, height, mode

    @classmethod
    def apply(
        cls,
        image: "Image.Image",
        plan: List[Op],
    ) -> "Image.Image":
        """Execute ``plan`` against ``image``. Returns the mutated image.

        Pure dispatcher: each branch performs the direct PIL mutation
        implied by the op's params. Apply never re-decides mode or size —
        ``plan`` has already made those choices.
        """
        for op in plan:
            if op.kind == "depalettize":
                image = ImgUtils.depalettize_image(image)
            elif op.kind == "mode_coerce":
                target = op.params["target_mode"]
                if image.mode != target:
                    image = cls._coerce_mode(image, target)
            elif op.kind in ("resize", "force_pot"):
                # Both resize to the size the PLAN chose. force_pot must not
                # call ImgUtils.ensure_pot: that re-derives its own nearest-POT
                # target, which is exactly the apply-re-decides-a-size drift
                # this split exists to prevent (and it would silently ignore
                # a budget's round-down).
                w, h = op.params["size"]
                image = ImgUtils.resize_image(image, w, h)
        return image

    @staticmethod
    def _coerce_mode(image: "Image.Image", target: str) -> "Image.Image":
        """Convert *image* to *target*, RESCALING a >8-bit integer source.

        Pillow implements ``I;16``/``I`` -> ``L`` as a CLIP at 255, not a range
        rescale: a smooth 0..65535 ramp comes back as two values, 99.6% of them
        pure white (probe-proven 2026-08-14). The plan's fallback to a map
        type's 8-bit target mode — what every container that cannot store the
        16-bit mode now gets — is meant to be a precision *reduction*; without
        this it is a destroyed map.

        The reduction itself belongs to ``ImgUtils.convert_i_to_l``, which
        already owns this rule package-wide. Rolling a private ``point``-based
        rescale here instead cost a crash: ``point`` refuses byte-order-
        qualified modes, so ``I;16B`` — what Pillow hands back for a
        big-endian 16-bit TIFF — raised ``ValueError: point operation not
        supported for this mode``. The shared helper goes through numpy and
        reads every byte order.

        Mode "F" is the same trap one dtype over, and the worst of the family:
        a float map's data lives in 0..1, so Pillow's truncation sends every
        texel below 1.0 to 0 — a 0..1 height ramp optimized to an essentially
        black 84-byte PNG, reported as a successful -99% pass. It goes through
        ``convert_f_to_l``, whose clamp-don't-normalize rule keeps the twins in
        agreement (both still name "L") where refusing the mode outright would
        make the writer raise on a prediction ``assess`` had called clean.

        Narrowing to another high-precision mode ("I" -> "I;16" for PNG) stays
        a plain convert — no range change is involved.
        """
        if target in ("I", "I;16") or not (
            image.mode == "F" or image.mode == "I" or image.mode.startswith("I;")
        ):
            return image.convert(target)
        reduced = (
            ImgUtils.convert_f_to_l(image)
            if image.mode == "F"
            else ImgUtils.convert_i_to_l(image)
        )
        return reduced if target == "L" else reduced.convert(target)

    @classmethod
    def _resolve_profile(
        cls,
        output_profile: Optional[str],
        map_type_key: Optional[str],
        output_type: Optional[str],
        max_size: Optional[int],
        force_pot: Optional[bool],
        enforce_budget: bool,
    ) -> Tuple[
        Optional[OutputSpec],
        Optional[DeliveryBudget],
        Optional[str],
        Optional[int],
        bool,
        str,
    ]:
        """Apply an output profile's two tiers to a call's arguments.

        Both :meth:`optimize_map` and its dry-run twin :meth:`assess` have to
        read a profile the same way or the prediction stops describing the run —
        the same reason the decision branches live in :meth:`plan`. So the
        precedence rules live here once:

        - **hard tier** — the profile's :class:`OutputSpec` supplies the
          container, but only where the caller named none.
        - **advisory tier** — the profile's :class:`DeliveryBudget` supplies
          ``max_size`` / ``force_pot`` *only* when ``enforce_budget`` is set, and
          even then an explicit value from the caller still wins. Both use
          ``is None`` to mean "caller said nothing": ``force_pot or
          budget.force_pot`` could not tell an explicit ``False`` from unset, so
          the documented opt-out did not work.

        Returns:
            tuple: ``(spec, budget, output_type, max_size, force_pot, pot_mode)``.
            ``spec`` and ``budget`` are None when no profile is active.
            ``pot_mode`` is ``"down"`` only when the POT snap came from the
            budget — a budget must never grow an asset.
        """
        pot_mode = "nearest"
        if not output_profile:
            return None, None, output_type, max_size, bool(force_pot), pot_mode

        spec = OutputTemplates.resolve(map_type_key, output_profile)
        if not output_type:
            output_type = spec.ext

        budget = OutputTemplates.budget(output_profile)
        if enforce_budget:
            if max_size is None:
                max_size = budget.max_size
            if force_pot is None:
                force_pot = budget.force_pot
                pot_mode = "down"

        return spec, budget, output_type, max_size, bool(force_pot), pot_mode

    @classmethod
    def resolve_quality(
        cls,
        lossy_quality: Optional[int],
        map_type_key: Optional[str],
        output_type: Optional[str],
        spec: Optional[OutputSpec] = None,
    ) -> Tuple[Optional[int], Optional[str]]:
        """Decide the lossy quality a run may actually use, and why not.

        Three independent conditions have to hold before a pixel is degraded, and
        each veto is *reported* rather than silently applied — a caller who asked
        for lossy and got lossless must be able to tell, otherwise the option
        looks broken when it is in fact protecting them:

        1. Someone asked. ``lossy_quality`` (call-level) outranks
           ``spec.quality`` (profile-level) so a UI toggle beats a preset.
        2. The container can express it — :attr:`ImgUtils.LOSSY_FORMATS`.
           Requesting q90 into a PNG is a no-op worth naming.
        3. The map type survives it — :meth:`MapRegistry.is_lossy_safe`. This
           is the one that matters: it is what stops a batch run from
           destroying every normal map in a folder because the operator set one
           dropdown.

        Both :meth:`optimize_map` and :meth:`assess` route through here, for the
        same reason the decision branches live in :meth:`plan` — a dry run that
        predicted lossy while the real run refused it would be worse than no
        prediction at all.

        Parameters:
            lossy_quality: Caller's explicit request, or None.
            map_type_key: Canonical map-type key driving the safety gate.
            output_type: Target container extension (with or without a dot).
            spec: Active profile's :class:`OutputSpec`, when one is resolved.

        Returns:
            tuple: ``(quality, skip_reason)``. ``quality`` is None when the run
            must stay lossless; ``skip_reason`` is a human-readable sentence
            when a request was *declined*, and None when none was made.
        """
        requested = lossy_quality if lossy_quality is not None else (
            spec.quality if spec else None
        )
        ext = (output_type or "").lower().lstrip(".")

        if ext == "ktx2":
            # Basis quality is a codec dial, not a container codec's: ETC1S
            # carries it (mapped onto qlevel by the encoder); UASTC's tier is
            # fixed. Gated by the same rule as the container codecs — ETC1S
            # is exactly as unsafe for data maps as WebP/JPEG are, and those
            # maps encode UASTC regardless (see :meth:`resolve_compression`).
            if requested is None:
                return None, None
            if MapRegistry().is_lossy_safe(map_type_key):
                return int(requested), None
            return None, (
                f"lossy quality {requested} refused for map type "
                f"'{map_type_key or 'unknown'}': data maps encode UASTC at the "
                f"encoder's fixed quality tier — the ETC1S quality dial only "
                f"applies to unpacked sRGB maps"
            )

        if requested is None:
            # Nobody named a quality -- but a container with no lossless mode
            # (JPEG) degrades the pixels anyway, so *choosing it* is the
            # request. Without this the gate only guarded the path nobody
            # takes: an explicit quality on a normal map was refused aloud
            # while picking ".jpg" off a format menu wrote it at the q95
            # default with no warning at all. There is nothing to veto here --
            # JPEG cannot be written losslessly and the caller asked for JPEG
            # -- so this reports rather than refuses, which is the honest
            # outcome and still lets a batch run be audited.
            if ext in ImgUtils.ALWAYS_LOSSY_FORMATS and not MapRegistry().is_lossy_safe(
                map_type_key
            ):
                return None, (
                    f"'{ext}' has no lossless mode, so map type "
                    f"'{map_type_key or 'unknown'}' is being written lossy: only "
                    f"unpacked sRGB maps (base color, emissive) survive a lossy "
                    f"codec — prefer png/webp for this map"
                )
            return None, None

        if ext and ext not in ImgUtils.LOSSY_FORMATS:
            return None, (
                f"lossy quality {requested} ignored: '{ext}' is a lossless "
                f"container (use {'/'.join(ImgUtils.LOSSY_FORMATS)})"
            )

        if not MapRegistry().is_lossy_safe(map_type_key):
            return None, (
                f"lossy quality {requested} refused for map type "
                f"'{map_type_key or 'unknown'}': only unpacked sRGB maps "
                f"(base color, emissive) survive a lossy codec — written lossless"
            )

        return int(requested), None

    @classmethod
    def resolve_compression(
        cls,
        map_type_key: Optional[str],
        output_type: Optional[str],
        spec: Optional[OutputSpec] = None,
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Resolve the GPU compression (and its colorspace label) for one map.

        For every container except ``ktx2`` this is a pass-through of the
        profile's ``OutputSpec.compression`` (the DDS path). For ``ktx2`` it
        owns the per-map Basis codec decision:

        - **Nothing requested** → derived from the registry:
          :meth:`MapRegistry.is_lossy_safe` maps (unpacked sRGB — base color,
          emissive) take **ETC1S**, the low-bitrate palettised codec whose
          error profile suits perceptual color; everything else — normals,
          ORM/packed masks, linear data, and unknown map types — takes
          **UASTC**, the high-quality codec. The registry gate exists for
          exactly this distinction, so the codec choice and the lossy-container
          gate can never disagree.
        - **ETC1S requested on a non-lossy-safe map** → upgraded to UASTC and
          *reported*, mirroring :meth:`resolve_quality`'s refusal semantics: a
          preset must not band every normal map in a batch because one field
          said ETC1S.
        - **Unknown codec name** → ``ValueError`` — a template/config error,
          raised identically in the real run and the dry run.

        The colorspace label rides along because only this decision point has
        the map type in hand: the registry's ``color_space`` becomes the KTX2
        transfer-function label ("sRGB"/"Linear"; unknown map types label sRGB
        — with UASTC derived above, a mislabel costs sampling correctness on a
        map nothing recognises, and sRGB is the overwhelmingly common case).

        Both :meth:`optimize_map` and :meth:`assess` route through here — same
        rationale as :meth:`resolve_quality`: a dry run that predicted a codec
        the real run would refuse is worse than no prediction.

        Parameters:
            map_type_key: Canonical map-type key driving the derivation.
            output_type: Target container extension (with or without a dot).
            spec: Active profile's :class:`OutputSpec`, when one is resolved.

        Returns:
            tuple: ``(compression, colorspace, note)``. ``compression`` /
            ``colorspace`` are what :meth:`ImgUtils.save_image` should receive;
            ``note`` is a human-readable sentence when a request was declined,
            None otherwise.
        """
        ext = (output_type or "").lower().lstrip(".")
        requested = spec.compression if spec else None
        if ext != "ktx2":
            return requested, None, None

        registry = MapRegistry()
        map_def = registry.get(map_type_key) if map_type_key else None
        colorspace = (map_def.color_space if map_def else None) or "sRGB"
        lossy_safe = registry.is_lossy_safe(map_type_key)

        if requested is None:
            return ("ETC1S" if lossy_safe else "UASTC"), colorspace, None

        codec = str(requested).upper()
        if codec not in ("ETC1S", "UASTC"):
            raise ValueError(
                f"Unknown ktx2 compression {requested!r}: expected 'ETC1S' or "
                f"'UASTC' (DDS block formats do not apply to this container)."
            )
        if codec == "ETC1S" and not lossy_safe:
            return "UASTC", colorspace, (
                f"ETC1S refused for map type '{map_type_key or 'unknown'}': "
                f"palettised encoding bands on normals / packed / linear data — "
                f"encoded UASTC instead"
            )
        return codec, colorspace, None

    @classmethod
    def optimize_map(
        cls,
        texture_path: str,
        output_dir: str = None,
        output_type: str = None,
        max_size: int = None,
        force_pot: Optional[bool] = None,
        suffix_old: str = None,
        suffix_opt: str = None,
        old_files_folder: str = None,
        optimize_bit_depth: bool = True,
        check_existing: bool = False,
        map_type: str = None,
        allow_palette: bool = False,
        output_profile: str = None,
        enforce_budget: bool = False,
        lossy_quality: int = None,
        pot_mode: Optional[str] = None,
    ) -> str:
        """Optimizes a texture by resizing, setting bit depth, and adjusting image type.

        Parameters:
            texture_path (str): Path to the texture file.
            output_dir (str, optional): Directory for the optimized texture. Defaults to same directory.
            output_type (str, optional): Output image format (e.g., PNG, TGA). If None, keeps original.
            max_size (int, optional): Maximum size for the longest dimension. Only applies if the image is larger. Defaults to None.
            force_pot (bool): Snap the LONGEST edge to a power of two and
                derive the short edge from the source aspect, so the rule
                bounds how BIG the map is without reshaping it. It does NOT
                guarantee both edges are POT: 1024x768 is already legal on
                its long edge and passes through unchanged. A square/POT-both
                target needs its own flag (see :meth:`_snap_pot`).
            suffix_old (str, optional): Suffix to rename the original file before optimization.
            suffix_opt (str, optional): Suffix to append to the optimized file (None = overwrite).
            old_files_folder (str, optional): Name of the folder to store old files.
            optimize_bit_depth (bool): Adjusts bit depth to match the map type.
            check_existing (bool): If True, returns existing optimized file if it exists and is newer.
            map_type (str, optional): The type of map (e.g., "Normal", "MaskMap") to enforce specific modes.
            allow_palette (bool): If True, palette (P) inputs may be preserved
                when the target mode is RGB/RGBA. Default False (strict) — this
                prevents PNG palette-transparency from being read as alpha by
                downstream FBX/DCC pipelines.
            output_profile (str, optional): Workflow profile (a ``WF`` key) whose
                output template drives the container / bit depth / compression.
            enforce_budget (bool): Apply the profile's advisory
                ``DeliveryBudget`` (max_size / POT) instead of only reporting it.
                Default False — an over-budget map is correct, just expensive, so
                it is not silently resampled. An explicit ``max_size`` /
                ``force_pot`` always outranks the budget.
            lossy_quality (int, optional): Request lossy container compression
                (1-100) for this run. Applied only where it is safe — see
                :meth:`resolve_quality`; a request against a normal / packed /
                linear map, or a lossless container, is reported and the map is
                written lossless. None (default) = always lossless.
            pot_mode (str, optional): How ``force_pot`` snaps — ``"nearest"``
                or ``"down"`` (see :meth:`plan`). None (default) keeps the
                derived behavior: "down" when the POT rule came from an
                enforced profile budget, "nearest" otherwise. Callers that
                resolve a :class:`DeliveryBudget` themselves (a DCC export
                task enforcing a template's budget without adopting its
                container) must pass ``"down"`` — a budget must never grow
                an asset.

        Returns:
            str: Path to the optimized texture.
        """
        ImgUtils.assert_pathlike(texture_path, "texture_path")

        if output_dir is None:
            output_dir = os.path.dirname(texture_path)
        os.makedirs(output_dir, exist_ok=True)

        map_type_suffix = MapFactory.resolve_map_type(texture_path, key=False)
        if map_type_suffix is None:
            map_type_suffix = ""
        map_type_key = map_type or MapFactory.resolve_map_type(
            texture_path, key=True
        )

        # An active profile drives the output format and, on opt-in, the size
        # budget — resolved by the same helper assess uses, so the two agree.
        # An explicit pot_mode from the caller outranks the derived one, same
        # precedence rule as the other advisory-tier arguments.
        spec, budget, output_type, max_size, force_pot, derived_pot_mode = (
            cls._resolve_profile(
                output_profile,
                map_type_key,
                output_type,
                max_size,
                force_pot,
                enforce_budget,
            )
        )
        pot_mode = pot_mode if pot_mode is not None else derived_pot_mode
        target_bit_depth = spec.bit_depth if spec else None

        # Calculate output path early to check for existence
        temp_path = MapFactory.resolve_texture_filename(
            texture_path,
            map_type_suffix,
            suffix=suffix_opt,
            ext=output_type,
        )
        final_output_path = os.path.join(output_dir, os.path.basename(temp_path))
        # The container that will really receive the pixels, resolved before
        # planning: with output_type=None the run keeps the source's extension,
        # and the planner's high-precision tolerance is a property of the
        # container written, not of the requested one.
        out_ext = os.path.splitext(final_output_path)[1]

        if check_existing and os.path.exists(final_output_path):
            if os.path.getmtime(final_output_path) > os.path.getmtime(texture_path):
                print(
                    f"Skipping optimization (existing/newer): "
                    f"{os.path.basename(final_output_path)}"
                )
                return final_output_path

        # Read the source's on-disk size before any archive/overwrite step can
        # move or replace it — it's the "from" half of the size report below.
        size_before = (
            os.path.getsize(texture_path) if os.path.isfile(texture_path) else None
        )

        image = ImgUtils.ensure_image(texture_path)
        dims_before = image.size

        plan = cls.plan(
            image,
            max_size=max_size,
            force_pot=force_pot,
            optimize_bit_depth=optimize_bit_depth,
            map_type_key=map_type_key,
            allow_palette=allow_palette,
            pot_mode=pot_mode,
            output_profile=output_profile,
            output_type=out_ext,
        )

        if any(op.kind == "resize" for op in plan):
            # Match the original log line so existing tooling that scrapes
            # logs continues to work.
            print(
                f"Resizing {texture_path} from {image.size[0]}x{image.size[1]} "
                f"to {max_size}x{max_size} .."
            )

        for message in cls._plan_warnings(plan):
            print(f"# Warning: {os.path.basename(final_output_path)}: {message}")

        image = cls.apply(image, plan)

        # File rename / archive handling — orchestrator concern, not planner.
        old_texture_path = (
            MapFactory.resolve_texture_filename(
                texture_path, map_type_suffix, suffix=suffix_old
            )
            if suffix_old
            else None
        )

        if old_files_folder:
            old_folder = os.path.join(output_dir, old_files_folder)
            FileUtils.move_file(
                texture_path,
                old_folder,
                new_name=(
                    os.path.basename(old_texture_path) if old_texture_path else None
                ),
            )

        # Widen to what the target container can hold BEFORE reporting, so
        # format_result describes the file on disk. WebP has no grayscale mode,
        # so an "L" roughness map is stored as RGB — reporting the in-memory
        # mode would claim 8-bit for a 24-bit file (and disagree with assess,
        # which applies the same step to its projection).
        container_mode = ImgUtils.effective_mode(image.mode, out_ext)
        if container_mode != image.mode:
            lost = cls.channel_loss_warning(image, out_ext)
            print(
                f"# {os.path.basename(final_output_path)}: "
                f"'{out_ext.lstrip('.')}' cannot store {image.mode}; "
                f"written as {container_mode}."
                + (f" *** DATA LOSS: {lost}. ***" if lost else "")
            )
            image = ImgUtils.enforce_mode(image, container_mode)

        # Resolved against the path actually being written, not the requested
        # output_type: with output_type=None the run keeps the source's
        # extension, and a lossy request has to be judged against the container
        # that will really receive it.
        quality, quality_skipped = cls.resolve_quality(
            lossy_quality,
            map_type_key,
            out_ext,
            spec,
        )
        if quality_skipped:
            print(
                f"# {os.path.basename(final_output_path)}: {quality_skipped}"
            )

        # GPU compression (DDS block format / ktx2 Basis codec) resolves against
        # the container actually being written, for the same reason quality
        # does. The colorspace label travels with it — ktx2 is the container
        # whose DFD records the transfer function.
        target_compression, target_colorspace, compression_note = (
            cls.resolve_compression(map_type_key, out_ext, spec)
        )
        if compression_note:
            print(f"# {os.path.basename(final_output_path)}: {compression_note}")

        # Route through the capability-aware writer (single save SSoT) so the
        # correct backend handles each format (PIL for most, cv2 for EXR/HDR).
        # The extension on final_output_path drives format dispatch; the profile
        # template (if any) supplies bit depth / compression.
        ImgUtils.save_image(
            image,
            final_output_path,
            optimize=True,
            bit_depth=target_bit_depth,
            compression=target_compression,
            quality=quality,
            colorspace=target_colorspace,
        )

        print(
            f"Saved optimized texture: {final_output_path} "
            f"({cls.format_result(final_output_path, size_before, dims_before, image)})"
        )

        # Reported against what was actually written, so an explicit max_size that
        # overshot the profile's budget still surfaces. An ENFORCED run is inside
        # the budget's size ceiling by construction, but can still report: the POT
        # snap only makes the LONG edge a power of two, so a non-square map keeps a
        # non-POT short edge that budget.check names. That residual is deliberate -
        # it is the aspect ratio not being paid for - and saying so is the point.
        if budget:
            for message in budget.check(*image.size):
                print(
                    f"# Warning: {os.path.basename(final_output_path)} "
                    f"[{output_profile}]: {message}"
                )
        return final_output_path

    @staticmethod
    def channel_loss_warning(image: "Image.Image", ext: str) -> Optional[str]:
        """Warn when *ext* would discard a channel of *image* that holds data.

        A dropped channel is only a loss if something was in it. Alpha that is
        uniformly opaque — a lightmap, an albedo with no transparency — costs
        nothing, and shouting about it would train the reader past the line
        that matters. A PACKED map is the case that matters: its alpha is a
        material input (MSAO keeps Smoothness there), so discarding it changes
        how the surface renders.

        The wording lives here, once, for the same reason :class:`Op` carries
        its own ``description``: :meth:`assess` must warn about this in a dry
        run and :meth:`optimize_map` must say the same thing while writing.
        Note it is invisible to the plan — a container's limits are not an op,
        so it can never appear in ``reasons``.

        Returns:
            str | None: The warning, or None when nothing of substance is lost.
        """
        lost = ImgUtils.dropped_channels(image.mode, ext)
        if not lost:
            return None
        try:
            extrema = dict(zip(image.getbands(), image.getextrema()))
        except OSError:  # truncated source — the writer will surface it
            return None
        carrying = [
            band
            for band in lost
            if isinstance(extrema.get(band), tuple)
            and extrema[band][0] != extrema[band][1]
        ]
        if not carrying:
            return None
        return (
            f"channel {'/'.join(carrying)} carries data and is discarded by "
            f"'{ext.lstrip('.').lower()}' - use png/tga/webp to keep it"
        )

    @staticmethod
    def format_result(
        output_path: str,
        size_before: Optional[int],
        dims_before: Optional[Tuple[int, int]],
        image: "Image.Image",
    ) -> str:
        """Render the one-line result summary for an optimized map.

        File size is the point of optimizing, so it leads the unchanged
        dimension/bit-depth pair with a ``before -> after`` transition. A
        dimension pair that didn't change renders once rather than as a
        no-op arrow.

        Parameters:
            output_path: Path just written (stat'd for the resulting size).
            size_before: Source byte count, or None when unknown.
            dims_before: Source ``(width, height)``, or None when unknown.
            image: The saved image (supplies final dims + bit depth).

        Returns:
            str: e.g. ``"4096x4096 -> 2048x2048, 24bit (8x3), 12.00 MB ->
                3.00 MB (-75%)"``.
        """
        width, height = image.size
        dims = f"{width}x{height}"
        if dims_before and tuple(dims_before) != (width, height):
            dims = f"{dims_before[0]}x{dims_before[1]} -> {dims}"

        size_after = (
            os.path.getsize(output_path) if os.path.isfile(output_path) else None
        )
        sizes = FileUtils.format_bytes_delta(size_before, size_after)

        return f"{dims}, {ImgUtils.format_bit_depth(image)}, {sizes}"

    @classmethod
    def batch_optimize_maps(cls, directory: str, **kwargs):
        """Batch optimizes all maps in a directory.

        Parameters:
            directory (str): Directory containing the maps to optimize.
            **kwargs: Forwarded to :meth:`optimize_map`.
        """
        ImgUtils.assert_pathlike(directory, "directory")

        textures = ImgUtils.get_images(directory)
        print(f"Optimizing maps in: {directory}")
        for texture_path in textures.keys():
            cls.optimize_map(texture_path, **kwargs)
        print(f"{len(textures)} maps optimized.")

    @classmethod
    def assess(
        cls,
        texture_path: str,
        max_size: int = None,
        force_pot: Optional[bool] = None,
        optimize_bit_depth: bool = True,
        map_type: str = None,
        allow_palette: bool = False,
        image: "Image.Image" = None,
        output_type: str = None,
        output_profile: str = None,
        predict_size: bool = False,
        enforce_budget: bool = False,
        lossy_quality: int = None,
        pot_mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Predict whether :meth:`optimize_map` would change ``texture_path``.

        Read-only wrapper around :meth:`plan`. Returns a dict the UI / report
        callers can render without re-deriving decision strings. This is the
        dry-run twin of :meth:`optimize_map` — nothing on disk is touched.

        Parameters:
            texture_path: Path to the texture file.
            max_size, force_pot, optimize_bit_depth, map_type, allow_palette:
                Same semantics as :meth:`optimize_map`.
            image: Optional pre-loaded ``PIL.Image.Image`` to skip the
                redundant header read for callers that already have one open.
            output_type: Output format the run would write (e.g. "png"). Only
                affects the predicted extension / size; None keeps the source's.
            output_profile: Output-template profile the run would use. Resolved
                through the same :meth:`_resolve_profile` helper the real run
                uses, so a profile that dictates the extension / bit depth /
                compression is reflected in the prediction rather than silently
                ignored.
            predict_size: Also report the resulting *byte* count. Costs a real
                encode (the plan is applied to a copy and written to a scratch
                file that is immediately discarded), so it is opt-in — bulk
                report callers assessing every texture in a scene shouldn't
                pay for it. Compression ratios can't be derived from
                dimensions and mode, so this is the only honest way to
                answer "how much smaller?" before committing.
            enforce_budget: Same semantics as :meth:`optimize_map` — predict a
                run that applies the profile's advisory ``DeliveryBudget``
                rather than one that only reports it.
            lossy_quality: Same semantics as :meth:`optimize_map`. A request the
                safety gate declines surfaces in ``warnings`` and the predicted
                size (when *predict_size*) is the lossless one the run would
                actually write.
            pot_mode: Same semantics as :meth:`optimize_map` — an explicit
                "nearest"/"down" outranks the profile-derived behavior, so a
                dry run predicts the same snap the caller's real run will make.

        Returns:
            dict with:
                recommended (bool): True if the plan is non-empty.
                reasons (list[str]): Per-op descriptions from the plan.
                warnings (list[str]): What the run would NOT do, or would cost:
                    advisory ``DeliveryBudget`` violations, a declined lossy
                    request, and a channel the output container discards.
                    Always present, empty when there is nothing to flag —
                    distinct from ``reasons``, which describe changes that
                    *would* be made. The channel-loss entry can only appear
                    here: a container's limits are not a plan op, so nothing
                    in ``reasons`` ever mentions them.
                current (dict): {path, name, width, height, mode, format,
                    size_bytes, bit_depth, map_type}.
                predicted (dict): {width, height, mode, bit_depth, ext, path,
                    size_bytes}. ``path`` is where a default-location run would
                    write. ``size_bytes`` is None unless *predict_size* was
                    requested (or the encode failed, in which case
                    ``size_error`` carries the reason).
                target_mode (str | None): Map-type-driven target mode, when
                    one exists.
                error (str): Only present when the file could not be read.

            An already-encoded ``.ktx2`` is reported from its header instead —
            PIL cannot open a Basis payload, and this method's main consumer is
            the exporters' optimize-textures gate assessing the output a task
            just wrote. Such a report is always ``recommended: False`` (nothing
            here can transcode one back to pixels) and carries ``mode`` /
            ``bit_depth`` as None — see :meth:`_assess_delivered_ktx2`.
        """
        ImgUtils.assert_pathlike(texture_path, "texture_path")

        if not os.path.exists(texture_path) and image is None:
            return {
                "recommended": False,
                "reasons": [],
                "warnings": [],
                "error": f"File not found: {texture_path}",
                "current": {
                    "path": texture_path,
                    "name": os.path.basename(texture_path),
                },
                "predicted": {},
                "target_mode": None,
            }

        size_bytes = (
            os.path.getsize(texture_path) if os.path.exists(texture_path) else None
        )

        # A Basis-encoded .ktx2 has no pure-Python decoder, so PIL cannot open
        # the one file this method is most often pointed at: the scene
        # exporters' optimize-textures gate assesses the staged output the task
        # just wrote. Reporting "Failed to read image" there made the gate flag
        # every ktx2 map it delivered, and handed a KeyError to anything
        # reading predicted["width"]. The header answers the geometry question
        # without a transcoder; see :meth:`_assess_delivered_ktx2`.
        if (
            image is None
            and FileUtils.format_path(texture_path, "ext").lower().lstrip(".") == "ktx2"
        ):
            return cls._assess_delivered_ktx2(
                texture_path, size_bytes, map_type, output_type, predict_size
            )

        try:
            if image is None:
                with ImgUtils.allow_large_images():
                    image = ImgUtils.ensure_image(texture_path)
            width, height = image.size
            mode = image.mode
            img_format = image.format
        except Exception as e:
            return {
                "recommended": False,
                "reasons": [],
                "warnings": [],
                "error": f"Failed to read image: {e}",
                "current": {
                    "path": texture_path,
                    "name": os.path.basename(texture_path),
                    "size_bytes": size_bytes,
                },
                "predicted": {},
                "target_mode": None,
            }

        map_type_key = map_type or MapFactory.resolve_map_type(
            texture_path, key=True
        )

        # Resolved before planning (the budget can add ops), through the same
        # helper optimize_map uses — so the dry run plans what the real run would.
        spec, budget, output_type, max_size, force_pot, derived_pot_mode = (
            cls._resolve_profile(
                output_profile,
                map_type_key,
                output_type,
                max_size,
                force_pot,
                enforce_budget,
            )
        )
        pot_mode = pot_mode if pot_mode is not None else derived_pot_mode

        # Resolved BEFORE planning, for the same reason optimize_map resolves
        # it before planning: the high-precision tolerance is a property of the
        # container the run writes, and with output_type=None that container is
        # the source's own extension.
        out_ext = (
            (output_type or FileUtils.format_path(texture_path, "ext"))
            .lower()
            .lstrip(".")
        )

        ops = cls.plan(
            image,
            max_size=max_size,
            force_pot=force_pot,
            optimize_bit_depth=optimize_bit_depth,
            map_type_key=map_type_key,
            allow_palette=allow_palette,
            pot_mode=pot_mode,
            output_profile=output_profile,
            output_type=out_ext,
        )

        # Surface the target mode the planner picked (first mode_coerce op),
        # if any. UI callers use this to render the current/target pair side
        # by side.
        target_mode: Optional[str] = None
        for op in ops:
            if op.kind == "mode_coerce":
                target_mode = op.params.get("target_mode")
                break

        # Replay the plan for the projected size/mode (cheap), then optionally
        # encode for the projected byte count (not cheap — hence opt-in).
        new_width, new_height, planned_mode = cls.project(ops, width, height, mode)
        # The plan decides a mode; the container decides whether it survives.
        # Applied after project() rather than inside it so the replay stays a
        # pure function of the plan — this is a property of the output format,
        # which the plan never sees. Both are kept: the difference between them
        # is exactly what the container costs, which is what gets warned about.
        new_mode = ImgUtils.effective_mode(planned_mode, out_ext)
        predicted: Dict[str, Any] = {
            "width": new_width,
            "height": new_height,
            "mode": new_mode,
            "bit_depth": ImgUtils.format_bit_depth(new_mode),
            "ext": out_ext,
            # Same resolve call optimize_map makes, so a dry run names the file
            # the real run would write (the map-type suffix gets normalized
            # here — predicting it by hand would drift).
            "path": MapFactory.resolve_texture_filename(
                texture_path,
                MapFactory.resolve_map_type(texture_path, key=False) or "",
                ext=output_type,
            ),
            "size_bytes": None,
        }
        # Same gate the real run applies, against the same resolved extension, so
        # a declined request is visible in the dry run instead of surfacing only
        # once the batch has already been written.
        quality, quality_skipped = cls.resolve_quality(
            lossy_quality, map_type_key, out_ext, spec
        )
        predicted["quality"] = quality

        # Same resolution the real run applies (ktx2 Basis codec derivation /
        # ETC1S refusal), so the prediction names the codec that will actually
        # be encoded and a declined request surfaces before the batch runs.
        compression, ktx2_colorspace, compression_note = cls.resolve_compression(
            map_type_key, out_ext, spec
        )
        if compression:
            predicted["compression"] = compression

        # Gated on the PLANNED mode, not the source's: a plan that already
        # coerces RGBA->RGB dropped the alpha itself and said so as an op, so
        # the container is not what loses it. Only the source image can answer
        # whether the channel held anything, and reading its extrema is far
        # cheaper than converting a copy just to ask.
        channel_loss = (
            cls.channel_loss_warning(image, out_ext)
            if ImgUtils.dropped_channels(planned_mode, out_ext)
            else None
        )

        if predict_size:
            predicted_bytes, size_error = cls._encoded_size(
                image,
                ops,
                out_ext,
                bit_depth=spec.bit_depth if spec else None,
                compression=compression,
                quality=quality,
                colorspace=ktx2_colorspace,
            )
            predicted["size_bytes"] = predicted_bytes
            if size_error:
                predicted["size_error"] = size_error

        return {
            "recommended": bool(ops),
            "reasons": [op.description for op in ops],
            # Against the *predicted* dimensions — the question a caller is asking
            # is whether the file this run would write lands within budget, not
            # whether the source did.
            "warnings": (budget.check(new_width, new_height) if budget else [])
            + cls._plan_warnings(ops)
            + ([quality_skipped] if quality_skipped else [])
            + ([compression_note] if compression_note else [])
            + ([channel_loss] if channel_loss else []),
            "current": {
                "path": texture_path,
                "name": os.path.basename(texture_path),
                "width": width,
                "height": height,
                "mode": mode,
                "format": img_format,
                "size_bytes": size_bytes,
                "bit_depth": ImgUtils.format_bit_depth(mode),
                "map_type": map_type_key,
            },
            "predicted": predicted,
            "target_mode": target_mode,
        }

    @classmethod
    def _assess_delivered_ktx2(
        cls,
        texture_path: str,
        size_bytes: Optional[int],
        map_type: Optional[str],
        output_type: Optional[str],
        predict_size: bool,
    ) -> Dict[str, Any]:
        """:meth:`assess` for a file that is already Basis-encoded.

        Nothing in this stack can transcode a ``.ktx2`` back to pixels, so a
        delivered GPU texture has no plan to make — the honest report is "done,
        and here is its geometry". :meth:`Ktx2Encoder.read_header` supplies the
        geometry from the fixed-layout header, which is what the exporters'
        optimize-textures gate actually reads back.

        ``mode`` / ``bit_depth`` are reported as None rather than guessed: a
        Basis payload's channel layout is a property of the transcode target
        the *loader* picks (ASTC / BC7 / ETC2), not of the file.
        """
        from pythontk.img_utils.ktx2_encoder import Ktx2Encoder

        current: Dict[str, Any] = {
            "path": texture_path,
            "name": os.path.basename(texture_path),
            "size_bytes": size_bytes,
        }
        try:
            header = Ktx2Encoder.read_header(texture_path)
        except (OSError, ValueError) as e:
            # Same shape (and same "Failed to read image" prefix) the PIL path
            # returns: a corrupt delivery must still read as unreadable, not as
            # a confident zero-size report.
            return {
                "recommended": False,
                "reasons": [],
                "warnings": [],
                "error": f"Failed to read image: {e}",
                "current": current,
                "predicted": {},
                "target_mode": None,
            }

        width, height = header["width"], header["height"]
        requested = (output_type or "ktx2").lower().lstrip(".")
        current.update(
            {
                "width": width,
                "height": height,
                "mode": None,
                "format": "KTX2",
                "bit_depth": None,
                "map_type": map_type
                or MapFactory.resolve_map_type(texture_path, key=True),
            }
        )
        return {
            "recommended": False,
            "reasons": [],
            "warnings": (
                []
                if requested == "ktx2"
                else [
                    f"'{os.path.basename(texture_path)}' is already Basis-encoded: "
                    f"a .ktx2 cannot be re-planned into '{requested}' here (no "
                    f"transcoder) — assessed as delivered"
                ]
            ),
            "current": current,
            "predicted": {
                "width": width,
                "height": height,
                "mode": None,
                "bit_depth": None,
                "ext": "ktx2",
                "path": texture_path,
                # No re-encode happens, so the delivered file IS the predicted
                # size — but only when the caller asked, keeping the opt-in
                # contract the PIL path documents.
                "size_bytes": size_bytes if predict_size else None,
                "quality": None,
            },
            "target_mode": None,
        }

    @classmethod
    def _encoded_size(
        cls,
        image: "Image.Image",
        plan: List[Op],
        ext: str,
        bit_depth: Optional[int] = None,
        compression: Optional[str] = None,
        quality: Optional[int] = None,
        colorspace: Optional[str] = None,
    ) -> Tuple[Optional[int], Optional[str]]:
        """Byte count ``plan`` would produce, measured by a throwaway encode.

        Routes through :meth:`ImgUtils.save_image` — the same writer
        :meth:`optimize_map` uses, with the same profile-driven bit depth and
        compression — so format dispatch (Pillow / cv2 / DDS) and mode fixups
        are identical to the real run instead of an approximation that could
        drift from it.

        The caller's image is passed through uncopied: every op in
        :meth:`apply` rebinds rather than mutating in place, which is the same
        invariant :meth:`optimize_map` already relies on, and copying an 8K
        texture just to throw it away is a real cost on the dry-run path.

        Returns:
            tuple: ``(size_bytes, error)`` — exactly one is non-None.
        """
        import shutil
        from pythontk.file_utils.temp_artifacts import TempArtifacts

        scratch = TempArtifacts("map_optimizer_dryrun").dir_path()
        try:
            probe = os.path.join(scratch, f"probe.{ext}")
            ImgUtils.save_image(
                cls.apply(image, plan),
                probe,
                optimize=True,
                bit_depth=bit_depth,
                compression=compression,
                quality=quality,
                colorspace=colorspace,
            )
            return os.path.getsize(probe), None
        except Exception as e:
            return None, str(e)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
