# !/usr/bin/python
# coding=utf-8
"""Extending ``MapFactory`` without touching core code.

Two extension points:

1. **Workflow handlers** (``WorkflowHandler`` subclasses) — produce output
   maps for a target pipeline. Registered via ``MapFactory.register_handler``;
   gated by a config flag so they only run when explicitly requested
   (the factory is non-greedy by design).
2. **Conversions** (``MapConversion``) — teach ``TextureProcessor.resolve_map``
   how to derive one map type from another. Registered via
   ``MapFactory.register_conversion``.

Handlers should resolve components through ``context.resolve_map`` (which
walks direct matches, then registered conversions, then safe input
fallbacks), save through ``context.save_map`` (which owns naming, format,
optimization, and dry-run handling), and report what they consumed via
``get_consumed_types`` so consumed sources aren't also passed through.
"""
import os
from typing import List, Optional

from pythontk import ImgUtils
from pythontk.core_utils.engines.textures.map_factory import (
    MapFactory,
    MapConversion,
    TextureProcessor,
    WorkflowHandler,
)


# ---------------------------------------------------------------------------
# 1. A custom workflow handler: pack Metallic (R), Roughness (G), AO (B),
#    Height (A) for a hypothetical engine.
# ---------------------------------------------------------------------------
class SubstancePackedHandler(WorkflowHandler):
    """Packs M/R/AO/H when config requests ``substance_packed=True``."""

    def can_handle(self, context: TextureProcessor) -> bool:
        return bool(context.config.get("substance_packed", False))

    def process(self, context: TextureProcessor) -> Optional[str]:
        # resolve_map derives missing components via the conversion registry
        # (e.g. Smoothness -> inverted Roughness, Specular -> Metallic).
        metallic = context.resolve_map("Metallic", allow_conversion=True)
        roughness = context.resolve_map("Roughness", allow_conversion=True)
        ao = context.resolve_map("Ambient_Occlusion", allow_conversion=False)
        height = context.resolve_map("Height", allow_conversion=False)

        if not (metallic and roughness):
            if context.logger:
                context.logger.warning(
                    "Missing Metallic/Roughness for Substance packed map"
                )
            return None

        # out_mode is explicit: without it, a missing Height would drop the
        # alpha channel entirely instead of using the mid-height fill.
        packed = ImgUtils.pack_channels(
            channel_files={"R": metallic, "G": roughness, "B": ao, "A": height},
            output_path=None,
            out_mode="RGBA",
            fill_values={"B": 255, "A": 128},  # neutral AO / mid height
        )
        if context.logger:
            context.logger.info("Created Substance packed map")
        sources = [m for m in (metallic, roughness, ao, height) if m]
        return context.save_map(packed, "SubstancePacked", source_images=sources)

    def get_consumed_types(self) -> List[str]:
        return [
            "Metallic",
            "Roughness",
            "Smoothness",
            "Glossiness",
            "Ambient_Occlusion",
            "Height",
            "Specular",
        ]


MapFactory.register_handler(SubstancePackedHandler)


# ---------------------------------------------------------------------------
# 2. A custom conversion: derive AO from a Curvature map. Once registered,
#    any handler asking resolve_map("Ambient_Occlusion") benefits from it.
# ---------------------------------------------------------------------------
def _curvature_to_ao(inventory, context):
    curvature_img = ImgUtils.ensure_image(inventory["Curvature"])
    ao_img = ImgUtils.invert_grayscale_image(curvature_img.convert("L"))
    if context.logger:
        context.logger.info("Derived AO from Curvature")
    return ao_img


MapFactory.register_conversion(
    MapConversion(
        target_type="Ambient_Occlusion",
        source_types=["Curvature"],
        converter=_curvature_to_ao,
        priority=8,
    )
)


# ---------------------------------------------------------------------------
# Usage — same entry point as the built-in workflows.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    source = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    results = MapFactory.prepare_maps(
        source,  # directory, file, or list of files
        output_dir=os.path.join(source, "output"),
        substance_packed=True,  # enables the custom handler above
        normal_type="OpenGL",
        output_extension="png",
        rename=True,
    )
    print(results)
