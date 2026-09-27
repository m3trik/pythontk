# !/usr/bin/python
# coding=utf-8
"""Grayscale maps packed into one texture's channels (ORM, MSAO, MRAO, albedo +
transparency, metallic + smoothness) and packed textures split back out (the
bodies behind the :class:`MapFactory` facade).
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore

from pythontk.img_utils._img_utils import ImgUtils
from pythontk.core_utils.engines.textures.map_registry import MapRegistry
from .processor import TextureProcessor, DEFAULT_EXTENSION, ALPHA_EXTENSION


class _ChannelPackingInternal:
    """Bodies of :class:`MapFactory`'s channel packing and unpacking.

    The public signatures and docstrings stay on :class:`MapFactory`; each
    delegates here. A ``MapFactory`` base, so ``cls`` is the factory (or a
    subclass) and its registries and helpers resolve through the MRO.
    """

    @classmethod
    def extract_channels(
        cls,
        packed_type: str,
        packed_path: Optional[str],
        targets: List[str],
        config: Dict[str, Any] = None,
    ) -> Optional[Dict[str, str]]:
        """Body of :meth:`MapFactory.extract_channels`."""
        if Image is None or not (packed_path and os.path.isfile(packed_path)):
            return None

        context = TextureProcessor(
            inventory={packed_type: packed_path},
            config=dict(config or {}),
            output_dir=os.path.dirname(packed_path),
            base_name=cls.get_base_texture_name(packed_path),
            tile_token=cls.get_tile_token(packed_path),
            ext=(config or {}).get("output_extension")
            or os.path.splitext(packed_path)[1].lstrip("."),
            conversion_registry=cls._conversion_registry,
            logger=cls.logger,
        )

        extracted: Dict[str, str] = {}
        for target in targets:
            # A real loose map already on disk under the canonical name wins —
            # overwriting it with extracted channel data would destroy user
            # files the caller simply didn't list.
            candidate = context.output_path_for(target)
            if os.path.isfile(candidate):
                cls.logger.info(
                    f"Reusing existing {target} map instead of extracting "
                    f"from {packed_type}: {candidate}",
                    extra={"preset": "highlight"},
                )
                extracted[target] = candidate
                continue

            try:
                image = context.resolve_map(target, allow_conversion=True)
            except Exception as e:
                cls.logger.warning(f"Extracting {target} from {packed_type}: {e}")
                return None
            if not image:
                return None
            extracted[target] = context.save_map(
                image, target, source_images=[packed_path]
            )
        return extracted

    @classmethod
    def pack_transparency_into_albedo(
        cls,
        albedo_map_path: str,
        alpha_map_path: str,
        output_dir: Optional[str] = None,
        suffix: Optional[str] = "_AlbedoTransparency",
        invert_alpha: bool = False,
        output_path: Optional[str] = None,
        save: bool = True,
    ) -> Union[str, "Image.Image"]:
        """Body of :meth:`MapFactory.pack_transparency_into_albedo`."""
        if isinstance(albedo_map_path, str):
            ImgUtils.assert_pathlike(albedo_map_path, "albedo_map_path")
        if isinstance(alpha_map_path, str):
            ImgUtils.assert_pathlike(alpha_map_path, "alpha_map_path")

        if save and output_path is None:
            if not isinstance(albedo_map_path, str):
                raise ValueError(
                    "Cannot determine output path from Image object. Please provide output_path or output_dir."
                )

            base_name = ImgUtils.get_base_texture_name(albedo_map_path)

            if output_dir is None:
                output_dir = os.path.dirname(albedo_map_path)
            elif not os.path.isdir(output_dir):
                raise ValueError(
                    f"The specified output directory '{output_dir}' is not valid."
                )

            output_path = os.path.join(
                output_dir, f"{base_name}{suffix}.{ALPHA_EXTENSION}"
            )
        elif not save:
            output_path = None

        return ImgUtils.pack_channel_into_alpha(
            albedo_map_path,
            alpha_map_path,
            output_path,
            invert_alpha=invert_alpha,
        )

    @classmethod
    def pack_smoothness_into_metallic(
        cls,
        metallic_map_path: str,
        alpha_map_path: str,
        output_dir: str = None,
        suffix: str = "_MetallicSmoothness",
        invert_alpha: bool = False,
        output_path: str = None,
        save: bool = True,
    ) -> Union[str, "Image.Image"]:
        """Body of :meth:`MapFactory.pack_smoothness_into_metallic`."""
        if isinstance(metallic_map_path, str):
            ImgUtils.assert_pathlike(metallic_map_path, "metallic_map_path")
        if isinstance(alpha_map_path, str):
            ImgUtils.assert_pathlike(alpha_map_path, "alpha_map_path")

        if save and output_path is None:
            if not isinstance(metallic_map_path, str):
                raise ValueError(
                    "Cannot determine output path from Image object. Please provide output_path or output_dir."
                )

            base_name = ImgUtils.get_base_texture_name(metallic_map_path)
            if output_dir is None:
                output_dir = os.path.dirname(metallic_map_path)
            elif not os.path.isdir(output_dir):
                raise ValueError(
                    f"The specified output directory '{output_dir}' is not valid."
                )

            output_path = os.path.join(
                output_dir, f"{base_name}{suffix}.{ALPHA_EXTENSION}"
            )
        elif not save:
            output_path = None

        return ImgUtils.pack_channel_into_alpha(
            metallic_map_path, alpha_map_path, output_path, invert_alpha=invert_alpha
        )

    @classmethod
    def pack_orm_texture(
        cls,
        ao_map_path: Optional[str],
        roughness_map_path: Optional[str],
        metallic_map_path: Optional[str],
        output_dir: str = None,
        suffix: str = "_ORM",
        invert_roughness: bool = False,
        output_path: str = None,
        save: bool = True,
    ) -> Union[str, "Image.Image"]:
        """Body of :meth:`MapFactory.pack_orm_texture`."""
        if ao_map_path and isinstance(ao_map_path, str):
            ImgUtils.assert_pathlike(ao_map_path, "ao_map_path")
        if roughness_map_path and isinstance(roughness_map_path, str):
            ImgUtils.assert_pathlike(roughness_map_path, "roughness_map_path")
        if metallic_map_path and isinstance(metallic_map_path, str):
            ImgUtils.assert_pathlike(metallic_map_path, "metallic_map_path")

        # Expanded after the assertions (which reject an Image) but the ORIGINAL
        # arguments still name the output below: expansion replaces a packed
        # path with in-memory channels, and deriving the name from those would
        # turn every packed input into the "cannot derive from Image" error.
        originals = (ao_map_path, roughness_map_path, metallic_map_path)
        ao_map_path, roughness_map_path, metallic_map_path = cls._resolve_orm_sources(
            *originals
        )

        if save and output_path is None:
            source_map = next((src for src in originals if src), None)
            if not source_map:
                raise ValueError("No source maps provided to derive output name")

            base_name = cls.get_base_texture_name(source_map)

            if output_dir is None:
                if isinstance(source_map, str):
                    output_dir = os.path.dirname(source_map)
                else:
                    raise ValueError(
                        "Cannot derive output directory from Image object; provide output_dir explicitly"
                    )
            elif not os.path.isdir(output_dir):
                raise ValueError(
                    f"The specified output directory '{output_dir}' is not valid."
                )

            output_path = os.path.join(
                output_dir, f"{base_name}{suffix}.{DEFAULT_EXTENSION}"
            )
        elif not save:
            output_path = None

        return ImgUtils.pack_channels(
            channel_files={
                "R": ao_map_path,
                "G": roughness_map_path,
                "B": metallic_map_path,
            },
            output_path=output_path,
            out_mode="RGB",
            invert_channels=["G"] if invert_roughness else None,
            fill_values={"R": 255, "G": 0, "B": 0},
            save=save,
        )

    @classmethod
    def pack_msao_texture(
        cls,
        metallic_map_path: str,
        ao_map_path: Optional[str],
        alpha_map_path: Optional[str],
        detail_map_path: Optional[str] = None,
        output_dir: str = None,
        suffix: str = "_MSAO",
        invert_alpha: bool = False,
        output_path: str = None,
        save: bool = True,
        layout: str = "rgba",
    ) -> Union[str, "Image.Image"]:
        """Body of :meth:`MapFactory.pack_msao_texture`."""
        layout = (layout or "rgba").lower()
        if layout not in ("rgba", "rgb"):
            raise ValueError(f"Unsupported MSAO layout: {layout!r}")

        if isinstance(metallic_map_path, str):
            ImgUtils.assert_pathlike(metallic_map_path, "metallic_map_path")
        if ao_map_path and isinstance(ao_map_path, str):
            ImgUtils.assert_pathlike(ao_map_path, "ao_map_path")
        if alpha_map_path and isinstance(alpha_map_path, str):
            ImgUtils.assert_pathlike(alpha_map_path, "alpha_map_path")
        if detail_map_path and isinstance(detail_map_path, str):
            ImgUtils.assert_pathlike(detail_map_path, "detail_map_path")

        if save and output_path is None:
            source_map = (
                metallic_map_path or ao_map_path or alpha_map_path or detail_map_path
            )
            if not source_map:
                raise ValueError("No source maps provided to derive output name")

            base_name = cls.get_base_texture_name(source_map)

            if output_dir is None:
                if isinstance(source_map, str):
                    output_dir = os.path.dirname(source_map)
                else:
                    raise ValueError(
                        "Cannot derive output directory from Image object; provide output_dir explicitly"
                    )
            elif not os.path.isdir(output_dir):
                raise ValueError(
                    f"The specified output directory '{output_dir}' is not valid."
                )

            output_path = os.path.join(
                output_dir, f"{base_name}{suffix}.{DEFAULT_EXTENSION}"
            )
        elif not save:
            output_path = None

        if layout == "rgb":
            # 3-channel parallel layout: R=Metallic, G=Smoothness, B=AO
            return ImgUtils.pack_channels(
                channel_files={
                    "R": metallic_map_path,
                    "G": alpha_map_path,
                    "B": ao_map_path,
                },
                output_path=output_path,
                out_mode="RGB",
                invert_channels=["G"] if invert_alpha else None,
                fill_values={"R": 0, "G": 255, "B": 255},
                save=save,
            )

        # Default HDRP Mask Map layout: R=Metallic, G=AO, B=Detail, A=Smoothness
        return ImgUtils.pack_channels(
            channel_files={
                "R": metallic_map_path,
                "G": ao_map_path,
                "B": detail_map_path,
                "A": alpha_map_path,
            },
            output_path=output_path,
            out_mode="RGBA",
            invert_channels=["A"] if invert_alpha else None,
            fill_values={"G": 255, "B": 0, "A": 255},
            save=save,
        )

    @classmethod
    def pack_mrao_texture(
        cls,
        metallic_map_path: Optional[str],
        roughness_map_path: Optional[str],
        ao_map_path: Optional[str],
        detail_map_path: Optional[str] = None,
        output_dir: str = None,
        suffix: str = "_MRAO",
        invert_roughness: bool = False,
        output_path: str = None,
        save: bool = True,
        layout: str = "rgb",
    ) -> Union[str, "Image.Image"]:
        """Body of :meth:`MapFactory.pack_mrao_texture`."""
        layout = (layout or "rgb").lower()
        if layout not in ("rgb", "rgba"):
            raise ValueError(f"Unsupported MRAO layout: {layout!r}")

        if metallic_map_path and isinstance(metallic_map_path, str):
            ImgUtils.assert_pathlike(metallic_map_path, "metallic_map_path")
        if roughness_map_path and isinstance(roughness_map_path, str):
            ImgUtils.assert_pathlike(roughness_map_path, "roughness_map_path")
        if ao_map_path and isinstance(ao_map_path, str):
            ImgUtils.assert_pathlike(ao_map_path, "ao_map_path")
        if detail_map_path and isinstance(detail_map_path, str):
            ImgUtils.assert_pathlike(detail_map_path, "detail_map_path")

        if save and output_path is None:
            source_map = (
                metallic_map_path
                or roughness_map_path
                or ao_map_path
                or detail_map_path
            )
            if not source_map:
                raise ValueError("No source maps provided to derive output name")

            base_name = cls.get_base_texture_name(source_map)

            if output_dir is None:
                if isinstance(source_map, str):
                    output_dir = os.path.dirname(source_map)
                else:
                    raise ValueError(
                        "Cannot derive output directory from Image object; provide output_dir explicitly"
                    )
            elif not os.path.isdir(output_dir):
                raise ValueError(
                    f"The specified output directory '{output_dir}' is not valid."
                )

            output_path = os.path.join(
                output_dir, f"{base_name}{suffix}.{DEFAULT_EXTENSION}"
            )
        elif not save:
            output_path = None

        if layout == "rgba":
            # Mirror of MSAO: R=Metallic, G=AO, B=Detail, A=Roughness
            return ImgUtils.pack_channels(
                channel_files={
                    "R": metallic_map_path,
                    "G": ao_map_path,
                    "B": detail_map_path,
                    "A": roughness_map_path,
                },
                output_path=output_path,
                out_mode="RGBA",
                invert_channels=["A"] if invert_roughness else None,
                fill_values={"R": 0, "G": 255, "B": 0, "A": 0},
                save=save,
            )

        # Default 3-channel industry layout: R=Metallic, G=Roughness, B=AO
        return ImgUtils.pack_channels(
            channel_files={
                "R": metallic_map_path,
                "G": roughness_map_path,
                "B": ao_map_path,
            },
            output_path=output_path,
            out_mode="RGB",
            invert_channels=["G"] if invert_roughness else None,
            fill_values={"R": 0, "G": 0, "B": 255},
            save=save,
        )

    @classmethod
    def foreign_packings(
        cls,
        sources: Iterable[Any],
        target: str = "ORM",
        workflow: Optional[str] = None,
    ) -> Dict[str, str]:
        """Body of :meth:`MapFactory.foreign_packings`."""
        registry = MapRegistry.instance()
        if workflow is not None and workflow not in registry.get_workflow_presets():
            cls.logger.warning(
                "foreign_packings: unknown workflow %r - reporting nothing. "
                "Known workflows: %s.",
                workflow,
                ", ".join(registry.get_workflow_presets()),
            )
            return {}
        found: Dict[str, str] = {}
        for src in sources:
            if not isinstance(src, str) or not src or src in found:
                continue
            map_type = cls.resolve_map_type(src)
            entry = registry.get(map_type) if map_type else None
            if not entry or not entry.is_packed or not entry.workflows:
                continue
            if workflow is not None:
                if workflow not in entry.workflows:
                    found[src] = map_type
            elif registry.shares_workflow(map_type, target) is False:
                found[src] = map_type
        return found

    @classmethod
    def unpack_to_channels(
        cls,
        source: Union[str, "Image.Image"],
        map_type: Optional[str] = None,
        save: bool = False,
    ) -> Dict[str, "Image.Image"]:
        """Body of :meth:`MapFactory.unpack_to_channels`."""
        if map_type is None and isinstance(source, str):
            map_type = cls.resolve_map_type(source)
        entry = cls.PACKED_UNPACKERS.get(map_type or "")
        if not entry:
            return {}
        method, carried = entry
        unpacked = getattr(cls, method)(source, save=save)
        return {
            name: image for name, image in zip(carried, unpacked) if image is not None
        }

    @classmethod
    def _resolve_orm_sources(
        cls,
        ao: Optional[Union[str, "Image.Image"]],
        roughness: Optional[Union[str, "Image.Image"]],
        metallic: Optional[Union[str, "Image.Image"]],
    ) -> Tuple[Any, Any, Any]:
        """Expand any packed map among the three ORM sources into loose channels.

        A packed source map in *any* of the three slots supplies **every**
        channel it carries, not just the slot it happened to arrive in -- that
        is what packing means. Without this, a caller holding only an MSAO map
        can describe it to :meth:`pack_orm_texture` in exactly one way (as one
        of the three slots), and every way is wrong: the packed RGBA is
        flattened to luminance for that one channel and the other two fall back
        to their fill values. Measured on a production room, MSAO named as the
        metallic source produced roughness **0** (mirror-smooth) and metallic
        0.43 against a true 0.016 -- worse than the unrepaired conversion,
        because it overwrote a roughly-correct ORM with a confidently wrong one.

        A loose map the caller passed explicitly always wins over the same
        channel recovered from a packed one: naming both means "use the packed
        map for what the loose maps don't cover".
        """
        # Classified once and threaded through: `resolve_map_type` parses the
        # filename against the whole alias list, and `unpack_to_channels` would
        # otherwise repeat it per source.
        packed = {}
        for src in (ao, roughness, metallic):
            if isinstance(src, str) and src not in packed:
                map_type = cls.resolve_map_type(src)
                if map_type in cls.PACKED_UNPACKERS:
                    packed[src] = map_type
        if not packed:
            return ao, roughness, metallic

        slots = {
            "Ambient_Occlusion": ao,
            "Roughness": roughness,
            "Metallic": metallic,
        }
        # Handled, but say so. A mask map from another engine family unpacks and
        # repacks correctly, and staying silent means the mismatch is never fixed
        # at the source -- while every push pays for a full-resolution channel
        # split and an 8-bit round trip (MSAO carries smoothness, so roughness is
        # reconstructed by inversion rather than read from an authored map), and
        # any channel ORM has no slot for is dropped.
        registry = MapRegistry.instance()
        foreign = cls.foreign_packings(packed, target="ORM")
        for src, map_type in foreign.items():
            cls.logger.warning(
                "pack_orm_texture: %r is a %s map (targets %s), not an "
                "ORM-family packing (targets %s). Unpacked and repacked it, "
                "but re-exporting the source set for an ORM target avoids "
                "the conversion.",
                os.path.basename(src),
                map_type,
                ", ".join(registry.get(map_type).workflows) or "unspecified",
                ", ".join(registry.get("ORM").workflows),
            )

        for src, map_type in packed.items():  # one unpack per distinct map
            carried = cls.unpack_to_channels(src, map_type=map_type)
            if (
                not any(name in slots for name in carried)
                and "Smoothness" not in carried
            ):
                # A packed map that carries no ORM channel at all (an
                # Albedo_Transparency, say) cannot fill any slot, and its path
                # is cleared below -- so if it was the only source, the pack
                # then fails with "no input images provided", which names
                # nothing useful. Say what was actually wrong here.
                cls.logger.warning(
                    "pack_orm_texture: %r is a %s map, which carries none of "
                    "occlusion/roughness/metallic - ignoring it.",
                    os.path.basename(src),
                    map_type,
                )
            if "Roughness" not in carried and "Smoothness" in carried:
                # glTF and every ORM consumer want roughness; the mask-map
                # family stores its inverse. Dropping the inversion is the
                # subtler half of the same bug -- it previews as "everything
                # is shiny where it should be matte", which reads as a bad
                # material rather than as a lost conversion.
                carried["Roughness"] = cls.convert_smoothness_to_roughness(
                    carried.pop("Smoothness"), save=False
                )
            for name, image in carried.items():
                if name not in slots:
                    continue  # a channel ORM has no slot for (Detail, Opacity)
                # Equality, not identity: the same packed map named in
                # several slots arrives as EQUAL strings that are DISTINCT
                # objects whenever the caller's spec crossed a JSON boundary
                # (the export sidecar). Identity recognised only the first
                # slot; the others were then cleared below and took their
                # black fills -- roughness 0, metallic 0, AO intact.
                if slots[name] == src or not slots[name]:
                    slots[name] = image
        # A slot still holding a packed path is one that map carries no channel
        # for; leaving the path would flatten the whole packed image into it.
        for name, value in slots.items():
            if isinstance(value, str) and value in packed:
                slots[name] = None
        return slots["Ambient_Occlusion"], slots["Roughness"], slots["Metallic"]

    @classmethod
    def unpack_orm_texture(
        cls,
        orm_map_path: str,
        output_dir: str = None,
        ao_suffix: str = "_AO",
        roughness_suffix: str = "_Roughness",
        metallic_suffix: str = "_Metallic",
        invert_roughness: bool = False,
        save: bool = True,
        **kwargs,
    ) -> Union[
        Tuple[str, str, str], Tuple["Image.Image", "Image.Image", "Image.Image"]
    ]:
        """Body of :meth:`MapFactory.unpack_orm_texture`."""
        channel_config = {
            "R": {"suffix": ao_suffix},
            "G": {"suffix": roughness_suffix, "invert": invert_roughness},
            "B": {"suffix": metallic_suffix},
        }

        results = ImgUtils.extract_channels(
            orm_map_path, channel_config, output_dir=output_dir, save=save, **kwargs
        )
        return results.get("R"), results.get("G"), results.get("B")

    @staticmethod
    def _detect_packed_layout(source: Union[str, "Image.Image"]) -> str:
        """Return ``"rgba"`` if ``source`` has an alpha channel, else ``"rgb"``."""
        try:
            img = ImgUtils.ensure_image(source)
            return "rgba" if "A" in img.getbands() else "rgb"
        except Exception:
            return "rgba"

    @classmethod
    def unpack_msao_texture(
        cls,
        msao_map_path: str,
        output_dir: str = None,
        metallic_suffix: str = "_Metallic",
        ao_suffix: str = "_AO",
        smoothness_suffix: str = "_Smoothness",
        invert_smoothness: bool = False,
        save: bool = True,
        layout: Optional[str] = None,
        **kwargs,
    ) -> Union[
        Tuple[str, str, str], Tuple["Image.Image", "Image.Image", "Image.Image"]
    ]:
        """Body of :meth:`MapFactory.unpack_msao_texture`."""
        resolved_layout = (layout or "").lower() or cls._detect_packed_layout(
            msao_map_path
        )
        if resolved_layout not in ("rgba", "rgb"):
            raise ValueError(f"Unsupported MSAO layout: {layout!r}")

        if resolved_layout == "rgb":
            channel_config = {
                "R": {"suffix": metallic_suffix},
                "G": {"suffix": smoothness_suffix, "invert": invert_smoothness},
                "B": {"suffix": ao_suffix},
            }
            results = ImgUtils.extract_channels(
                msao_map_path,
                channel_config,
                output_dir=output_dir,
                save=save,
                **kwargs,
            )
            # (metallic, ao, smoothness)
            return results.get("R"), results.get("B"), results.get("G")

        channel_config = {
            "R": {"suffix": metallic_suffix},
            "G": {"suffix": ao_suffix},
            "A": {"suffix": smoothness_suffix, "invert": invert_smoothness},
        }
        results = ImgUtils.extract_channels(
            msao_map_path, channel_config, output_dir=output_dir, save=save, **kwargs
        )
        return results.get("R"), results.get("G"), results.get("A")

    @classmethod
    def unpack_mrao_texture(
        cls,
        mrao_map_path: str,
        output_dir: str = None,
        metallic_suffix: str = "_Metallic",
        roughness_suffix: str = "_Roughness",
        ao_suffix: str = "_AO",
        invert_roughness: bool = False,
        save: bool = True,
        layout: Optional[str] = None,
        **kwargs,
    ) -> Union[
        Tuple[str, str, str], Tuple["Image.Image", "Image.Image", "Image.Image"]
    ]:
        """Body of :meth:`MapFactory.unpack_mrao_texture`."""
        resolved_layout = (layout or "").lower() or cls._detect_packed_layout(
            mrao_map_path
        )
        if resolved_layout not in ("rgb", "rgba"):
            raise ValueError(f"Unsupported MRAO layout: {layout!r}")

        if resolved_layout == "rgba":
            channel_config = {
                "R": {"suffix": metallic_suffix},
                "G": {"suffix": ao_suffix},
                "A": {"suffix": roughness_suffix, "invert": invert_roughness},
            }
            results = ImgUtils.extract_channels(
                mrao_map_path,
                channel_config,
                output_dir=output_dir,
                save=save,
                **kwargs,
            )
            # (metallic, roughness, ao)
            return results.get("R"), results.get("A"), results.get("G")

        channel_config = {
            "R": {"suffix": metallic_suffix},
            "G": {"suffix": roughness_suffix, "invert": invert_roughness},
            "B": {"suffix": ao_suffix},
        }
        results = ImgUtils.extract_channels(
            mrao_map_path, channel_config, output_dir=output_dir, save=save, **kwargs
        )
        return results.get("R"), results.get("G"), results.get("B")

    @classmethod
    def unpack_albedo_transparency(
        cls,
        albedo_map_path: str,
        output_dir: str = None,
        base_color_suffix: str = "_BaseColor",
        opacity_suffix: str = "_Opacity",
        save: bool = True,
        **kwargs,
    ) -> Union[Tuple[str, str], Tuple["Image.Image", "Image.Image"]]:
        """Body of :meth:`MapFactory.unpack_albedo_transparency`."""
        channel_config = {
            "RGB": {"suffix": base_color_suffix},
            "A": {"suffix": opacity_suffix},
        }

        results = ImgUtils.extract_channels(
            albedo_map_path, channel_config, output_dir=output_dir, save=save, **kwargs
        )
        return results.get("RGB"), results.get("A")

    @classmethod
    def unpack_metallic_smoothness(
        cls,
        map_path: str,
        output_dir: str = None,
        metallic_suffix: str = "_Metallic",
        smoothness_suffix: str = "_Smoothness",
        invert_smoothness: bool = False,
        save: bool = True,
        **kwargs,
    ) -> Union[Tuple[str, str], Tuple["Image.Image", "Image.Image"]]:
        """Body of :meth:`MapFactory.unpack_metallic_smoothness`."""
        channel_config = {
            "RGB": {"suffix": metallic_suffix},
            "A": {"suffix": smoothness_suffix, "invert": invert_smoothness},
        }

        results = ImgUtils.extract_channels(
            map_path, channel_config, output_dir=output_dir, save=save, **kwargs
        )
        return results.get("RGB"), results.get("A")

    @classmethod
    def unpack_specular_gloss(
        cls,
        map_path: str,
        output_dir: str = None,
        specular_suffix: str = "_Specular",
        gloss_suffix: str = "_Glossiness",
        invert_gloss: bool = False,
        save: bool = True,
        **kwargs,
    ) -> Union[Tuple[str, str], Tuple["Image.Image", "Image.Image"]]:
        """Body of :meth:`MapFactory.unpack_specular_gloss`."""
        channel_config = {
            "RGB": {"suffix": specular_suffix},
            "A": {"suffix": gloss_suffix, "invert": invert_gloss},
        }

        results = ImgUtils.extract_channels(
            map_path, channel_config, output_dir=output_dir, save=save, **kwargs
        )
        return results.get("RGB"), results.get("A")
