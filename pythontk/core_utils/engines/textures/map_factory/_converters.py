# !/usr/bin/python
# coding=utf-8
"""The converter implementations ``register_conversions`` wires into the
``conversions`` registry: normal-map handedness and bump -> normal,
specular/gloss -> PBR, smoothness <-> roughness (the bodies behind the
:class:`MapFactory` facade).
"""

from __future__ import annotations

import os
from typing import Any, Optional, Tuple, Union

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore
try:
    from PIL import Image, ImageOps, ImageEnhance, ImageFilter
except ImportError:
    Image = ImageOps = ImageEnhance = ImageFilter = None  # type: ignore

from pythontk.img_utils._img_utils import ImgUtils
from pythontk.file_utils._file_utils import FileUtils
from .conversions import ConversionRegistry
from .processor import DEFAULT_EXTENSION


class _MapConverterInternal:
    """Bodies of :class:`MapFactory`'s map-to-map converters.

    The public signatures and docstrings stay on :class:`MapFactory`; each
    delegates here. A ``MapFactory`` base, so ``cls`` is the factory (or a
    subclass) and its registries and helpers resolve through the MRO.
    """

    @classmethod
    def register_conversions(cls, registry: ConversionRegistry):
        """Body of :meth:`MapFactory.register_conversions`."""
        # Metallic conversions
        registry.register(
            "Metallic",
            "Specular",
            lambda inv, ctx: ctx.convert_specular_to_metallic(inv["Specular"]),
            priority=5,
        )

        # Roughness conversions
        registry.register(
            "Roughness",
            "Smoothness",
            lambda inv, ctx: ctx.convert_smoothness_to_roughness(inv["Smoothness"]),
            priority=10,
        )
        registry.register(
            "Roughness",
            "Glossiness",
            lambda inv, ctx: ctx.convert_smoothness_to_roughness(inv["Glossiness"]),
            priority=9,
        )
        registry.register(
            "Roughness",
            "Specular",
            lambda inv, ctx: ctx.convert_specular_to_roughness(inv["Specular"]),
            priority=5,
        )

        # Glossiness conversions
        registry.register(
            "Glossiness",
            "Specular",
            lambda inv, ctx: ctx.extract_gloss_from_spec(inv["Specular"]),
            priority=5,
        )
        registry.register(
            "Glossiness",
            "Roughness",
            lambda inv, ctx: ctx.convert_roughness_to_smoothness(
                inv["Roughness"]
            ),  # Inverted Roughness = Smoothness ≈ Glossiness
            priority=9,
        )
        registry.register(
            "Glossiness",
            "Smoothness",
            lambda inv, ctx: ctx.copy_map(inv["Smoothness"], "Glossiness"),
            priority=10,
        )

        # Smoothness conversions
        registry.register(
            "Smoothness",
            "Roughness",
            lambda inv, ctx: ctx.convert_roughness_to_smoothness(inv["Roughness"]),
            priority=10,
        )

        # Normal conversions
        registry.register(
            "Normal_OpenGL",
            "Normal_DirectX",
            lambda inv, ctx: ctx.convert_dx_to_gl(inv["Normal_DirectX"]),
            priority=10,
        )
        registry.register(
            "Normal_DirectX",
            "Normal_OpenGL",
            lambda inv, ctx: ctx.convert_gl_to_dx(inv["Normal_OpenGL"]),
            priority=10,
        )

        # Bump/Height to Normal conversions. Bind the loop var as a default
        # arg — a plain closure late-binds, leaving every registration
        # reading inv["Height"] (KeyError when only a Bump map exists).
        for target in ["Normal_OpenGL", "Normal_DirectX", "Normal"]:
            for source in ["Bump", "Height"]:
                registry.register(
                    target,
                    source,
                    lambda inv, ctx, s=source: ctx.convert_bump_to_normal(inv[s]),
                    priority=5,
                )
        registry.register(
            "Normal",
            ["Bump", "Height"],
            lambda inv, ctx: ctx.convert_bump_to_normal(
                inv.get("Bump") or inv["Height"]
            ),
            priority=5,
        )

        # Packing conversions (ORM)
        # Priority 10: All components present, native Roughness
        registry.register(
            "ORM",
            ["Metallic", "Roughness", "Ambient_Occlusion"],
            lambda inv, ctx: ctx.create_orm_map(inv),
            priority=10,
        )
        # Priority 9: All components present, converted Smoothness
        registry.register(
            "ORM",
            ["Metallic", "Smoothness", "Ambient_Occlusion"],
            lambda inv, ctx: ctx.create_orm_map(inv),
            priority=9,
        )
        # Priority 8: Missing AO, native Roughness
        registry.register(
            "ORM",
            ["Metallic", "Roughness"],
            lambda inv, ctx: ctx.create_orm_map(inv),
            priority=8,
        )
        # Priority 7: Missing AO, converted Smoothness
        registry.register(
            "ORM",
            ["Metallic", "Smoothness"],
            lambda inv, ctx: ctx.create_orm_map(inv),
            priority=7,
        )

        # Packing conversions (MSAO/MaskMap)
        # Priority 10: All components present, native Smoothness
        registry.register(
            "MSAO",
            ["Metallic", "Ambient_Occlusion", "Smoothness"],
            lambda inv, ctx: ctx.create_mask_map(inv),
            priority=10,
        )
        # Priority 9: All components present, converted Roughness
        registry.register(
            "MSAO",
            ["Metallic", "Ambient_Occlusion", "Roughness"],
            lambda inv, ctx: ctx.create_mask_map(inv),
            priority=9,
        )
        # Priority 8: Missing AO, native Smoothness
        registry.register(
            "MSAO",
            ["Metallic", "Smoothness"],
            lambda inv, ctx: ctx.create_mask_map(inv),
            priority=8,
        )
        # Priority 7: Missing AO, converted Roughness
        registry.register(
            "MSAO",
            ["Metallic", "Roughness"],
            lambda inv, ctx: ctx.create_mask_map(inv),
            priority=7,
        )
        # Priority 6: Missing Smoothness (Metallic + AO)
        registry.register(
            "MSAO",
            ["Metallic", "Ambient_Occlusion"],
            lambda inv, ctx: ctx.create_mask_map(inv),
            priority=6,
        )
        # Priority 5: Missing Metallic (Smoothness + AO)
        registry.register(
            "MSAO",
            ["Ambient_Occlusion", "Smoothness"],
            lambda inv, ctx: ctx.create_mask_map(inv),
            priority=5,
        )

        # Packing conversions (Metallic_Smoothness)
        registry.register(
            "Metallic_Smoothness",
            ["Metallic", "Smoothness"],
            lambda inv, ctx: ctx.create_metallic_smoothness_map(inv),
            priority=10,
        )
        registry.register(
            "Metallic_Smoothness",
            ["Metallic", "Roughness"],
            lambda inv, ctx: ctx.create_metallic_smoothness_map(inv),
            priority=9,
        )

        # Unpacking conversions (Metallic_Smoothness)
        registry.register(
            "Metallic",
            "Metallic_Smoothness",
            lambda inv, ctx: ctx.get_metallic_from_packed(inv["Metallic_Smoothness"]),
            priority=8,
        )

        registry.register(
            "Smoothness",
            "Metallic_Smoothness",
            lambda inv, ctx: ctx.get_smoothness_from_packed(inv["Metallic_Smoothness"]),
            priority=8,
        )
        registry.register(
            "Roughness",
            "Metallic_Smoothness",
            lambda inv, ctx: ctx.get_roughness_from_packed(inv["Metallic_Smoothness"]),
            priority=8,
        )

        # Unpacking conversions (MSAO)
        registry.register(
            "Metallic",
            "MSAO",
            lambda inv, ctx: ctx.get_metallic_from_msao(inv["MSAO"]),
            priority=8,
        )
        registry.register(
            "Smoothness",
            "MSAO",
            lambda inv, ctx: ctx.get_smoothness_from_msao(inv["MSAO"]),
            priority=8,
        )
        registry.register(
            "Roughness",
            "MSAO",
            lambda inv, ctx: ctx.get_roughness_from_msao(inv["MSAO"]),
            priority=8,
        )
        registry.register(
            "Ambient_Occlusion",
            "MSAO",
            lambda inv, ctx: ctx.get_ao_from_msao(inv["MSAO"]),
            priority=8,
        )
        registry.register(
            "AO",
            "MSAO",
            lambda inv, ctx: ctx.get_ao_from_msao(inv["MSAO"]),
            priority=8,
        )

        # Packing conversions (MRAO)
        # Priority 10: All components present, native Roughness
        registry.register(
            "MRAO",
            ["Metallic", "Roughness", "Ambient_Occlusion"],
            lambda inv, ctx: ctx.create_mrao_map(inv),
            priority=10,
        )
        # Priority 9: All components present, converted Smoothness
        registry.register(
            "MRAO",
            ["Metallic", "Smoothness", "Ambient_Occlusion"],
            lambda inv, ctx: ctx.create_mrao_map(inv),
            priority=9,
        )
        # Priority 8: Missing AO, native Roughness
        registry.register(
            "MRAO",
            ["Metallic", "Roughness"],
            lambda inv, ctx: ctx.create_mrao_map(inv),
            priority=8,
        )
        # Priority 7: Missing AO, converted Smoothness
        registry.register(
            "MRAO",
            ["Metallic", "Smoothness"],
            lambda inv, ctx: ctx.create_mrao_map(inv),
            priority=7,
        )

        # Unpacking conversions (MRAO)
        registry.register(
            "Metallic",
            "MRAO",
            lambda inv, ctx: ctx.get_metallic_from_mrao(inv["MRAO"]),
            priority=8,
        )
        registry.register(
            "Roughness",
            "MRAO",
            lambda inv, ctx: ctx.get_roughness_from_mrao(inv["MRAO"]),
            priority=8,
        )
        registry.register(
            "Smoothness",
            "MRAO",
            lambda inv, ctx: ctx.get_smoothness_from_mrao(inv["MRAO"]),
            priority=8,
        )
        registry.register(
            "Ambient_Occlusion",
            "MRAO",
            lambda inv, ctx: ctx.get_ao_from_mrao(inv["MRAO"]),
            priority=8,
        )
        registry.register(
            "AO",
            "MRAO",
            lambda inv, ctx: ctx.get_ao_from_mrao(inv["MRAO"]),
            priority=8,
        )

        # Unpacking conversions (ORM)
        registry.register(
            "Ambient_Occlusion",
            "ORM",
            lambda inv, ctx: ctx.get_ao_from_orm(inv["ORM"]),
            priority=8,
        )
        registry.register(
            "AO",
            "ORM",
            lambda inv, ctx: ctx.get_ao_from_orm(inv["ORM"]),
            priority=8,
        )
        registry.register(
            "Roughness",
            "ORM",
            lambda inv, ctx: ctx.get_roughness_from_orm(inv["ORM"]),
            priority=8,
        )
        registry.register(
            "Smoothness",
            "ORM",
            lambda inv, ctx: ctx.get_smoothness_from_orm(inv["ORM"]),
            priority=8,
        )
        registry.register(
            "Metallic",
            "ORM",
            lambda inv, ctx: ctx.get_metallic_from_orm(inv["ORM"]),
            priority=8,
        )

        # Unpacking conversions (Albedo_Transparency)
        registry.register(
            "Base_Color",
            "Albedo_Transparency",
            lambda inv, ctx: ctx.get_base_color_from_albedo_transparency(
                inv["Albedo_Transparency"]
            ),
            priority=8,
        )
        registry.register(
            "Opacity",
            "Albedo_Transparency",
            lambda inv, ctx: ctx.get_opacity_from_albedo_transparency(
                inv["Albedo_Transparency"]
            ),
            priority=8,
        )

    @classmethod
    def detect_normal_map_format(
        cls,
        image: Union[str, "Image.Image"],
        threshold: float = 0.25,
        min_gradient_std: float = 1.0,
    ) -> Optional[str]:
        """Body of :meth:`MapFactory.detect_normal_map_format`."""
        try:
            # convert("RGB") always returns our own copy (PIL copies even when
            # the mode already matches), so the in-place thumbnail() below never
            # mutates a caller-supplied Image.
            img = ImgUtils.ensure_image(image).convert("RGB")

            # Reducing first keeps the common case cheap, but it is a LOW-PASS
            # over exactly the gradients this statistic reads, so it cannot be
            # the last word: on maps whose relief is fine and shallow it
            # averages the signal flat (measured on real OpenGL bakes: r fell
            # -0.368 -> -0.105 and -0.19 -> -0.09, both under the threshold, a
            # correct answer downgraded to "don't know"). So the reduction is a
            # FAST PATH -- taken when it answers, re-read at native resolution
            # when it does not. Full res costs ~68 ms on a 2048 map against the
            # ~50 ms already spent decoding it, so the escalation is cheap and
            # only the indeterminate minority pays it.
            reduced = img
            if max(img.size) > 512:
                # `resize` rather than `copy() + thumbnail()`: thumbnail is
                # in-place, and the full-size image has to survive for the
                # re-read below, so taking it that way costs a full-size copy
                # first (48 MB on a 4k map). Same `reducing_gap` two-step, same
                # aspect rule, byte-identical output, and measurably faster.
                width, height = img.size
                scale = 512 / max(width, height)
                reduced = img.resize(
                    (max(1, round(width * scale)), max(1, round(height * scale))),
                    reducing_gap=2.0,
                )

            correlation = cls._normal_handedness_correlation(reduced, min_gradient_std)
            if (correlation is None or abs(correlation) <= threshold) and (
                reduced is not img
            ):
                full = cls._normal_handedness_correlation(img, min_gradient_std)
                if full is not None:
                    correlation = full

            if correlation is None:
                return None
            if correlation < -threshold:
                return "OpenGL"
            if correlation > threshold:
                return "DirectX"
            return None

        except Exception as e:
            cls.logger.warning(f"Error detecting normal map format: {e}")
            return None

    @staticmethod
    def _normal_handedness_correlation(
        img: "Image.Image", min_gradient_std: float
    ) -> Optional[float]:
        """``corr(dR/dy, dG/dx)`` for *img*, or ``None`` if it is meaningless.

        The integrability statistic behind :meth:`detect_normal_map_format`,
        split out so the same computation serves both the reduced fast path and
        the native-resolution re-read. Negative = OpenGL, positive = DirectX;
        the caller owns the threshold.

        ``None`` means "no usable signal here", not "flat": either channel's
        gradient falling under *min_gradient_std* (8-bit units) makes the
        correlation noise, and a non-finite result (a constant channel) is the
        same answer arrived at by division.
        """
        # Only R and G carry the signal, so only R and G are materialized --
        # `np.array(img)` would build the blue plane too, a third of the
        # allocation for nothing (measured on a 2048 map: 218 -> 201 MB peak,
        # and marginally faster). Identical correlation to 0e+00.
        dRy = np.gradient(  # dR/dy along image rows
            np.asarray(img.getchannel("R"), dtype=np.float32), axis=0
        ).ravel()
        dGx = np.gradient(  # dG/dx along image cols
            np.asarray(img.getchannel("G"), dtype=np.float32), axis=1
        ).ravel()
        # Variance floor: flat or near-flat inputs produce meaningless
        # correlations (often NaN, often spuriously signed).
        if dRy.std() < min_gradient_std or dGx.std() < min_gradient_std:
            return None
        correlation = np.corrcoef(dRy, dGx)[0, 1]
        return float(correlation) if np.isfinite(correlation) else None

    @classmethod
    def convert_normal_map_format(
        cls,
        file: str,
        target_format: str,
        output_path: str = None,
        save: bool = True,
        **kwargs,
    ) -> Union[str, "Image.Image"]:
        """Body of :meth:`MapFactory.convert_normal_map_format`."""
        ImgUtils.assert_pathlike(file, "file")

        target_format = target_format.lower()
        if target_format not in ("opengl", "directx"):
            raise ValueError("target_format must be 'opengl' or 'directx'")

        # Determine source format for validation and naming
        if target_format == "opengl":
            source_type_key = "Normal_DirectX"
            target_type_key = "Normal_OpenGL"
        else:
            source_type_key = "Normal_OpenGL"
            target_type_key = "Normal_DirectX"

        try:
            typ = cls.resolve_map_type(file, key=False, validate=source_type_key)
        except ValueError:
            try:
                typ = cls.resolve_map_type(file, key=False, validate="Normal")
            except ValueError:
                typ = ""

        inverted_image = ImgUtils.invert_channels(file, "g")

        if not save:
            return inverted_image

        if output_path is None:
            output_dir = FileUtils.format_path(file, "path")
            name = FileUtils.format_path(file, "name")
            ext = FileUtils.format_path(file, "ext")

            # Keep the source file's naming style by swapping only the
            # convention tag. This used to pair the two alias tuples by INDEX,
            # which silently depended on them being the same length and in
            # lockstep order.
            new_suffix = target_type_key
            if typ:
                if typ in cls.map_types[source_type_key]:
                    new_suffix = cls._map_registry.counterpart_normal_spelling(
                        typ, target_type_key
                    )

                name = name.removesuffix(typ)

            output_path = f"{output_dir}/{name}{new_suffix}.{ext}"

        output_path = os.path.abspath(output_path)
        ImgUtils.save_image(inverted_image, output_path, **kwargs)
        return output_path

    @classmethod
    def convert_bump_to_normal(
        cls,
        bump_map: Union[str, "Image.Image"],
        output_path: str = None,
        intensity: float = 1.0,
        output_format: str = "opengl",
        smooth_filter: bool = True,
        filter_radius: float = 0.5,
        edge_wrap: bool = False,
        save: bool = True,
        **kwargs,
    ) -> Union[str, "Image.Image"]:
        """Body of :meth:`MapFactory.convert_bump_to_normal`."""
        # Load and ensure grayscale; validate path only when a path is provided
        if isinstance(bump_map, str):
            ImgUtils.assert_pathlike(bump_map, "bump_map")
        image = ImgUtils.ensure_image(bump_map, "L")

        # Apply smoothing filter to reduce aliasing if requested
        if smooth_filter and filter_radius > 0:
            # Use Gaussian blur to smooth height data before gradient calculation
            image = image.filter(ImageFilter.GaussianBlur(radius=filter_radius))

        # Convert to numpy array for gradient calculations
        height_srgb = np.asarray(image, dtype=np.float32) / 255.0

        # Convert sRGB grayscale to linear before computing derivatives (safer filtering/derivatives)
        height_lin = ImgUtils._srgb_to_linear_np(height_srgb)

        # Calculate gradients using Sobel operator (industry standard)
        # Sobel X kernel: [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        # Sobel Y kernel: [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
        if edge_wrap:
            # Pad with wrapped edges for seamless tiling
            padded = np.pad(height_lin, 1, mode="wrap")
        else:
            # Pad with edge values
            padded = np.pad(height_lin, 1, mode="edge")

        # Sobel X gradient (horizontal edges)
        grad_x = (
            -1 * padded[:-2, :-2]
            + 1 * padded[:-2, 2:]
            + -2 * padded[1:-1, :-2]
            + 2 * padded[1:-1, 2:]
            + -1 * padded[2:, :-2]
            + 1 * padded[2:, 2:]
        ) / 8.0

        # Sobel Y gradient (vertical edges)
        grad_y = (
            -1 * padded[:-2, :-2]
            + -2 * padded[:-2, 1:-1]
            + -1 * padded[:-2, 2:]
            + 1 * padded[2:, :-2]
            + 2 * padded[2:, 1:-1]
            + 1 * padded[2:, 2:]
        ) / 8.0

        # Scale gradients by intensity
        grad_x *= intensity
        grad_y *= intensity

        # Calculate normal vectors. The surface normal of z=H is
        # (-dH/dx, -dH/dy_up, 1). grad_y is the IMAGE-ROW derivative
        # (row increases downward), and textures display right side up
        # (image top = V max), so dH/dy_up = -grad_y and the green (Y-up)
        # component is +grad_y. The old `-grad_y` silently produced
        # DirectX orientation under an OpenGL label (verified against a
        # labeled real-world map via the integrability correlation).
        normal_x = -grad_x
        normal_y = grad_y
        normal_z = np.ones_like(grad_x)

        # Normalize the normal vectors (with epsilon to avoid division by zero)
        length = np.sqrt(normal_x**2 + normal_y**2 + normal_z**2)
        length = np.maximum(length, 1e-8)
        normal_x /= length
        normal_y /= length
        normal_z /= length

        # Handle DirectX vs OpenGL Y-channel orientation
        if output_format.lower() == "directx":
            # DirectX expects Y+ to point down, so invert Y component
            normal_y = -normal_y
        # OpenGL is the default (Y+ points up)

        # Convert from [-1,1] to [0,255] range for RGB channels
        # R = X component, G = Y component, B = Z component
        red_f = (normal_x + 1.0) * 127.5
        green_f = (normal_y + 1.0) * 127.5
        blue_f = (normal_z + 1.0) * 127.5

        # Clamp to valid [0,255] range before casting
        red = np.clip(red_f, 0, 255).astype(np.uint8)
        green = np.clip(green_f, 0, 255).astype(np.uint8)
        blue = np.clip(blue_f, 0, 255).astype(np.uint8)

        # Create RGB image from normal components
        normal_array = np.stack([red, green, blue], axis=-1)
        normal_image = Image.fromarray(normal_array, "RGB")

        if not save:
            return normal_image

        # Generate output path if not provided
        if output_path is None:
            if isinstance(bump_map, str):
                base_path = bump_map
            else:
                # If PIL Image was passed, create generic output name
                base_path = f"bump_map.{DEFAULT_EXTENSION}"

            format_suffix = (
                "DirectX" if output_format.lower() == "directx" else "OpenGL"
            )
            output_path = cls.resolve_texture_filename(
                base_path,
                f"Normal_{format_suffix}",
                suffix=(
                    f"_intensity{intensity}".replace(".", "p")
                    if intensity != 1.0
                    else None
                ),
            )

        # Save the normal map
        ImgUtils.save_image(normal_image, output_path, **kwargs)

        return output_path

    @classmethod
    def extract_gloss_from_spec(
        cls, specular_map: str, channel: str = "A"
    ) -> Union["Image.Image", None]:
        """Body of :meth:`MapFactory.extract_gloss_from_spec`."""
        spec = ImgUtils.ensure_image(specular_map)

        # Attempt channel extraction
        if channel.upper() in spec.getbands():
            gloss = spec.getchannel(channel.upper())
            if gloss.getextrema() != (0, 0):  # Ensure non-empty
                return gloss.convert("L")

        print(
            f"// Warning: No gloss found in '{channel}' channel; using normalized grayscale..."
        )
        spec_gray = spec.convert("L")
        spec_gray = ImageEnhance.Brightness(spec_gray).enhance(1.2)
        gloss = ImageOps.autocontrast(spec_gray)

        return gloss.convert("L")

    @classmethod
    def convert_spec_gloss_to_pbr(
        cls,
        specular_map: Union[str, "Image.Image"],
        glossiness_map: Union[str, "Image.Image"],
        diffuse_map: Union[str, "Image.Image"] = None,
        output_dir: str = None,
        convert_diffuse_to_albedo: bool = False,
        output_type: str = None,
        image_size: Optional[int] = None,
        optimize_bit_depth: bool = True,
        write_files: bool = False,
    ) -> Union[
        Tuple["Image.Image", "Image.Image", "Image.Image"], Tuple[str, str, str]
    ]:
        """Body of :meth:`MapFactory.convert_spec_gloss_to_pbr`."""
        spec = ImgUtils.ensure_image(specular_map, "RGB")
        gloss = ImgUtils.ensure_image(glossiness_map, "L")
        diffuse = ImgUtils.ensure_image(diffuse_map, "RGB") if diffuse_map else None

        metallic = cls.create_metallic_from_spec(specular_map)
        base_color = cls.create_base_color_from_spec(diffuse, spec, metallic)
        roughness = cls.create_roughness_from_spec(spec, gloss)

        if convert_diffuse_to_albedo:
            base_color = cls.convert_base_color_to_albedo(base_color, metallic)

        if optimize_bit_depth:
            base_color = ImgUtils.set_bit_depth(base_color, "Base_Color")
            metallic = ImgUtils.set_bit_depth(metallic, "Metallic")
            roughness = ImgUtils.set_bit_depth(roughness, "Roughness")

        # Optional downscale to target max dimension while preserving original if not requested
        if isinstance(image_size, int) and image_size > 0:
            if max(base_color.size) > image_size:
                base_color = ImgUtils.resize_image(base_color, image_size, image_size)
            if max(metallic.size) > image_size:
                metallic = ImgUtils.resize_image(metallic, image_size, image_size)
            if max(roughness.size) > image_size:
                roughness = ImgUtils.resize_image(roughness, image_size, image_size)

        if not write_files:
            return base_color, metallic, roughness

        if output_dir is None:
            output_dir = (
                os.path.dirname(specular_map)
                if isinstance(specular_map, str)
                else os.getcwd()
            )
        elif not os.path.isdir(output_dir):
            raise ValueError(
                f"The specified output directory '{output_dir}' is not valid."
            )

        # Output filenames are derived from the specular_map path; a PIL Image
        # has no path, so fail clearly here instead of raising an obscure
        # TypeError from resolve_texture_filename's assert_pathlike below.
        if not isinstance(specular_map, (str, os.PathLike)):
            raise ValueError(
                "convert_spec_gloss_to_pbr(write_files=True) needs a file-path "
                "specular_map to derive output filenames; pass path inputs, or "
                "call with write_files=False to receive Image objects."
            )

        base_color_type = "Albedo" if convert_diffuse_to_albedo else "Base_Color"
        base_color_file = cls.resolve_texture_filename(
            specular_map, base_color_type, ext=output_type
        )
        metallic_file = cls.resolve_texture_filename(
            specular_map, "Metallic", ext=output_type
        )
        roughness_file = cls.resolve_texture_filename(
            specular_map, "Roughness", ext=output_type
        )

        ImgUtils.save_image(base_color, base_color_file)
        ImgUtils.save_image(metallic, metallic_file)
        ImgUtils.save_image(roughness, roughness_file)

        print(
            f"PBR Conversion complete. Files saved:\n- {base_color_file}\n- {metallic_file}\n- {roughness_file}"
        )
        return base_color_file, metallic_file, roughness_file

    @classmethod
    def create_base_color_from_spec(
        cls,
        diffuse: Union[str, "Image.Image"],
        spec: Union[str, "Image.Image"],
        metalness: Union[str, "Image.Image"],
        conserve_energy: bool = True,
        metal_darkening: float = 0.22,
    ) -> "Image.Image":
        """Body of :meth:`MapFactory.create_base_color_from_spec`."""
        spec = np.array(ImgUtils.ensure_image(spec, "RGB"), dtype=np.float32) / 255.0
        metalness = (
            np.array(ImgUtils.ensure_image(metalness, "L"), dtype=np.float32) / 255.0
        )

        if diffuse:
            diffuse = (
                np.array(ImgUtils.ensure_image(diffuse, "RGB"), dtype=np.float32)
                / 255.0
            )
            base_color = (
                diffuse * (1 - metalness[..., None]) + spec * metalness[..., None]
            )
        else:
            base_color = spec * (1 - metalness[..., None])

        # Darken metal areas (Reduce brightness in metals)
        # NOTE: Standard PBR does not require darkening metals, but this can help
        # if the source specular map is too bright or contains baked lighting.
        if metal_darkening > 0:
            base_color = np.where(
                metalness[..., None] > 0.5,
                base_color * (1.0 - metal_darkening),
                base_color,
            )

        # Apply energy conservation fix
        # NOTE: This is an artistic tweak to boost metal brightness, not strict PBR.
        if conserve_energy:
            base_color = np.clip(
                base_color / (1.0 - 0.08 * metalness[..., None] + 1e-6), 0.0, 1.0
            )

        return Image.fromarray((base_color * 255).astype(np.uint8), mode="RGB")

    @classmethod
    def create_metallic_from_spec(
        cls,
        specular_map: Union[str, "Image.Image"],
        glossiness_map: Union[str, "Image.Image"] = None,
        threshold: int = 55,
        softness: float = 0.2,
    ) -> "Image.Image":
        """Body of :meth:`MapFactory.create_metallic_from_spec`."""
        spec_rgb = ImgUtils.ensure_image(specular_map, "RGB")
        spec_lum = np.array(spec_rgb.convert("L"), dtype=np.float32) / 255.0

        # Step 1: Get gloss
        if glossiness_map:
            gloss = (
                np.array(ImgUtils.ensure_image(glossiness_map, "L"), dtype=np.float32)
                / 255.0
            )
            print("// Using gloss map to refine metallic computation.")
        else:
            gloss_img = cls.extract_gloss_from_spec(specular_map)
            gloss = np.array(gloss_img, dtype=np.float32) / 255.0 if gloss_img else None
            if gloss is not None:
                print("// Extracted gloss from specular map.")
            else:
                print("// No valid gloss map found; using spec only.")

        # Step 2: Base metallic estimate
        metallic = np.clip((spec_lum - (threshold / 255.0)) / softness, 0.0, 1.0)

        # Step 3: Refine with gloss
        if gloss is not None:
            metallic *= 1.0 - gloss  # Reduce metallic in high-gloss regions

        return Image.fromarray((metallic * 255).astype(np.uint8), mode="L")

    @classmethod
    def create_roughness_from_spec(
        cls,
        specular_map: Union[str, "Image.Image"],
        glossiness_map: Union[str, "Image.Image"] = None,
    ) -> "Image.Image":
        """Body of :meth:`MapFactory.create_roughness_from_spec`."""
        spec = ImgUtils.ensure_image(specular_map, "RGB")

        # Step 1: Use provided gloss map or extract from specular
        gloss = (
            ImgUtils.ensure_image(glossiness_map, "L")
            if glossiness_map
            else cls.extract_gloss_from_spec(specular_map)
        )
        if not gloss:
            print(
                "// No valid gloss map found; estimating roughness directly from spec."
            )
            spec_gray = spec.convert("L")
            gloss = ImageOps.autocontrast(spec_gray)

        # Step 2: Convert glossiness to roughness
        gloss = np.array(gloss, dtype=np.float32) / 255.0
        roughness = 1.0 - gloss  # Direct inversion

        # Step 3: Apply gamma correction (for perceptual accuracy)
        gamma = 2.2  # Industry standard
        roughness = roughness**gamma

        # Step 4: Normalize roughness to maintain balanced shading
        roughness = np.clip(roughness, 0.0, 1.0)

        return Image.fromarray((roughness * 255).astype(np.uint8), mode="L")

    @classmethod
    def convert_base_color_to_albedo(
        cls, base_color: "Image.Image", metalness: "Image.Image"
    ) -> "Image.Image":
        """Body of :meth:`MapFactory.convert_base_color_to_albedo`."""
        base_color = ImgUtils.ensure_image(base_color)

        # Ensure we have at least RGB
        if base_color.mode not in ["RGB", "RGBA"]:
            base_color = base_color.convert("RGB")

        metalness = ImgUtils.ensure_image(metalness, "L")

        # Convert metalness to grayscale and threshold (Metal = 1, Non-Metal = 0)
        # Metal (>128) -> 255 (White)
        # Non-Metal (<=128) -> 0 (Black)
        metal_mask = metalness.point(lambda p: 255 if p > 128 else 0)

        # Create a black image for metals
        # Match base color mode (RGB or RGBA)
        black_image = Image.new(
            base_color.mode,
            base_color.size,
            (0, 0, 0, 0) if "A" in base_color.mode else (0, 0, 0),
        )
        # Mask 0 (Non-Metal) -> Uses base_color
        albedo = Image.composite(black_image, base_color, metal_mask)

        return albedo

    @staticmethod
    def get_converted_map(map_type: str, available: dict) -> Optional[Any]:
        """Body of :meth:`MapFactory.get_converted_map`."""
        # Deferred: the facade imports this module to build its bases.
        from pythontk.core_utils.engines.textures.map_factory._map_factory import (
            MapFactory,
        )

        # Smoothness <-> Roughness
        if map_type == "Smoothness" and "Roughness" in available:
            rough = available["Roughness"]
            return ImgUtils.invert_grayscale_image(rough)
        if map_type == "Roughness" and "Smoothness" in available:
            smooth = available["Smoothness"]
            return ImgUtils.invert_grayscale_image(smooth)
        # Glossiness <-> Roughness
        if map_type == "Glossiness" and "Roughness" in available:
            rough = available["Roughness"]
            return ImgUtils.invert_grayscale_image(rough)
        if map_type == "Roughness" and "Glossiness" in available:
            gloss = available["Glossiness"]
            return ImgUtils.invert_grayscale_image(gloss)
        # Glossiness <-> Smoothness
        if map_type == "Smoothness" and "Glossiness" in available:
            gloss = available["Glossiness"]
            return ImgUtils.invert_grayscale_image(gloss)
        if map_type == "Glossiness" and "Smoothness" in available:
            smooth = available["Smoothness"]
            return ImgUtils.invert_grayscale_image(smooth)
        # AO from Base_Color
        if map_type == "Ambient_Occlusion" and "Base_Color" in available:
            color = available["Base_Color"]
            return ImgUtils.ensure_image(color, "L")
        # Normal DirectX <-> OpenGL
        if map_type == "Normal_DirectX" and "Normal_OpenGL" in available:
            return MapFactory.convert_normal_map_format(
                available["Normal_OpenGL"], target_format="directx", save=False
            )
        if map_type == "Normal_OpenGL" and "Normal_DirectX" in available:
            return MapFactory.convert_normal_map_format(
                available["Normal_DirectX"], target_format="opengl", save=False
            )
        return None

    @classmethod
    def convert_smoothness_to_roughness(
        cls, smoothness_path: str, output_dir: str = None, save: bool = True, **kwargs
    ) -> Union[str, "Image.Image"]:
        """Body of :meth:`MapFactory.convert_smoothness_to_roughness`."""
        if isinstance(smoothness_path, str):
            ImgUtils.assert_pathlike(smoothness_path, "smoothness_path")
            if not os.path.exists(smoothness_path):
                raise FileNotFoundError(f"Input file not found: {smoothness_path}")

        # Load and invert the smoothness map
        smoothness_image = ImgUtils.ensure_image(smoothness_path, "L")
        roughness_image = ImgUtils.invert_grayscale_image(smoothness_image)

        if not save:
            return roughness_image

        if not isinstance(smoothness_path, str):
            raise ValueError(
                "Input must be a file path when save=True, or provide output_dir/name handling (not implemented for Image input)."
            )

        # Generate output path
        base_name = cls.get_base_texture_name(smoothness_path)

        if output_dir is None:
            output_dir = os.path.dirname(smoothness_path)
        elif not os.path.isdir(output_dir):
            raise ValueError(
                f"The specified output directory '{output_dir}' is not valid."
            )

        # Get original extension
        original_ext = os.path.splitext(smoothness_path)[1]
        output_path = os.path.join(output_dir, f"{base_name}_Roughness{original_ext}")

        # Save the roughness map
        ImgUtils.save_image(roughness_image, output_path, **kwargs)

        return output_path

    @classmethod
    def convert_roughness_to_smoothness(
        cls, roughness_path: str, output_dir: str = None, save: bool = True, **kwargs
    ) -> Union[str, "Image.Image"]:
        """Body of :meth:`MapFactory.convert_roughness_to_smoothness`."""
        if isinstance(roughness_path, str):
            ImgUtils.assert_pathlike(roughness_path, "roughness_path")
            if not os.path.exists(roughness_path):
                raise FileNotFoundError(f"Input file not found: {roughness_path}")

        # Load and invert the roughness map
        roughness_image = ImgUtils.ensure_image(roughness_path, "L")
        smoothness_image = ImgUtils.invert_grayscale_image(roughness_image)

        if not save:
            return smoothness_image

        if not isinstance(roughness_path, str):
            raise ValueError(
                "Input must be a file path when save=True, or provide output_dir/name handling (not implemented for Image input)."
            )

        # Generate output path
        base_name = cls.get_base_texture_name(roughness_path)

        if output_dir is None:
            output_dir = os.path.dirname(roughness_path)
        elif not os.path.isdir(output_dir):
            raise ValueError(
                f"The specified output directory '{output_dir}' is not valid."
            )

        # Get original extension
        original_ext = os.path.splitext(roughness_path)[1]
        output_path = os.path.join(output_dir, f"{base_name}_Smoothness{original_ext}")

        # Save the smoothness map
        ImgUtils.save_image(smoothness_image, output_path, **kwargs)

        return output_path
