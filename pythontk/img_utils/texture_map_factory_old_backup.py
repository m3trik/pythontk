# !/usr/bin/python
# coding=utf-8
"""Texture Map Factory for PBR workflow preparation.

Provides a DRY (Don't Repeat Yourself) factory for processing and preparing
texture maps for various PBR workflows (Unity, Unreal, glTF, etc.).
"""
import os
from typing import List, Dict, Optional, Callable

from pythontk.img_utils._img_utils import ImgUtils


class TextureMapFactory:
    """Factory class for processing and preparing PBR texture maps.

    This DRY factory handles all texture map preparation including:
    - Map detection and classification (uses ImgUtils.map_types)
    - Format conversion (DirectX ↔ OpenGL normals, Roughness ↔ Smoothness)
    - Map packing (Albedo+Transparency, Metallic+Smoothness, MSAO)
    - Specular/Glossiness to PBR conversion
    - Missing map generation from available sources

    All image operations leverage pythontk.img_utils methods.
    """

    @staticmethod
    def prepare_maps(
        textures: List[str],
        workflow_config: dict,
        callback: Callable = print,
    ) -> List[str]:
        """Main factory method to prepare all texture maps for a PBR workflow.

        Parameters:
            textures: List of source texture file paths
            workflow_config: Dictionary with keys:
                - albedo_transparency (bool): Pack opacity into albedo alpha
                - metallic_smoothness (bool): Pack smoothness into metallic alpha
                - mask_map (bool): Create Unity HDRP MSAO mask map
                - normal_type (str): "OpenGL" or "DirectX"
                - output_extension (str): Extension for generated maps
            callback: Progress/status callback function (default: print)

        Returns:
            List of processed texture file paths ready for shader connection
        """
        # Create map inventory using ImgUtils
        map_inventory = TextureMapFactory._build_map_inventory(textures)

        # Apply workflow-specific processing
        processed_maps = TextureMapFactory._apply_workflow(
            map_inventory, workflow_config, callback
        )

        # If no maps were processed, return original texture list
        # This ensures we don't lose textures that couldn't be categorized
        if not processed_maps and textures:
            return textures

        return processed_maps

    @staticmethod
    def _build_map_inventory(textures: List[str]) -> Dict[str, str]:
        """Build an inventory of available texture maps by type.

        Uses ImgUtils.map_types to detect all supported map types.

        Parameters:
            textures: List of texture file paths

        Returns:
            Dictionary mapping map types to file paths
        """
        inventory = {}

        # Use ImgUtils.map_types keys to avoid hardcoding
        for map_type in ImgUtils.map_types.keys():
            maps = ImgUtils.filter_images_by_type(textures, map_type)
            if maps:
                inventory[map_type] = maps[0]  # Take first match

        return inventory

    @staticmethod
    def _apply_workflow(
        inventory: Dict[str, str], config: dict, callback: Callable
    ) -> List[str]:
        """Apply workflow-specific transformations to map inventory.

        Parameters:
            inventory: Map type → file path dictionary
            config: Workflow configuration
            callback: Status callback

        Returns:
            List of final texture file paths
        """
        output_maps = []
        used_maps = set()  # Track which source maps have been consumed

        # Get output directory from first available map
        first_map = next(iter(inventory.values()), None)
        if not first_map:
            return []

        output_dir = os.path.dirname(first_map)
        base_name = ImgUtils.get_base_texture_name(first_map)
        ext = config.get("output_extension", "png")

        # 0. Convert Spec/Gloss workflow to PBR if detected
        if config.get("convert_specgloss_to_pbr", False):
            inventory = TextureMapFactory._convert_specgloss_workflow(
                inventory, output_dir, base_name, ext, callback
            )

        # 1. Handle Base Color / Albedo (with optional transparency packing)
        base_color_map = TextureMapFactory._prepare_base_color(
            inventory,
            config.get("albedo_transparency", False),
            config.get("cleanup_base_color", False),
            output_dir,
            base_name,
            ext,
            callback,
        )
        if base_color_map:
            output_maps.append(base_color_map)
            used_maps.update(
                [
                    "Base_Color",
                    "Diffuse",
                    "Albedo",
                    "Opacity",
                    "Transparency",
                    "Albedo_Transparency",
                ]
            )

        # 2. Handle Metallic/Smoothness workflows
        if config.get("orm_map", False):
            # Unreal/glTF ORM Map (Occlusion+Roughness+Metallic)
            orm_map = TextureMapFactory._prepare_orm_map(
                inventory, output_dir, base_name, ext, callback
            )
            if orm_map:
                output_maps.append(orm_map)
                used_maps.update(
                    [
                        "Metallic",
                        "Roughness",
                        "Smoothness",
                        "Ambient_Occlusion",
                        "Specular",
                        "Glossiness",
                        "ORM",
                    ]
                )

        elif config.get("mask_map", False):
            # Unity HDRP Mask Map (MSAO)
            mask_map = TextureMapFactory._prepare_mask_map(
                inventory, output_dir, base_name, ext, callback
            )
            if mask_map:
                output_maps.append(mask_map)
                used_maps.update(
                    [
                        "Metallic",
                        "Metallic_Smoothness",
                        "MSAO",
                        "Roughness",
                        "Smoothness",
                        "Ambient_Occlusion",
                        "Specular",
                        "Glossiness",
                    ]
                )

        elif config.get("metallic_smoothness", False):
            # Packed Metallic+Smoothness
            metallic_smooth = TextureMapFactory._prepare_metallic_smoothness(
                inventory, output_dir, base_name, ext, callback
            )
            if metallic_smooth:
                output_maps.append(metallic_smooth)
                used_maps.update(
                    [
                        "Metallic",
                        "Metallic_Smoothness",
                        "Roughness",
                        "Smoothness",
                        "Specular",
                        "Glossiness",
                    ]
                )

        else:
            # Separate Metallic and Roughness
            metallic = TextureMapFactory._prepare_metallic(
                inventory, output_dir, base_name, ext, callback
            )
            roughness = TextureMapFactory._prepare_roughness(
                inventory, output_dir, base_name, ext, callback
            )
            if metallic:
                output_maps.append(metallic)
                used_maps.update(["Metallic", "Specular"])
            if roughness:
                output_maps.append(roughness)
                used_maps.update(["Roughness", "Smoothness", "Glossiness", "Specular"])

        # 3. Handle Normal maps (with format conversion if needed)
        normal_map = TextureMapFactory._prepare_normal(
            inventory,
            config.get("normal_type", "OpenGL"),
            output_dir,
            base_name,
            ext,
            callback,
        )
        if normal_map:
            output_maps.append(normal_map)
            used_maps.update(
                ["Normal", "Normal_OpenGL", "Normal_DirectX", "Bump", "Height"]
            )

        # 4. Pass through remaining maps that weren't consumed
        passthrough_types = [
            "Emissive",
            "Emission",
            "Ambient_Occlusion",
            "Height",
            "Displacement",
        ]
        for map_type in passthrough_types:
            if map_type in inventory and map_type not in used_maps:
                output_maps.append(inventory[map_type])

        return output_maps

    @staticmethod
    def _prepare_base_color(
        inventory: Dict[str, str],
        pack_transparency: bool,
        cleanup_base_color: bool,
        output_dir: str,
        base_name: str,
        ext: str,
        callback: Callable,
    ) -> Optional[str]:
        """Prepare base color/albedo map with optional transparency packing."""
        # Check for existing combined map
        if "Albedo_Transparency" in inventory:
            return inventory["Albedo_Transparency"]

        # Get base color source (prioritize Base_Color > Diffuse > Albedo)
        base_color = (
            inventory.get("Base_Color")
            or inventory.get("Diffuse")
            or inventory.get("Albedo")
        )
        if not base_color:
            return None

        # Pack transparency if requested
        if pack_transparency:
            opacity = inventory.get("Opacity") or inventory.get("Transparency")
            if opacity:
                try:
                    combined = ImgUtils.pack_transparency_into_albedo(
                        base_color, opacity
                    )
                    callback(
                        '<br><hl style="color:rgb(100, 160, 100);">Packed transparency into albedo</hl>'
                    )
                    return combined
                except Exception as e:
                    callback(
                        f'<br><hl style="color:rgb(255, 100, 100);">Error packing transparency: {str(e)}</hl>'
                    )

        # Clean up base color if requested (remove baked reflections, fix metallic areas)
        if cleanup_base_color and base_color and "Metallic" in inventory:
            try:
                base_img = ImgUtils.load_image(base_color)
                metallic_img = ImgUtils.load_image(inventory["Metallic"])
                cleaned = ImgUtils.convert_base_color_to_albedo(base_img, metallic_img)
                cleaned_path = os.path.join(output_dir, f"{base_name}_Albedo.{ext}")
                ImgUtils.save_image(cleaned, cleaned_path)
                callback(
                    '<br><hl style="color:rgb(100, 160, 100);">Cleaned base color to true albedo</hl>'
                )
                return cleaned_path
            except Exception as e:
                callback(
                    f'<br><hl style="color:rgb(255, 100, 100);">Error cleaning base color: {str(e)}</hl>'
                )

        return base_color

    @staticmethod
    def _prepare_metallic_smoothness(
        inventory: Dict[str, str],
        output_dir: str,
        base_name: str,
        ext: str,
        callback: Callable,
    ) -> Optional[str]:
        """Prepare packed metallic+smoothness map."""
        # Check for existing combined map
        if "Metallic_Smoothness" in inventory:
            return inventory["Metallic_Smoothness"]

        # Get metallic source (or convert from specular)
        metallic = inventory.get("Metallic")
        if not metallic and "Specular" in inventory:
            try:
                metallic_img = ImgUtils.create_metallic_from_spec(inventory["Specular"])
                metallic = os.path.join(output_dir, f"{base_name}_Metallic.{ext}")
                metallic_img.save(metallic)
                callback(
                    '<br><hl style="color:rgb(100, 160, 100);">Created metallic from specular</hl>'
                )
            except Exception as e:
                callback(
                    f'<br><hl style="color:rgb(255, 100, 100);">Error creating metallic: {str(e)}</hl>'
                )
                return None

        if not metallic:
            return None

        # Get smoothness/roughness source
        if "Smoothness" in inventory:
            alpha_map = inventory["Smoothness"]
            invert = False
        elif "Roughness" in inventory:
            alpha_map = inventory["Roughness"]
            invert = True
        elif "Glossiness" in inventory:
            alpha_map = inventory["Glossiness"]
            invert = False
        elif "Specular" in inventory:
            try:
                rough_img = ImgUtils.create_roughness_from_spec(inventory["Specular"])
                alpha_map = os.path.join(output_dir, f"{base_name}_Roughness.{ext}")
                rough_img.save(alpha_map)
                invert = True
                callback(
                    '<br><hl style="color:rgb(100, 160, 100);">Created roughness from specular</hl>'
                )
            except Exception as e:
                callback(
                    f'<br><hl style="color:rgb(255, 100, 100);">Error creating roughness: {str(e)}</hl>'
                )
                return metallic
        else:
            return metallic

        # Pack metallic+smoothness
        try:
            combined = ImgUtils.pack_smoothness_into_metallic(
                metallic, alpha_map, invert_alpha=invert
            )
            callback(
                '<br><hl style="color:rgb(100, 160, 100);">Packed smoothness into metallic</hl>'
            )
            return combined
        except Exception as e:
            callback(
                f'<br><hl style="color:rgb(255, 100, 100);">Error packing metallic/smoothness: {str(e)}</hl>'
            )
            return metallic

    @staticmethod
    def _prepare_metallic(
        inventory: Dict[str, str],
        output_dir: str,
        base_name: str,
        ext: str,
        callback: Callable,
    ) -> Optional[str]:
        """Prepare separate metallic map."""
        if "Metallic" in inventory:
            return inventory["Metallic"]

        # Convert from specular if needed
        if "Specular" in inventory:
            try:
                metallic_img = ImgUtils.create_metallic_from_spec(inventory["Specular"])
                metallic = os.path.join(output_dir, f"{base_name}_Metallic.{ext}")
                metallic_img.save(metallic)
                callback(
                    '<br><hl style="color:rgb(100, 160, 100);">Created metallic from specular</hl>'
                )
                return metallic
            except Exception as e:
                callback(
                    f'<br><hl style="color:rgb(255, 100, 100);">Error creating metallic: {str(e)}</hl>'
                )

        return None

    @staticmethod
    def _prepare_roughness(
        inventory: Dict[str, str],
        output_dir: str,
        base_name: str,
        ext: str,
        callback: Callable,
    ) -> Optional[str]:
        """Prepare roughness map (with automatic conversion from smoothness/glossiness)."""
        if "Roughness" in inventory:
            return inventory["Roughness"]

        # Convert from smoothness/glossiness
        if "Smoothness" in inventory:
            try:
                roughness = ImgUtils.convert_smoothness_to_roughness(
                    inventory["Smoothness"], output_dir
                )
                callback(
                    '<br><hl style="color:rgb(100, 160, 100);">Converted smoothness to roughness</hl>'
                )
                return roughness
            except Exception as e:
                callback(
                    f'<br><hl style="color:rgb(255, 100, 100);">Error converting smoothness: {str(e)}</hl>'
                )

        if "Glossiness" in inventory:
            try:
                roughness = ImgUtils.convert_smoothness_to_roughness(
                    inventory["Glossiness"], output_dir
                )
                callback(
                    '<br><hl style="color:rgb(100, 160, 100);">Converted glossiness to roughness</hl>'
                )
                return roughness
            except Exception as e:
                callback(
                    f'<br><hl style="color:rgb(255, 100, 100);">Error converting glossiness: {str(e)}</hl>'
                )

        # Convert from specular if needed
        if "Specular" in inventory:
            try:
                rough_img = ImgUtils.create_roughness_from_spec(inventory["Specular"])
                roughness = os.path.join(output_dir, f"{base_name}_Roughness.{ext}")
                rough_img.save(roughness)
                callback(
                    '<br><hl style="color:rgb(100, 160, 100);">Created roughness from specular</hl>'
                )
                return roughness
            except Exception as e:
                callback(
                    f'<br><hl style="color:rgb(255, 100, 100);">Error creating roughness: {str(e)}</hl>'
                )

        return None

    @staticmethod
    def _prepare_normal(
        inventory: Dict[str, str],
        target_format: str,
        output_dir: str,
        base_name: str,
        ext: str,
        callback: Callable,
    ) -> Optional[str]:
        """Prepare normal map with format conversion if needed."""
        target_key = f"Normal_{target_format}"

        # Check for exact match
        if target_key in inventory:
            return inventory[target_key]

        # Check for generic normal map
        if "Normal" in inventory:
            return inventory["Normal"]

        # Convert from opposite format
        if target_format == "OpenGL" and "Normal_DirectX" in inventory:
            try:
                converted = ImgUtils.create_gl_from_dx(inventory["Normal_DirectX"])
                callback(
                    '<br><hl style="color:rgb(100, 160, 100);">Converted DirectX normal to OpenGL</hl>'
                )
                return converted
            except Exception as e:
                callback(
                    f'<br><hl style="color:rgb(255, 100, 100);">Error converting normal: {str(e)}</hl>'
                )
                return inventory["Normal_DirectX"]

        elif target_format == "DirectX" and "Normal_OpenGL" in inventory:
            try:
                converted = ImgUtils.create_dx_from_gl(inventory["Normal_OpenGL"])
                callback(
                    '<br><hl style="color:rgb(100, 160, 100);">Converted OpenGL normal to DirectX</hl>'
                )
                return converted
            except Exception as e:
                callback(
                    f'<br><hl style="color:rgb(255, 100, 100);">Error converting normal: {str(e)}</hl>'
                )
                return inventory["Normal_OpenGL"]

        # Generate from bump/height map
        if "Bump" in inventory or "Height" in inventory:
            source = inventory.get("Bump") or inventory["Height"]
            try:
                normal = ImgUtils.convert_bump_to_normal(
                    source,
                    output_format=target_format.lower(),
                )
                callback(
                    '<br><hl style="color:rgb(100, 160, 100);">Generated normal map from height/bump</hl>'
                )
                return normal
            except Exception as e:
                callback(
                    f'<br><hl style="color:rgb(255, 100, 100);">Error generating normal: {str(e)}</hl>'
                )

        return None

    @staticmethod
    def _prepare_mask_map(
        inventory: Dict[str, str],
        output_dir: str,
        base_name: str,
        ext: str,
        callback: Callable,
    ) -> Optional[str]:
        """Prepare Unity HDRP Mask Map (MSAO)."""
        # Check for existing mask map
        if "MSAO" in inventory:
            return inventory["MSAO"]

        # Need at least metallic
        metallic = inventory.get("Metallic")
        if not metallic:
            if "Specular" in inventory:
                try:
                    metallic_img = ImgUtils.create_metallic_from_spec(
                        inventory["Specular"]
                    )
                    metallic = os.path.join(output_dir, f"{base_name}_Metallic.{ext}")
                    metallic_img.save(metallic)
                except Exception as e:
                    callback(
                        '<br><hl style="color:rgb(200, 200, 100);">Warning: No metallic map for Mask Map</hl>'
                    )
                    return None
            else:
                callback(
                    '<br><hl style="color:rgb(200, 200, 100);">Warning: No metallic map for Mask Map</hl>'
                )
                return None

        # Get AO (or use white)
        ao = inventory.get("Ambient_Occlusion") or metallic

        # Get smoothness (or convert from roughness)
        if "Smoothness" in inventory:
            alpha = inventory["Smoothness"]
            invert = False
        elif "Roughness" in inventory:
            alpha = inventory["Roughness"]
            invert = True
        elif "Glossiness" in inventory:
            alpha = inventory["Glossiness"]
            invert = False
        else:
            alpha = metallic
            invert = False

        try:
            mask_map = ImgUtils.pack_msao_texture(
                metallic_map_path=metallic,
                ao_map_path=ao,
                alpha_map_path=alpha,
                output_dir=output_dir,
                suffix="_MaskMap",
                invert_alpha=invert,
            )
            callback(
                '<br><hl style="color:rgb(100, 160, 100);">Created Unity HDRP Mask Map</hl>'
            )
            return mask_map
        except Exception as e:
            callback(
                f'<br><hl style="color:rgb(255, 100, 100);">Error creating Mask Map: {str(e)}</hl>'
            )
            return None

    @staticmethod
    def _prepare_orm_map(
        inventory: Dict[str, str],
        output_dir: str,
        base_name: str,
        ext: str,
        callback: Callable,
    ) -> Optional[str]:
        """Prepare Unreal/glTF ORM Map (Occlusion+Roughness+Metallic).

        ORM format:
        - R: Ambient Occlusion
        - G: Roughness
        - B: Metallic

        This is the standard packed format for Unreal Engine and glTF 2.0.
        """
        # Check for existing ORM map
        if "ORM" in inventory:
            return inventory["ORM"]

        # Get AO (or use white)
        ao = inventory.get("Ambient_Occlusion")
        if not ao:
            callback(
                '<br><hl style="color:rgb(200, 200, 100);">Warning: No AO map, using white for ORM red channel</hl>'
            )

        # Get roughness (or convert from smoothness)
        roughness = inventory.get("Roughness")
        if not roughness:
            if "Smoothness" in inventory:
                try:
                    roughness = ImgUtils.convert_smoothness_to_roughness(
                        inventory["Smoothness"], output_dir
                    )
                except Exception as e:
                    callback(
                        f'<br><hl style="color:rgb(255, 100, 100);">Error converting smoothness: {str(e)}</hl>'
                    )
                    return None
            elif "Glossiness" in inventory:
                try:
                    roughness = ImgUtils.convert_smoothness_to_roughness(
                        inventory["Glossiness"], output_dir
                    )
                except Exception as e:
                    callback(
                        f'<br><hl style="color:rgb(255, 100, 100);">Error converting glossiness: {str(e)}</hl>'
                    )
                    return None
            elif "Specular" in inventory:
                try:
                    rough_img = ImgUtils.create_roughness_from_spec(
                        inventory["Specular"]
                    )
                    roughness = os.path.join(output_dir, f"{base_name}_Roughness.{ext}")
                    rough_img.save(roughness)
                except Exception as e:
                    callback(
                        f'<br><hl style="color:rgb(255, 100, 100);">Error creating roughness: {str(e)}</hl>'
                    )
                    return None

        if not roughness:
            callback(
                '<br><hl style="color:rgb(200, 200, 100);">Warning: No roughness map for ORM green channel</hl>'
            )
            return None

        # Get metallic (or convert from specular)
        metallic = inventory.get("Metallic")
        if not metallic and "Specular" in inventory:
            try:
                metallic_img = ImgUtils.create_metallic_from_spec(inventory["Specular"])
                metallic = os.path.join(output_dir, f"{base_name}_Metallic.{ext}")
                metallic_img.save(metallic)
            except Exception as e:
                callback(
                    f'<br><hl style="color:rgb(255, 100, 100);">Error creating metallic: {str(e)}</hl>'
                )
                return None

        if not metallic:
            callback(
                '<br><hl style="color:rgb(200, 200, 100);">Warning: No metallic map for ORM blue channel</hl>'
            )
            return None

        try:
            # Pack ORM using generic channel packing
            orm_map = ImgUtils.pack_channels(
                channel_files={
                    "R": ao if ao else None,
                    "G": roughness,
                    "B": metallic,
                },
                output_path=os.path.join(output_dir, f"{base_name}_ORM.{ext}"),
                fill_values={"R": 255} if not ao else None,  # White AO if missing
            )
            callback(
                '<br><hl style="color:rgb(100, 160, 100);">Created Unreal/glTF ORM map</hl>'
            )
            return orm_map
        except Exception as e:
            callback(
                f'<br><hl style="color:rgb(255, 100, 100);">Error creating ORM map: {str(e)}</hl>'
            )
            return None

    @staticmethod
    def _convert_specgloss_workflow(
        inventory: Dict[str, str],
        output_dir: str,
        base_name: str,
        ext: str,
        callback: Callable,
    ) -> Dict[str, str]:
        """Convert Spec/Gloss workflow to PBR Metal/Rough workflow.

        Detects Specular + Glossiness/Smoothness and converts to:
        - Base Color (from diffuse + spec + metallic estimation)
        - Metallic
        - Roughness
        """
        has_specular = "Specular" in inventory
        has_glossiness = "Glossiness" in inventory or "Smoothness" in inventory
        has_diffuse = "Diffuse" in inventory

        if not (has_specular and (has_glossiness or has_diffuse)):
            # Not a spec/gloss workflow, return unchanged
            return inventory

        try:
            # Use the comprehensive spec/gloss conversion
            base_color_path, metallic_path, roughness_path = (
                ImgUtils.convert_spec_gloss_to_pbr(
                    specular_map=inventory["Specular"],
                    glossiness_map=inventory.get("Glossiness")
                    or inventory.get("Smoothness"),
                    diffuse_map=inventory.get("Diffuse"),
                    output_dir=output_dir,
                    write_files=True,
                )
            )

            # Update inventory with converted maps
            new_inventory = inventory.copy()
            new_inventory["Base_Color"] = base_color_path
            new_inventory["Metallic"] = metallic_path
            new_inventory["Roughness"] = roughness_path

            # Remove spec/gloss maps from inventory (they're now converted)
            new_inventory.pop("Specular", None)
            new_inventory.pop("Glossiness", None)
            new_inventory.pop("Smoothness", None)
            new_inventory.pop("Diffuse", None)  # Now using Base_Color

            callback(
                '<br><hl style="color:rgb(100, 160, 100);">Converted Spec/Gloss workflow to PBR Metal/Rough</hl>'
            )
            return new_inventory

        except Exception as e:
            callback(
                f'<br><hl style="color:rgb(255, 100, 100);">Error converting Spec/Gloss: {str(e)}</hl>'
            )
            return inventory
