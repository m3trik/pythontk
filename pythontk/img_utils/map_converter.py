# !/usr/bin/python
# coding=utf-8
from typing import List, Union, Tuple, Dict, Any

# from this package:
from pythontk.img_utils._img_utils import ImgUtils
from pythontk.file_utils._file_utils import FileUtils


class MapConverterSlots(ImgUtils):
    texture_file_types = [
        "*.png",
        "*.jpg",
        "*.bmp",
        "*.tga",
        "*.tiff",
        "*.gif",
        "*.exr",
    ]

    def __init__(self, **kwargs):
        super().__init__()
        self.sb = kwargs.get("switchboard")
        self.ui = self.sb.loaded_ui.map_converter

        self._source_dir = kwargs.get("source_dir", "")

    @property
    def source_dir(self):
        """Get the starting directory for file dialogs."""
        return self._source_dir

    @source_dir.setter
    def source_dir(self, value):
        """Set the starting directory for file dialogs."""
        self._source_dir = value

    def b000(self):
        """Convert DirectX to OpenGL"""
        dx_map_path = self.sb.file_dialog(
            file_types=self.texture_file_types,
            title="Select a DirectX normal map to convert:",
            start_dir=self.source_dir,
            allow_multiple=False,
        )
        if not dx_map_path:
            return

        print(f"Converting: {dx_map_path} ..")
        gl_map_path = self.create_gl_from_dx(dx_map_path)
        print(f"// Result: {gl_map_path}")
        self.source_dir = FileUtils.format_path(gl_map_path, "path")

    def b001(self):
        """Convert OpenGL to DirectX"""
        gl_map_path = self.sb.file_dialog(
            file_types=self.texture_file_types,
            title="Select an OpenGL normal map to convert:",
            start_dir=self.source_dir,
            allow_multiple=False,
        )
        if not gl_map_path:
            return

        print(f"Converting: {gl_map_path} ..")
        dx_map_path = self.create_dx_from_gl(gl_map_path)
        print(f"// Result: {dx_map_path}")
        self.source_dir = FileUtils.format_path(dx_map_path, "path")

    def b002(self):
        """Pack Transparency into Albedo, converting Base Color if necessary."""
        paths = self.sb.file_dialog(
            file_types=self.texture_file_types,
            title="Select an Albedo (or Base Color) map and a Transparency map:",
            start_dir=self.source_dir,
            allow_multiple=True,
        )
        if not paths:
            return
        if len(paths) < 2:
            raise ValueError(
                "Please select both an Albedo (or Base Color) map and a Transparency map."
            )

        albedo_map_path = None
        alpha_map_path = None
        base_color_path = None

        # Determine file types
        for path in paths:
            map_type = self.resolve_map_type(path)
            if map_type == "Base_Color":
                base_color_path = path
            elif map_type == "Albedo_Transparency":
                albedo_map_path = path
            elif map_type == "Opacity":
                alpha_map_path = (
                    path  # Ensure we correctly identify the transparency map
                )
        if not alpha_map_path:
            raise FileNotFoundError(
                "Transparency (Opacity) map not found in the selected files."
            )
        # If only Base Color is given, convert it to Albedo
        if not albedo_map_path and base_color_path:
            print(
                f"Converting {base_color_path} to Albedo before packing transparency..."
            )
            albedo_image = self.ensure_image(base_color_path)
            metalness_map = self.resolve_texture_filename(base_color_path, "Metallic")

            # Convert Base Color to Albedo using our helper method
            albedo_image = self.convert_base_color_to_albedo(
                albedo_image, self.ensure_image(metalness_map)
            )

            # Save the new Albedo map
            albedo_map_path = self.resolve_texture_filename(
                base_color_path, "Albedo_Transparency"
            )
            albedo_image.save(albedo_map_path)
            print(f"Converted Base Color to Albedo: {albedo_map_path}")

        if not albedo_map_path:
            raise FileNotFoundError(
                "Neither Albedo nor Base Color map found in the selected files."
            )
        print(f"Packing transparency from {alpha_map_path} into {albedo_map_path} ..")
        # Pack Transparency into Albedo
        albedo_transparency_map_path = self.pack_transparency_into_albedo(
            albedo_map_path, alpha_map_path
        )
        print(f"// Result: {albedo_transparency_map_path}")
        self.source_dir = FileUtils.format_path(albedo_transparency_map_path, "path")

    def b003(self):
        """Pack Smoothness or Roughness into Metallic"""
        paths = self.sb.file_dialog(
            file_types=self.texture_file_types,
            title="Select a metallic map and a smoothness or roughness map:",
            start_dir=self.source_dir,
            allow_multiple=True,
        )
        if not paths:
            return
        elif len(paths) < 2:
            raise ValueError(
                "Please select both a metallic map and a smoothness or roughness map."
            )

        metallic_map_path = None
        alpha_map_path = None
        invert_alpha = False

        for path in paths:
            map_type = self.resolve_map_type(path)
            if map_type == "Metallic":
                metallic_map_path = path
            elif map_type == "Smoothness":
                alpha_map_path = path
                invert_alpha = False
            elif map_type == "Roughness" and alpha_map_path is None:
                alpha_map_path = path
                invert_alpha = True

        if not metallic_map_path:
            raise FileNotFoundError("Metallic map not found in the selected files.")
        if not alpha_map_path:
            raise FileNotFoundError(
                "Smoothness or Roughness map not found in the selected files."
            )

        print(
            f"Packing {'smoothness' if not invert_alpha else 'roughness'} from {alpha_map_path} into {metallic_map_path} .."
        )
        metallic_smoothness_map_path = self.pack_smoothness_into_metallic(
            metallic_map_path, alpha_map_path, invert_alpha=invert_alpha
        )
        print(f"// Result: {metallic_smoothness_map_path}")
        self.source_dir = FileUtils.format_path(metallic_smoothness_map_path, "path")

    def b004(self):
        """Batch converts Spec/Gloss maps to PBR Metal/Rough.

        User selects multiple texture sets. The function groups them per base name
        and converts them accordingly.

        Maps are saved as Metallic/Roughness maps in the same directory.
        """
        spec_map_paths = self.sb.file_dialog(
            file_types=self.texture_file_types,
            title="Select Specular, Gloss (optional), and Diffuse maps to convert:",
            start_dir=self.source_dir,
            allow_multiple=True,
        )
        if not spec_map_paths:
            return

        # **Group maps by set using base names**
        texture_sets = self.group_textures_by_set(spec_map_paths)

        for base_name, texture_paths in texture_sets.items():
            sorted_maps = self.sort_images_by_type(texture_paths)

            # Required maps for conversion
            specular_map = sorted_maps.get("Specular", [None])[0]
            glossiness_map = sorted_maps.get("Glossiness", [None])[0]
            diffuse_map = sorted_maps.get("Diffuse", [None])[0]

            if not specular_map:
                print(f"Skipping {base_name}: No specular map found.")
                continue

            # Extract gloss if missing
            if not glossiness_map:
                print(
                    f"No gloss map found for {base_name}. Checking specular for packed gloss..."
                )
                glossiness_map = self.extract_gloss_from_spec(specular_map)
                if glossiness_map:
                    print(f"Extracted gloss map for {base_name}.")
                else:
                    print(
                        f"No gloss found; estimating roughness from spec for {base_name}."
                    )
                    glossiness_map = self.create_roughness_from_spec(specular_map)

            # Convert to PBR Metal/Rough
            self.convert_spec_gloss_to_pbr(
                specular_map,
                glossiness_map,
                diffuse_map,
                output_type="PNG",
                image_size=4096,
                optimize_bit_depth=True,
            )
            print(f"Spec/Gloss to PBR conversion complete for {base_name}.")
            self.source_dir = FileUtils.format_path(specular_map, "path")

    def b005(self):
        """Extract Gloss map from Specular map."""
        spec_map_path = self.sb.file_dialog(
            file_types=self.texture_file_types,
            title="Select a specular map to extract gloss from:",
            start_dir=self.source_dir,
            allow_multiple=False,
        )
        if not spec_map_path:
            return

        print(f"Extracting gloss from {spec_map_path} ..")
        gloss_image = self.extract_gloss_from_spec(spec_map_path)

        # Resolve the correct gloss texture filename
        gloss_map_path = self.resolve_texture_filename(
            spec_map_path, "Gloss", ext="PNG"
        )
        # Save gloss map
        gloss_image.save(gloss_map_path, format="PNG")

        print(f"// Result: {gloss_map_path}")
        self.source_dir = FileUtils.format_path(gloss_map_path, "path")

    def b006(self):
        """Optimize a texture map(s)"""
        texture_paths = self.sb.file_dialog(
            file_types=self.texture_file_types,
            title="Select texture map(s) to optimize:",
            start_dir=self.source_dir,
            allow_multiple=True,
        )
        if not texture_paths:
            return

        for texture_path in texture_paths:
            print(f"Optimizing: {texture_path} ..")
            optimized_map_path = self.optimize_texture(
                texture_path,
                output_type="PNG",
                max_size=4096,
                old_files_folder="old",
                optimize_bit_depth=True,
            )
            print(f"// Result: {optimized_map_path}")
        self.source_dir = FileUtils.format_path(texture_paths[0], "path")


class MapConverterUi:
    def __new__(self):
        """Get the Map Converter UI."""
        from uitk import Switchboard

        sb = Switchboard(ui_source="map_converter.ui", slot_source=MapConverterSlots)
        ui = sb.loaded_ui.map_converter

        ui.set_attributes(WA_TranslucentBackground=True)
        ui.set_flags(FramelessWindowHint=True, WindowStaysOnTopHint=True)
        ui.set_style(theme="dark", style_class="translucentBgWithBorder")
        ui.header.configure_buttons(
            menu_button=True, minimize_button=True, hide_button=True
        )
        return ui


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    MapConverterUi().show(pos="screen", app_exec=True)

# -----------------------------------------------------------------------------
# Notes
# -----------------------------------------------------------------------------
