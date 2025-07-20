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
        """Batch pack Transparency into Albedo across texture sets."""
        paths = self.sb.file_dialog(
            file_types=self.texture_file_types,
            title="Select one or more sets of Albedo/Base Color and Transparency maps:",
            start_dir=self.source_dir,
            allow_multiple=True,
        )
        if not paths:
            return

        texture_sets = self.group_textures_by_set(paths)

        for base_name, files in texture_sets.items():
            sorted_maps = self.sort_images_by_type(files)

            albedo_map_path = sorted_maps.get("Albedo_Transparency", [None])[0]
            base_color_path = sorted_maps.get("Base_Color", [None])[0]
            opacity_map_path = sorted_maps.get("Opacity", [None])[0]

            if not (albedo_map_path or base_color_path):
                print(f"Skipping {base_name}: No Albedo or Base Color map found.")
                continue

            if not opacity_map_path:
                print(f"Skipping {base_name}: No Transparency (Opacity) map found.")
                continue

            rgb_map_path = albedo_map_path or base_color_path

            print(
                f"Packing Transparency from: {opacity_map_path}\n\tinto: {rgb_map_path} .."
            )

            packed_path = self.pack_transparency_into_albedo(
                rgb_map_path,
                opacity_map_path,
                invert_alpha=False,
            )
            print(f"// Result: {packed_path}")

        try:
            self.source_dir = FileUtils.format_path(paths[0], "path")
        except Exception:
            pass

    def b003(self):
        """Batch pack Smoothness or Roughness into Metallic across texture sets."""
        paths = self.sb.file_dialog(
            file_types=self.texture_file_types,
            title="Select one or more sets of metallic and smoothness/roughness maps:",
            start_dir=self.source_dir,
            allow_multiple=True,
        )
        if not paths:
            return

        texture_sets = self.group_textures_by_set(paths)

        for base_name, files in texture_sets.items():
            sorted_maps = self.sort_images_by_type(files)

            metallic_map_path = sorted_maps.get("Metallic", [None])[0]
            smooth_map_path = sorted_maps.get("Smoothness", [None])[0]
            rough_map_path = sorted_maps.get("Roughness", [None])[0]

            if not metallic_map_path:
                print(f"Skipping {base_name}: No Metallic map found.")
                continue

            alpha_map_path = smooth_map_path or rough_map_path
            invert_alpha = rough_map_path is not None

            if not alpha_map_path:
                print(f"Skipping {base_name}: No Smoothness or Roughness map found.")
                continue

            print(
                f"Packing {'Roughness' if invert_alpha else 'Smoothness'} from: {alpha_map_path}\n\tinto: {metallic_map_path} .."
            )

            packed_path = self.pack_smoothness_into_metallic(
                metallic_map_path,
                alpha_map_path,
                invert_alpha=invert_alpha,
            )
            print(f"// Result: {packed_path}")

        try:
            self.source_dir = FileUtils.format_path(paths[0], "path")
        except Exception:
            pass

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

    def tb000_init(self, widget):
        """ """
        widget.menu.add(
            "QComboBox",
            setObjectName="cmb001",
            setToolTip="Set the output file type.",
        )

        widget.menu.cmb001.addItems(["PNG", "TGA", "BMP", "JPG", "TIF", "EXR", "HDR"])
        widget.menu.add(
            "QComboBox",
            setObjectName="cmb000",
            setToolTip="Set the maximum texture size.",
        )
        widget.menu.cmb000.addItems(["256", "512", "1024", "2048", "4096", "8192"])

    def tb000(self, widget):
        """Optimize a texture map(s)"""
        texture_paths = self.sb.file_dialog(
            file_types=self.texture_file_types,
            title="Select texture map(s) to optimize:",
            start_dir=self.source_dir,
            allow_multiple=True,
        )
        if not texture_paths:
            return

        file_type = widget.menu.cmb001.currentText()
        max_size = int(widget.menu.cmb000.currentText())

        for texture_path in texture_paths:
            print(f"Optimizing: {texture_path} ..")
            optimized_map_path = self.optimize_texture(
                texture_path,
                output_type=file_type,
                max_size=max_size,
                old_files_folder="old",
                optimize_bit_depth=True,
            )
            print(f"// Result: {optimized_map_path}")
        self.source_dir = FileUtils.format_path(texture_paths[0], "path")

    def tb001_init(self, widget):
        """ """
        widget.menu.add(
            "QCheckBox",
            setText="Create MetallicSmoothness map",
            setObjectName="chk000",
            setToolTip="Also create a MetallicSmoothness map.",
        )

    def tb001(self, widget):
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

        create_metallic_smoothness = widget.menu.chk000.isChecked()

        # **Group maps by set using base names**
        texture_sets = self.group_textures_by_set(spec_map_paths)
        print(f"Found {len(texture_sets)} texture sets:")
        for key in texture_sets:
            print(f" - {key}: {texture_sets[key]}")

        for base_name, texture_paths in texture_sets.items():
            print(f"Processing set: {base_name} with {len(texture_paths)} files")

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
            base_color, metallic, roughness = self.convert_spec_gloss_to_pbr(
                specular_map,
                glossiness_map,
                diffuse_map,
                output_type="PNG",
                image_size=4096,
                optimize_bit_depth=True,
                write_files=True,
            )

            if create_metallic_smoothness:
                # Create MetallicSmoothness map
                metallic_smoothness_map_path = self.pack_smoothness_into_metallic(
                    metallic, roughness, invert_alpha=True
                )
                print(f"// Result: {metallic_smoothness_map_path}")

            print(f"Spec/Gloss to PBR conversion complete for {base_name}.")

        try:
            self.source_dir = FileUtils.format_path(specular_map, "path")
        except Exception:
            pass


class MapConverterUi:
    def __new__(self):
        """Get the Map Converter UI."""
        from uitk import Switchboard

        sb = Switchboard(ui_source="map_converter.ui", slot_source=MapConverterSlots)
        ui = sb.loaded_ui.map_converter

        ui.set_attributes(WA_TranslucentBackground=True)
        ui.set_flags(FramelessWindowHint=True)
        ui.style.set(theme="dark", style_class="translucentBgWithBorder")
        ui.header.config_buttons(
            menu_button=True, minimize_button=True, hide_button=True
        )
        return ui


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    MapConverterUi().show(pos="screen", app_exec=True)

# -----------------------------------------------------------------------------
# Notes
# -----------------------------------------------------------------------------
