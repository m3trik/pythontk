# !/usr/bin/python
# coding=utf-8
from typing import List, Union, Tuple, Dict, Any

# from this package:
from pythontk.img_utils._img_utils import ImgUtils
from pythontk.img_utils.texture_map_factory import TextureMapFactory
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

    def __init__(self, switchboard, **kwargs):
        super().__init__()

        self.sb = switchboard
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
        """Batch converts Spec/Gloss maps to PBR Metal/Rough using TextureMapFactory.

        User selects multiple texture sets. The function groups them per base name
        and converts them accordingly using the DRY TextureMapFactory.

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

        # Use TextureMapFactory for DRY conversion
        workflow_config = {
            "albedo_transparency": False,
            "metallic_smoothness": create_metallic_smoothness,
            "mask_map": False,
            "normal_type": "OpenGL",
            "output_extension": "png",
            "convert_specgloss_to_pbr": True,
        }

        print(f"Processing {len(spec_map_paths)} files...")

        try:
            results = TextureMapFactory.prepare_maps(
                spec_map_paths,
                workflow_config,
                callback=print,
            )

            if isinstance(results, dict):
                print(f"Processed {len(results)} texture sets.")
                for base_name, maps in results.items():
                    print(f"Set: {base_name}")
                    for m in maps:
                        print(f"  - {m}")
            else:
                print("Processed single set.")
                for m in results:
                    print(f"  - {m}")

        except Exception as e:
            print(f"Error during batch processing: {e}")
            import traceback

            traceback.print_exc()

        try:
            self.source_dir = FileUtils.format_path(spec_map_paths[0], "path")
        except Exception:
            pass

    def tb003_init(self, widget):
        """Initialize a 'Bump to Normal' toolbutton with options."""
        widget.menu.add(
            "QComboBox",
            setObjectName="tb003_cmb_format",
            setToolTip="OpenGL: Y+ up, DirectX: Y+ down",
        )
        # Display-friendly items with data values
        cmb = widget.menu.tb003_cmb_format
        cmb.clear()
        cmb.addItem("Format: OpenGL", "opengl")
        cmb.addItem("Format: DirectX", "directx")

        widget.menu.add(
            "QDoubleSpinBox",
            setObjectName="tb003_dsb_intensity",
            setMinimum=0.1,
            setMaximum=5.0,
            setSingleStep=0.1,
            setValue=1.0,
            setDecimals=2,
            setPrefix="Intensity: ",
            setToolTip="Controls how deep the height values are interpreted",
        )

    def tb003(self, widget):
        """Bump/Height to Normal converter (single entry point with options)."""
        bump_map_paths = self.sb.file_dialog(
            file_types=self.texture_file_types,
            title="Select bump/height maps to convert:",
            start_dir=self.source_dir,
            allow_multiple=True,
        )
        if not bump_map_paths:
            return

        # Options
        try:
            output_format = widget.menu.tb003_cmb_format.currentData() or "opengl"
        except Exception:
            fmt_text = widget.menu.tb003_cmb_format.currentText().lower()
            output_format = "directx" if "directx" in fmt_text else "opengl"
        intensity = widget.menu.tb003_dsb_intensity.value()

        for bump_path in bump_map_paths:
            print(f"Converting bump to normal ({output_format.upper()}): {bump_path}")

            try:
                normal_path = self.convert_bump_to_normal(
                    bump_path,
                    output_format=output_format,
                    intensity=intensity,
                    smooth_filter=True,
                    filter_radius=0.5,
                )
                print(f"// Result: {normal_path}")

            except Exception as e:
                print(f"// Error converting {bump_path}: {e}")

        try:
            self.source_dir = FileUtils.format_path(bump_map_paths[0], "path")
        except Exception:
            pass

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

    def b004(self):
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

    def b005(self):
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

    def b006(self):
        """Unpack Metallic and Smoothness maps from MetallicSmoothness textures."""
        print("Unpacking Metallic and Smoothness maps ..")
        metallic_smoothness_paths = self.sb.file_dialog(
            file_types=self.texture_file_types,
            title="Select MetallicSmoothness maps to unpack:",
            start_dir=self.source_dir,
            allow_multiple=True,
        )
        if not metallic_smoothness_paths:
            return

        for metallic_smoothness_path in metallic_smoothness_paths:
            print(f"Unpacking: {metallic_smoothness_path} ..")

            try:
                metallic_path, smoothness_path = self.unpack_metallic_smoothness(
                    metallic_smoothness_path
                )
                print(f"// Metallic map: {metallic_path}")
                print(f"// Smoothness map: {smoothness_path}")

            except Exception as e:
                print(f"// Error unpacking {metallic_smoothness_path}: {e}")

        try:
            self.source_dir = FileUtils.format_path(
                metallic_smoothness_paths[0], "path"
            )
        except Exception:
            pass

    def b007(self):
        """Unpack Specular and Gloss maps from SpecularGloss textures."""
        specular_gloss_paths = self.sb.file_dialog(
            file_types=self.texture_file_types,
            title="Select SpecularGloss maps to unpack:",
            start_dir=self.source_dir,
            allow_multiple=True,
        )
        if not specular_gloss_paths:
            return

        for specular_gloss_path in specular_gloss_paths:
            print(f"Unpacking: {specular_gloss_path} ..")

            try:
                specular_path, gloss_path = self.unpack_specular_gloss(
                    specular_gloss_path
                )
                print(f"// Specular map: {specular_path}")
                print(f"// Gloss map: {gloss_path}")

            except Exception as e:
                print(f"// Error unpacking {specular_gloss_path}: {e}")

        try:
            self.source_dir = FileUtils.format_path(specular_gloss_paths[0], "path")
        except Exception:
            pass

    def b008(self):
        """Batch pack Metallic (R), AO (G), and Smoothness (A) across texture sets."""
        paths = self.sb.file_dialog(
            file_types=self.texture_file_types,
            title="Select one or more sets of Metallic, Ambient Occlusion, and Smoothness maps:",
            start_dir=self.source_dir,
            allow_multiple=True,
        )
        if not paths:
            return

        texture_sets = self.group_textures_by_set(paths)

        for base_name, files in texture_sets.items():
            sorted_maps = self.sort_images_by_type(files)

            metallic_map_path = sorted_maps.get("Metallic", [None])[0]
            ao_map_path = sorted_maps.get("Ambient_Occlusion", [None])[0]
            smoothness_map_path = sorted_maps.get("Smoothness", [None])[0]
            roughness_map_path = sorted_maps.get("Roughness", [None])[0]

            if not metallic_map_path:
                print(f"Skipping {base_name}: No Metallic map found.")
                continue

            if not ao_map_path:
                print(f"Skipping {base_name}: No Ambient Occlusion map found.")
                continue

            # Use smoothness or convert roughness
            alpha_map_path = smoothness_map_path or roughness_map_path
            invert_alpha = roughness_map_path is not None

            if not alpha_map_path:
                print(f"Skipping {base_name}: No Smoothness or Roughness map found.")
                continue

            print(f"Packing MSAO for {base_name}:")
            print(f"  Metallic (R): {metallic_map_path}")
            print(f"  AO (G): {ao_map_path}")
            print(
                f"  {'Roughness' if invert_alpha else 'Smoothness'} (A): {alpha_map_path}"
            )

            packed_path = self.pack_msao_texture(
                metallic_map_path,
                ao_map_path,
                alpha_map_path,
                invert_alpha=invert_alpha,
            )
            print(f"// Result: {packed_path}")

        try:
            self.source_dir = FileUtils.format_path(paths[0], "path")
        except Exception:
            pass

    def b009(self):
        """Unpack Metallic, AO, and Smoothness maps from MSAO textures."""
        msao_paths = self.sb.file_dialog(
            file_types=self.texture_file_types,
            title="Select MSAO (MetallicSmoothnessAO) maps to unpack:",
            start_dir=self.source_dir,
            allow_multiple=True,
        )
        if not msao_paths:
            return

        for msao_path in msao_paths:
            print(f"Unpacking MSAO: {msao_path} ..")

            try:
                metallic_path, ao_path, smoothness_path = self.unpack_msao_texture(
                    msao_path
                )
                print(f"// Metallic map: {metallic_path}")
                print(f"// AO map: {ao_path}")
                print(f"// Smoothness map: {smoothness_path}")

            except Exception as e:
                print(f"// Error unpacking {msao_path}: {e}")

        try:
            self.source_dir = FileUtils.format_path(msao_paths[0], "path")
        except Exception:
            pass

    def b010(self):
        """Convert Smoothness maps to Roughness maps."""
        smoothness_paths = self.sb.file_dialog(
            file_types=self.texture_file_types,
            title="Select Smoothness maps to convert to Roughness:",
            start_dir=self.source_dir,
            allow_multiple=True,
        )
        if not smoothness_paths:
            return

        for smoothness_path in smoothness_paths:
            print(f"Converting Smoothness to Roughness: {smoothness_path} ..")

            try:
                roughness_path = self.convert_smoothness_to_roughness(smoothness_path)
                print(f"// Result: {roughness_path}")

            except Exception as e:
                print(f"// Error converting {smoothness_path}: {e}")

        try:
            self.source_dir = FileUtils.format_path(smoothness_paths[0], "path")
        except Exception:
            pass

    def b011(self):
        """Convert Roughness maps to Smoothness maps."""
        roughness_paths = self.sb.file_dialog(
            file_types=self.texture_file_types,
            title="Select Roughness maps to convert to Smoothness:",
            start_dir=self.source_dir,
            allow_multiple=True,
        )
        if not roughness_paths:
            return

        for roughness_path in roughness_paths:
            print(f"Converting Roughness to Smoothness: {roughness_path} ..")

            try:
                smoothness_path = self.convert_roughness_to_smoothness(roughness_path)
                print(f"// Result: {smoothness_path}")

            except Exception as e:
                print(f"// Error converting {roughness_path}: {e}")

        try:
            self.source_dir = FileUtils.format_path(roughness_paths[0], "path")
        except Exception:
            pass

    def b012(self):
        """Batch prepare textures for PBR workflow using TextureMapFactory.

        Supports multiple workflows:
        - Standard PBR (separate maps)
        - Unity URP (Albedo+Transparency, Metallic+Smoothness)
        - Unity HDRP (MSAO Mask Map)
        - Unreal Engine
        - glTF 2.0
        - Godot
        - Specular/Glossiness
        """
        from uitk import Switchboard

        # Get texture paths
        texture_paths = self.sb.file_dialog(
            file_types=self.texture_file_types,
            title="Select texture maps for PBR workflow preparation:",
            start_dir=self.source_dir,
            allow_multiple=True,
        )
        if not texture_paths:
            return

        # Present workflow options
        workflow_options = [
            "Standard PBR (Separate Maps)",
            "Unity URP (Packed: Albedo+Alpha, Metallic+Smoothness)",
            "Unity HDRP (Mask Map: MSAO)",
            "Unreal Engine (BaseColor+Alpha)",
            "glTF 2.0 (Separate Maps)",
            "Godot (Separate Maps)",
            "Specular/Glossiness Workflow",
        ]

        from PySide2.QtWidgets import QInputDialog

        workflow, ok = QInputDialog.getItem(
            None,
            "Select PBR Workflow",
            "Choose target workflow:",
            workflow_options,
            0,
            False,
        )
        if not ok:
            return

        # Map workflow to config
        workflow_configs = {
            "Standard PBR (Separate Maps)": {
                "albedo_transparency": False,
                "metallic_smoothness": False,
                "mask_map": False,
                "normal_type": "OpenGL",
                "output_extension": "png",
            },
            "Unity URP (Packed: Albedo+Alpha, Metallic+Smoothness)": {
                "albedo_transparency": True,
                "metallic_smoothness": True,
                "mask_map": False,
                "normal_type": "OpenGL",
                "output_extension": "png",
            },
            "Unity HDRP (Mask Map: MSAO)": {
                "albedo_transparency": False,
                "metallic_smoothness": False,
                "mask_map": True,
                "normal_type": "OpenGL",
                "output_extension": "png",
            },
            "Unreal Engine (BaseColor+Alpha)": {
                "albedo_transparency": True,
                "metallic_smoothness": False,
                "mask_map": False,
                "normal_type": "DirectX",
                "output_extension": "png",
            },
            "glTF 2.0 (Separate Maps)": {
                "albedo_transparency": False,
                "metallic_smoothness": False,
                "mask_map": False,
                "normal_type": "OpenGL",
                "output_extension": "png",
            },
            "Godot (Separate Maps)": {
                "albedo_transparency": False,
                "metallic_smoothness": False,
                "mask_map": False,
                "normal_type": "OpenGL",
                "output_extension": "png",
            },
            "Specular/Glossiness Workflow": {
                "albedo_transparency": False,
                "metallic_smoothness": True,
                "mask_map": False,
                "normal_type": "OpenGL",
                "output_extension": "png",
            },
        }

        config = workflow_configs.get(workflow)
        if not config:
            print(f"Unknown workflow: {workflow}")
            return

        print(f"\n{'='*60}")
        print(f"Preparing textures for {workflow}")
        print(f"{'='*60}\n")

        try:
            results = TextureMapFactory.prepare_maps(
                texture_paths,
                config,
                callback=print,
            )

            if isinstance(results, dict):
                print(f"Processed {len(results)} texture sets.")
                for base_name, maps in results.items():
                    print(f"\n✓ Set: {base_name}")
                    for m in maps:
                        print(f"  - {FileUtils.format_path(m, 'name')}")
            else:
                print("\n✓ Processed single set.")
                for m in results:
                    print(f"  - {FileUtils.format_path(m, 'name')}")

        except Exception as e:
            print(f"Error during batch processing: {e}")

        print(f"{'='*60}")
        print(f"Workflow preparation complete!")
        print(f"{'='*60}\n")

        try:
            self.source_dir = FileUtils.format_path(texture_paths[0], "path")
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
        ui.header.config_buttons("menu", "minimize", "hide")
        return ui


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    MapConverterUi().show(pos="screen", app_exec=True)

# -----------------------------------------------------------------------------
# Notes
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    MapConverterUi().show(pos="screen", app_exec=True)

# -----------------------------------------------------------------------------
# Notes
# -----------------------------------------------------------------------------
