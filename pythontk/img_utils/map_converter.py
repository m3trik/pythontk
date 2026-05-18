# !/usr/bin/python
# coding=utf-8
"""Map Converter UI — slot file for ``map_converter.ui``.

Bundles texture-map conversion, channel packing, PBR-workflow prep, and bulk
optimization into a single uitk Switchboard panel. The heavy lifting lives in
``MapFactory`` / ``ImgUtils`` — this module is the UI wiring only.

Two public classes:
    MapConverterSlots
        Switchboard slot class. Method names map to widget ``objectName`` in
        the .ui file: ``tb*`` = toolbutton (has an options menu populated by
        the matching ``*_init`` hook), ``b*`` = plain button. Host integrations
        can inject a ``texture_provider`` callable to read the DCC selection.
    MapConverterUi
        Standalone launcher. ``MapConverterUi()`` returns a wired-up UI; run
        the module directly to open it outside any host (``python -m
        pythontk.img_utils.map_converter``).
"""
import os
import tempfile
from typing import List, Union, Tuple, Dict, Any

# From this package:
from pythontk.img_utils._img_utils import ImgUtils
from pythontk.img_utils.map_factory import MapFactory
from pythontk.img_utils.map_registry import MapRegistry
from pythontk.file_utils._file_utils import FileUtils


class MapConverterSlots(ImgUtils):
    """Switchboard slots for ``map_converter.ui``.

    Slot methods are bound to widgets by name. The ``Use Selection`` header
    toggle (installed by :meth:`header_init`) routes every tool through
    :attr:`texture_provider` when set; otherwise tools fall back to a file
    dialog. Set :attr:`source_dir` to seed the initial dialog directory.
    """

    def __init__(self, switchboard, **kwargs):
        super().__init__()

        self.sb = switchboard
        self.ui = self.sb.loaded_ui.map_converter

        self._source_dir = kwargs.get("source_dir", "")
        self._texture_provider = kwargs.get("texture_provider", None)

    @property
    def source_dir(self):
        """Get the starting directory for file dialogs."""
        return self._source_dir

    @source_dir.setter
    def source_dir(self, value):
        """Set the starting directory for file dialogs."""
        self._source_dir = value

    @property
    def texture_provider(self):
        """Callable returning a list of texture paths from the host DCC selection.

        Set by the host integration (e.g. tentacle's materials slot) to power
        the global "Use Selection" header toggle. Returns None when running
        standalone — every tool then falls back to the file dialog regardless
        of the toggle's state.
        """
        return self._texture_provider

    @texture_provider.setter
    def texture_provider(self, fn):
        self._texture_provider = fn

    def header_init(self, widget):
        """Add the global Use-Selection toggle to the header menu."""
        widget.menu.add(
            "QCheckBox",
            setText="Use Selection",
            setObjectName="chk_use_selection",
            setChecked=False,
            setToolTip=(
                "When enabled, every tool reads texture paths from the host "
                "DCC's current selection instead of opening a file browser."
            ),
        )

    def _selection_enabled(self):
        """True when the header's Use-Selection toggle is on."""
        try:
            return self.ui.header.menu.chk_use_selection.isChecked()
        except (AttributeError, RuntimeError):
            return False

    def _get_texture_paths(self, *, title, map_type_filter=None, allow_multiple=True):
        """Resolve texture paths via the global Use-Selection toggle.

        Parameters:
            title (str): Title shown if the file dialog is used.
            map_type_filter (Iterable[str], optional): When pulling from
                selection, restrict to these MapRegistry keys (e.g. ``["Normal",
                "Normal_DirectX"]``). Ignored when the file dialog is used.
            allow_multiple (bool): Forwarded to ``file_dialog``.

        Returns:
            List[str]: Existing absolute paths. Empty list when nothing valid.
        """
        use_selected = self._selection_enabled()

        if use_selected and self.texture_provider:
            paths = list(self.texture_provider() or [])
            if map_type_filter:
                wanted = set(map_type_filter)
                kept, dropped = [], []
                for p in paths:
                    key = MapFactory.resolve_map_type(p, key=True)
                    (kept if key in wanted else dropped).append(p)
                if dropped:
                    print(
                        f"// Skipping {len(dropped)} map(s) not in "
                        f"{sorted(wanted)} from selection."
                    )
                paths = kept
            if not paths:
                print("// No matching textures found on the selected objects.")
                return []
        else:
            if use_selected:
                print(
                    "// 'Use Selection' is enabled but no texture provider is "
                    "wired up — falling back to file dialog."
                )
            paths = self.sb.file_dialog(
                file_types=[f"*.{ext}" for ext in self.texture_file_types],
                title=title,
                start_dir=self.source_dir,
                allow_multiple=allow_multiple,
            )
            paths = list(paths or [])

        valid = [p for p in paths if p and os.path.isfile(p)]
        for missing in (p for p in paths if p not in valid):
            print(f"// Skipping (file not found): {missing}")
        return valid

    def tb000_init(self, widget):
        """Populate the Optimize toolbutton's option menu (format, clamp, modifier)."""
        widget.option_box.menu.setTitle("Optimize")
        widget.option_box.menu.add(
            "QComboBox",
            setObjectName="cmb001",
            setToolTip="Set the output file type. 'Original' keeps each texture's existing format.",
        )
        # Falsy sentinels (empty string / 0) — uitk's prefix-mode combobox
        # replaces explicit None data with the label string, so use values
        # that still evaluate falsy in the ``if not file_type`` / ``if not
        # max_size`` checks below.
        widget.option_box.menu.cmb001.add(
            [("Original", "")] + [(ext.upper(), ext) for ext in self.texture_file_types],
            prefix="Format:",
        )

        widget.option_box.menu.add(
            "QComboBox",
            setObjectName="cmb000",
            setToolTip="Maximum dimension (longest side). 'None' disables resizing.",
        )
        widget.option_box.menu.cmb000.add(
            [("None", 0)] + [(s, int(s)) for s in ("256", "512", "1024", "2048", "4096", "8192")],
            prefix="Clamp:",
        )

        widget.option_box.menu.add(
            "QComboBox",
            setObjectName="cmb_secondary_scale",
            setToolTip=(
                "Downscale non-critical maps (roughness, metallic, AO, masks, "
                "height, etc.) by this fraction of the clamp. Resolution-critical "
                "maps (base color, normals, emissive) always use the full clamp."
            ),
        )
        widget.option_box.menu.cmb_secondary_scale.add(
            [("Full", 1.0), ("1/2", 0.5), ("1/4", 0.25), ("1/8", 0.125)],
            prefix="Secondary:",
        )

        widget.option_box.menu.add(
            "QComboBox",
            setObjectName="cmb_mode",
            setToolTip="Apply the modifier as a Suffix (after base name) or Prefix (before base name). Either way it sits before the map-type suffix.",
        )
        widget.option_box.menu.cmb_mode.addItems(["Suffix", "Prefix"])

        widget.option_box.menu.add(
            "QLineEdit",
            setObjectName="txt_modifier",
            setPlaceholderText="e.g. LD",
            setToolTip=(
                "Text inserted into the base name (before the map-type suffix). "
                "Empty = overwrite the original file."
            ),
        )

        widget.option_box.menu.add(
            "QLineEdit",
            setObjectName="txt_old_folder",
            setText="old",
            setPlaceholderText="e.g. old",
            setToolTip=(
                "Subdirectory under the texture's folder to move the original into. "
                "Empty = don't move the original."
            ),
        )

    def tb000(self, widget):
        """Optimize a texture map(s)"""
        texture_paths = self._get_texture_paths(
            title="Select texture map(s) to optimize:"
        )
        if not texture_paths:
            return

        # Falsy sentinels ("", 0) → None so optimize_texture preserves
        # the original format / skips clamping respectively.
        file_type = widget.option_box.menu.cmb001.currentData() or None
        max_size = widget.option_box.menu.cmb000.currentData() or None
        secondary_scale = (
            widget.option_box.menu.cmb_secondary_scale.currentData() or 1.0
        )
        mode = widget.option_box.menu.cmb_mode.currentText().lower()
        modifier = widget.option_box.menu.txt_modifier.text().strip().strip("_")
        old_folder = (
            widget.option_box.menu.txt_old_folder.text().strip().strip("/").strip("\\")
        )

        registry = MapRegistry()

        for texture_path in texture_paths:
            print(f"Optimizing: {texture_path} ..")

            # Apply the secondary scale to non-critical maps so masks/roughness/
            # etc. shrink relative to base color and normals.
            effective_max_size = max_size
            if max_size and secondary_scale != 1.0:
                map_type_key = MapFactory.resolve_map_type(texture_path, key=True)
                if not registry.is_resolution_critical(map_type_key):
                    effective_max_size = max(1, int(max_size * secondary_scale))
                    print(
                        f"// Secondary scale {secondary_scale:g}x → clamp "
                        f"{effective_max_size} ({map_type_key or 'unknown type'})"
                    )

            if not modifier:
                # Overwrite mode: optimize in place. optimize_texture handles
                # the optional move-to-old-folder for us.
                optimized_map_path = self.optimize_texture(
                    texture_path,
                    output_type=file_type,
                    max_size=effective_max_size,
                    old_files_folder=old_folder or None,
                    optimize_bit_depth=True,
                )
            else:
                # Rename mode: place the modifier between base name and
                # map-type suffix and save alongside the original.
                directory = FileUtils.format_path(texture_path, "path")
                base_name = self.get_base_texture_name(texture_path)
                map_type = MapFactory.resolve_map_type(texture_path, key=False) or ""
                out_ext = (
                    (file_type or FileUtils.format_path(texture_path, "ext"))
                    .lower()
                    .lstrip(".")
                )
                new_base = (
                    f"{modifier}_{base_name}"
                    if mode == "prefix"
                    else f"{base_name}_{modifier}"
                )
                out_filename = (
                    f"{new_base}_{map_type}.{out_ext}"
                    if map_type
                    else f"{new_base}.{out_ext}"
                )
                target_path = os.path.join(directory, out_filename)

                # Same-drive temp dir so the final os.replace is a fast rename
                # and overwrites cleanly on re-run. We can't pass
                # old_files_folder here because optimize_texture would archive
                # the original into the temp dir and lose it on cleanup.
                with tempfile.TemporaryDirectory(dir=directory) as temp_dir:
                    temp_result = self.optimize_texture(
                        texture_path,
                        output_dir=temp_dir,
                        output_type=file_type,
                        max_size=effective_max_size,
                        optimize_bit_depth=True,
                    )
                    os.replace(temp_result, target_path)

                if old_folder:
                    FileUtils.move_file(
                        texture_path, os.path.join(directory, old_folder)
                    )

                optimized_map_path = target_path

            print(f"// Result: {optimized_map_path}")
        self.source_dir = FileUtils.format_path(texture_paths[0], "path")

    def tb001_init(self, widget):
        """ """
        widget.option_box.menu.setTitle("Spec Gloss to PBR")
        widget.option_box.menu.add(
            "QCheckBox",
            setText="Create MetallicSmoothness map",
            setObjectName="chk000",
            setToolTip="Also create a MetallicSmoothness map.",
        )

    def tb001(self, widget):
        """Batch converts Spec/Gloss maps to PBR Metal/Rough using MapFactory.

        User selects multiple texture sets. The function groups them per base name
        and converts them accordingly using the DRY MapFactory.

        Maps are saved as Metallic/Roughness maps in the same directory.
        """
        spec_map_paths = self._get_texture_paths(
            title="Select Specular, Gloss (optional), and Diffuse maps to convert:",
        )
        if not spec_map_paths:
            return

        create_metallic_smoothness = widget.option_box.menu.chk000.isChecked()

        # Use MapFactory for DRY conversion
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
            results = MapFactory.prepare_maps(
                spec_map_paths,
                callback=print,
                **workflow_config,
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
        widget.option_box.menu.setTitle("Bump to Normal")
        widget.option_box.menu.add(
            "QComboBox",
            setObjectName="tb003_cmb_format",
            setToolTip="OpenGL: Y+ up, DirectX: Y+ down",
        )
        # Display-friendly items with data values
        cmb = widget.option_box.menu.tb003_cmb_format
        cmb.clear()
        cmb.addItem("Format: OpenGL", "opengl")
        cmb.addItem("Format: DirectX", "directx")

        widget.option_box.menu.add(
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
        bump_map_paths = self._get_texture_paths(
            title="Select bump/height maps to convert:",
            map_type_filter=["Bump", "Height"],
        )
        if not bump_map_paths:
            return

        # Options
        try:
            output_format = (
                widget.option_box.menu.tb003_cmb_format.currentData() or "opengl"
            )
        except Exception:
            fmt_text = widget.option_box.menu.tb003_cmb_format.currentText().lower()
            output_format = "directx" if "directx" in fmt_text else "opengl"
        intensity = widget.option_box.menu.tb003_dsb_intensity.value()

        for bump_path in bump_map_paths:
            print(f"Converting bump to normal ({output_format.upper()}): {bump_path}")

            try:
                normal_path = MapFactory.convert_bump_to_normal(
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
        dx_map_paths = self._get_texture_paths(
            title="Select a DirectX normal map to convert:",
            map_type_filter=["Normal", "Normal_DirectX"],
        )
        if not dx_map_paths:
            return

        for dx_map_path in dx_map_paths:
            print(f"Converting: {dx_map_path} ..")
            gl_map_path = MapFactory.convert_normal_map_format(
                dx_map_path, target_format="opengl"
            )
            print(f"// Result: {gl_map_path}")
        self.source_dir = FileUtils.format_path(dx_map_paths[0], "path")

    def b001(self):
        """Convert OpenGL to DirectX"""
        gl_map_paths = self._get_texture_paths(
            title="Select an OpenGL normal map to convert:",
            map_type_filter=["Normal", "Normal_OpenGL"],
        )
        if not gl_map_paths:
            return

        for gl_map_path in gl_map_paths:
            print(f"Converting: {gl_map_path} ..")
            dx_map_path = MapFactory.convert_normal_map_format(
                gl_map_path, target_format="directx"
            )
            print(f"// Result: {dx_map_path}")
        self.source_dir = FileUtils.format_path(gl_map_paths[0], "path")

    def b004(self):
        """Batch pack Transparency into Albedo across texture sets."""
        paths = self._get_texture_paths(
            title="Select one or more sets of Albedo/Base Color and Transparency maps:",
        )
        if not paths:
            return

        texture_sets = MapFactory.group_textures_by_set(paths)

        for base_name, files in texture_sets.items():
            sorted_maps = MapFactory.sort_images_by_type(files)

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

            packed_path = MapFactory.pack_transparency_into_albedo(
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
        paths = self._get_texture_paths(
            title="Select one or more sets of metallic and smoothness/roughness maps:",
        )
        if not paths:
            return

        texture_sets = MapFactory.group_textures_by_set(paths)

        for base_name, files in texture_sets.items():
            sorted_maps = MapFactory.sort_images_by_type(files)

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

            packed_path = MapFactory.pack_smoothness_into_metallic(
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
        metallic_smoothness_paths = self._get_texture_paths(
            title="Select MetallicSmoothness maps to unpack:",
            map_type_filter=["Metallic_Smoothness"],
        )
        if not metallic_smoothness_paths:
            return

        for metallic_smoothness_path in metallic_smoothness_paths:
            print(f"Unpacking: {metallic_smoothness_path} ..")

            try:
                metallic_path, smoothness_path = MapFactory.unpack_metallic_smoothness(
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
        specular_gloss_paths = self._get_texture_paths(
            title="Select SpecularGloss maps to unpack:",
            map_type_filter=["Specular"],
        )
        if not specular_gloss_paths:
            return

        for specular_gloss_path in specular_gloss_paths:
            print(f"Unpacking: {specular_gloss_path} ..")

            try:
                specular_path, gloss_path = MapFactory.unpack_specular_gloss(
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
        paths = self._get_texture_paths(
            title="Select one or more sets of Metallic, Ambient Occlusion, and Smoothness maps:",
        )
        if not paths:
            return

        texture_sets = MapFactory.group_textures_by_set(paths)

        for base_name, files in texture_sets.items():
            sorted_maps = MapFactory.sort_images_by_type(files)

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

            packed_path = MapFactory.pack_msao_texture(
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
        msao_paths = self._get_texture_paths(
            title="Select MSAO (MetallicSmoothnessAO) maps to unpack:",
            map_type_filter=["MSAO"],
        )
        if not msao_paths:
            return

        for msao_path in msao_paths:
            print(f"Unpacking MSAO: {msao_path} ..")

            try:
                metallic_path, ao_path, smoothness_path = (
                    MapFactory.unpack_msao_texture(msao_path)
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
        smoothness_paths = self._get_texture_paths(
            title="Select Smoothness maps to convert to Roughness:",
            map_type_filter=["Smoothness"],
        )
        if not smoothness_paths:
            return

        for smoothness_path in smoothness_paths:
            print(f"Converting Smoothness to Roughness: {smoothness_path} ..")

            try:
                roughness_path = MapFactory.convert_smoothness_to_roughness(
                    smoothness_path
                )
                print(f"// Result: {roughness_path}")

            except Exception as e:
                print(f"// Error converting {smoothness_path}: {e}")

        try:
            self.source_dir = FileUtils.format_path(smoothness_paths[0], "path")
        except Exception:
            pass

    def b011(self):
        """Convert Roughness maps to Smoothness maps."""
        roughness_paths = self._get_texture_paths(
            title="Select Roughness maps to convert to Smoothness:",
            map_type_filter=["Roughness"],
        )
        if not roughness_paths:
            return

        for roughness_path in roughness_paths:
            print(f"Converting Roughness to Smoothness: {roughness_path} ..")

            try:
                smoothness_path = MapFactory.convert_roughness_to_smoothness(
                    roughness_path
                )
                print(f"// Result: {smoothness_path}")

            except Exception as e:
                print(f"// Error converting {roughness_path}: {e}")

        try:
            self.source_dir = FileUtils.format_path(roughness_paths[0], "path")
        except Exception:
            pass

    def b012(self):
        """Batch prepare textures for PBR workflow using MapFactory.

        Supports multiple workflows:
        - Standard PBR (separate maps)
        - Unity URP (Albedo+Transparency, Metallic+Smoothness)
        - Unity HDRP (MSAO Mask Map)
        - Unreal Engine
        - glTF 2.0
        - Godot
        - Specular/Glossiness
        """
        # Get texture paths
        texture_paths = self._get_texture_paths(
            title="Select texture maps for PBR workflow preparation:",
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

        from qtpy.QtWidgets import QInputDialog

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
            results = MapFactory.prepare_maps(
                texture_paths,
                callback=print,
                **config,
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
    """Standalone launcher. Constructing the class returns a configured UI.

    ``__new__`` is overridden to return the wired Switchboard UI directly,
    so ``MapConverterUi()`` yields the UI (not an instance). Use this when
    running outside a host DCC. Hosts that need to inject a
    ``texture_provider`` should register :class:`MapConverterSlots`
    themselves rather than going through this launcher.
    """

    def __new__(cls):
        from uitk import Switchboard

        sb = Switchboard(ui_source="map_converter.ui", slot_source=MapConverterSlots)
        ui = sb.loaded_ui.map_converter

        ui.set_attributes(WA_TranslucentBackground=True)
        ui.set_flags(FramelessWindowHint=True)
        ui.style.set(theme="dark", style_class="translucentBgWithBorder")
        ui.header.config_buttons("menu", "minimize", "hide")
        return ui


if __name__ == "__main__":
    MapConverterUi().show(pos="screen", app_exec=True)
