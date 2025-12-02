# !/usr/bin/python
# coding=utf-8
from typing import List, Dict, Optional, Any
from pythontk.img_utils._img_utils import ImgUtils
from pythontk.file_utils._file_utils import FileUtils


class MapPackerSlots(ImgUtils):
    texture_file_types = [
        "*.png",
        "*.jpg",
        "*.bmp",
        "*.tga",
        "*.tiff",
        "*.gif",
        "*.exr",
    ]
    channels = ["R", "G", "B", "A"]
    grayscale_types = [
        "None",
        "Metallic",
        "Roughness",
        "Ambient_Occlusion",
        "Smoothness",
        "Opacity",
        "Height",
        "Thickness",
        "Glossiness",
        "Displacement",
    ]

    def __init__(self, switchboard, **kwargs):
        super().__init__()

        self.sb = switchboard
        self.ui = self.sb.loaded_ui.map_packer

        self._source_dir = kwargs.get("source_dir", "")

        self._init_ui_comboboxes()
        self._set_channel_label_colors()
        self.ui.b001.setEnabled(False)  # Disable the open output dir button

    def _set_channel_label_colors(self):
        """Set background color for each channel label."""
        channel_colors = {
            "R": "#ef9a9a",  # Pastel Red
            "G": "#a5d6a7",  # Pastel Green
            "B": "#90caf9",  # Pastel Blue
            "A": "#bdbdbd",  # Pastel Gray
        }
        for c in self.channels:
            lbl = getattr(self.ui, f"lbl{c}", None)
            if lbl:
                lbl.setStyleSheet(
                    f"background-color: {channel_colors[c]}; color: white; border-radius: 3px;"
                )

    def _init_ui_comboboxes(self):
        """Initialize channel and format comboboxes and connect format change signal."""
        for c in self.channels:
            cmb = getattr(self.ui, f"cmb{c}")
            cmb.clear()
            cmb.addItems(self.grayscale_types)
            cmb.restore_state = True  # <-- Enable state restore

        self.ui.cmbFormat.clear()
        self.ui.cmbFormat.addItems(["PNG", "TGA", "JPG", "BMP", "TIFF", "EXR"])
        self.ui.cmbFormat.restore_state = True  # <-- Enable state restore
        self.ui.cmbFormat.currentTextChanged.connect(self._on_format_changed)
        self._on_format_changed(self.ui.cmbFormat.currentText())

    def _on_format_changed(self, fmt: str):
        """Disable alpha combobox for formats without alpha support."""
        supports_alpha = fmt.upper() in {"PNG", "TGA", "TIFF", "EXR", "BMP"}
        self.ui.cmbA.setEnabled(supports_alpha)
        if not supports_alpha:
            self.ui.cmbA.setCurrentIndex(self.ui.cmbA.findText("None"))

    @property
    def source_dir(self):
        return self._source_dir

    @source_dir.setter
    def source_dir(self, value):
        self._source_dir = value

    def b000(self):
        """Batch pack up to 4 channels into RGBA maps across texture sets."""
        file_paths = self.sb.file_dialog(
            file_types=self.texture_file_types,
            title="Select textures for batch packing (multiple sets allowed):",
            start_dir=self.source_dir,
            allow_multiple=True,
        )
        if not file_paths:
            print("No files selected.")
            self.ui.b001.setEnabled(False)
            return

        texture_sets = self.group_textures_by_set(file_paths)
        combos = [
            self.ui.cmbR.currentText(),
            self.ui.cmbG.currentText(),
            self.ui.cmbB.currentText(),
            self.ui.cmbA.currentText(),
        ]
        suffix = self.ui.txtSuffix.text().strip() or "_Packed"
        if not suffix.startswith("_"):
            suffix = f"_{suffix}"
        ext = self.ui.cmbFormat.currentText().lower()
        fmt = self.ui.cmbFormat.currentText().upper()

        success = False
        for base_name, files in texture_sets.items():
            sorted_maps = self.sort_images_by_type(files)
            assigned = {c: None for c in self.channels}
            available_map_types = {self.resolve_map_type(f): f for f in files}
            used_files = set()

            for idx, channel in enumerate(self.channels):
                map_type = combos[idx]
                if map_type == "None":
                    continue
                file = next(
                    (
                        f
                        for f in files
                        if self.resolve_map_type(f) == map_type and f not in used_files
                    ),
                    None,
                )
                if file:
                    assigned[channel] = file
                    used_files.add(file)
                    continue
                # Try conversion if not found
                converted = self.get_converted_map(map_type, available_map_types)
                if converted is not None:
                    assigned[channel] = converted
                    continue
                print(
                    f"// Required map '{map_type}' for channel {channel} in '{base_name}' not found and cannot be converted. Skipping."
                )
                break  # skip this set if not all required maps are present

            if any(assigned[c] for c in self.channels):
                out_mode = "RGBA" if assigned["A"] else "RGB"
                output_dir = FileUtils.format_path(files[0], "path")
                output_path = f"{output_dir}/{base_name}{suffix}.{ext}"
                self.pack_channels(
                    channel_files=assigned,
                    output_path=output_path,
                    out_mode=out_mode,
                    output_format=fmt,
                )
                print(f"// Packed map saved: {output_path}")
                success = True

        if success:
            self._last_output_dir = FileUtils.format_path(file_paths[0], "path")
            self.source_dir = self._last_output_dir
            self.ui.b001.setEnabled(True)
        else:
            self.ui.b001.setEnabled(False)

    def b001(self):
        """Open the last output directory in the system file explorer."""
        import os
        import sys

        output_dir = getattr(self, "_last_output_dir", None)
        if not output_dir or not os.path.isdir(output_dir):
            print("// No output directory available.")
            return
        if sys.platform.startswith("darwin"):
            os.system(f'open "{output_dir}"')
        elif sys.platform.startswith("win"):
            os.startfile(output_dir)
        elif sys.platform.startswith("linux"):
            os.system(f'xdg-open "{output_dir}"')
        else:
            print("// Unsupported OS for opening directories.")


class MapPackerUi:
    def __new__(self):
        from uitk import Switchboard

        sb = Switchboard(ui_source="map_packer.ui", slot_source=MapPackerSlots)
        ui = sb.loaded_ui.map_packer
        ui.set_attributes(WA_TranslucentBackground=True)
        ui.set_flags(FramelessWindowHint=True)
        ui.style.set(theme="dark", style_class="translucentBgWithBorder")
        ui.header.config_buttons("menu", "minimize", "hide")
        return ui


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    MapPackerUi().show(pos="screen", app_exec=True)


# -----------------------------------------------------------------------------
# Notes
# -----------------------------------------------------------------------------
