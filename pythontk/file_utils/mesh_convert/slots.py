# !/usr/bin/python
# coding=utf-8
import os
from typing import Callable, Iterable, List, Optional

from pythontk.file_utils._file_utils import FileUtils
from pythontk.file_utils.mesh_convert._mesh_convert import (
    FBX2GLTF_VERSION,
    MeshConvert,
)


class MeshConvertSlots(MeshConvert):
    """uitk Switchboard slots for the Mesh Converter UI.

    Inherits from :class:`MeshConvert` so all conversion logic
    (``fbx_to_glb``, ``resolve_binary``) is available on ``self``.
    """

    def __init__(self, switchboard, **kwargs):
        super().__init__()

        self.sb = switchboard
        self.ui = self.sb.loaded_ui.mesh_convert

        self._source_dir: str = kwargs.get("source_dir", "")
        self._fbx_provider: Optional[Callable[[], Iterable[str]]] = kwargs.get(
            "fbx_provider", None
        )

    @property
    def source_dir(self) -> str:
        """Starting directory for the FBX file dialog."""
        return self._source_dir

    @source_dir.setter
    def source_dir(self, value: str) -> None:
        self._source_dir = value

    @property
    def fbx_provider(self) -> Optional[Callable[[], Iterable[str]]]:
        """Callable returning FBX paths from the host DCC selection.

        Set by the host integration (e.g. tentacle's scene slot) to power
        the header "Use Selection" toggle. Returns None when running
        standalone, in which case every tool falls back to the file dialog.
        """
        return self._fbx_provider

    @fbx_provider.setter
    def fbx_provider(self, fn: Optional[Callable[[], Iterable[str]]]) -> None:
        self._fbx_provider = fn

    def header_init(self, widget) -> None:
        """Add the global Use-Selection toggle to the header menu."""
        widget.menu.add(
            "QCheckBox",
            setText="Use Selection",
            setObjectName="chk_use_selection",
            setChecked=False,
            setToolTip=(
                "When enabled, every tool reads FBX paths from the host DCC's "
                "current selection (provided by the host integration) instead "
                "of opening a file browser."
            ),
        )

    def _selection_enabled(self) -> bool:
        try:
            return self.ui.header.menu.chk_use_selection.isChecked()
        except (AttributeError, RuntimeError):
            return False

    def _get_fbx_paths(self, *, title: str, allow_multiple: bool = True) -> List[str]:
        """Resolve FBX paths via Use-Selection toggle, falling back to dialog."""
        use_selected = self._selection_enabled()

        if use_selected and self.fbx_provider:
            paths = list(self.fbx_provider() or [])
            if not paths:
                print("// No FBX paths found on the selected objects.")
                return []
        else:
            if use_selected:
                print(
                    "// 'Use Selection' is enabled but no FBX provider is "
                    "wired up — falling back to file dialog."
                )
            paths = self.sb.file_dialog(
                file_types=["*.fbx"],
                title=title,
                start_dir=self.source_dir,
                allow_multiple=allow_multiple,
            )
            paths = list(paths or [])

        valid = [
            p
            for p in paths
            if p and os.path.isfile(p) and p.lower().endswith(".fbx")
        ]
        for missing in (p for p in paths if p not in valid):
            print(f"// Skipping (not an existing .fbx file): {missing}")
        return valid

    def _ensure_binary_or_offer_download(self) -> Optional[str]:
        """Find FBX2glTF; if missing, prompt via Qt and install on confirm."""
        existing = self.resolve_binary(required=False, auto_install=False)
        if existing:
            return existing

        from qtpy.QtWidgets import QMessageBox

        answer = QMessageBox.question(
            None,
            "FBX2glTF not installed",
            (
                f"FBX2glTF v{FBX2GLTF_VERSION} is required to convert FBX to GLB.\n\n"
                "Download it now (a few MB) into ~/.pythontk/tools/?"
            ),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if answer != QMessageBox.Yes:
            print("// Mesh Converter: user declined FBX2glTF installation.")
            return None

        try:
            return self.resolve_binary(
                required=True, auto_install=True, prompt=False
            )
        except (RuntimeError, OSError, FileNotFoundError, LookupError) as exc:
            QMessageBox.critical(
                None,
                "FBX2glTF install failed",
                f"Could not install FBX2glTF:\n\n{exc}",
            )
            return None

    def tb000_init(self, widget) -> None:
        """Set up the FBX -> GLB tool button option box."""
        widget.option_box.menu.setTitle("FBX to GLB")
        widget.option_box.menu.add(
            "QCheckBox",
            setText="Draco compression",
            setObjectName="chk_draco",
            setChecked=False,
            setToolTip="Apply Draco mesh compression — smaller file, slightly slower decode.",
        )
        widget.option_box.menu.add(
            "QCheckBox",
            setText="Overwrite existing",
            setObjectName="chk_overwrite",
            setChecked=True,
            setToolTip="Replace any existing .glb at the destination.",
        )

    def tb000(self, widget) -> None:
        """Convert the selected FBX file(s) to GLB beside their source."""
        fbx_paths = self._get_fbx_paths(title="Select FBX file(s) to convert to GLB:")
        if not fbx_paths:
            return

        if not self._ensure_binary_or_offer_download():
            return

        draco = widget.option_box.menu.chk_draco.isChecked()
        overwrite = widget.option_box.menu.chk_overwrite.isChecked()
        extra_args = ["--draco"] if draco else None

        # Conversion of a half-GB FBX takes ~75 s. Block re-clicks and signal
        # busy state so the window doesn't look like it has hung.
        from qtpy.QtCore import Qt
        from qtpy.QtWidgets import QApplication

        widget.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            ok_count = 0
            for fbx_path in fbx_paths:
                print(f"Converting: {fbx_path} ..")
                QApplication.processEvents()  # drain UI events between files
                try:
                    out_path = self.fbx_to_glb(
                        fbx_path,
                        overwrite=overwrite,
                        auto_install=False,
                        extra_args=extra_args,
                    )
                    print(f"// Result: {out_path}")
                    ok_count += 1
                except FileExistsError as exc:
                    print(f"// Skipped (exists, overwrite=False): {exc}")
                except (RuntimeError, OSError) as exc:
                    print(f"// Error converting {fbx_path}: {exc}")

            print(f"// Mesh Converter: {ok_count}/{len(fbx_paths)} succeeded.")
        finally:
            QApplication.restoreOverrideCursor()
            widget.setEnabled(True)

        try:
            self.source_dir = FileUtils.format_path(fbx_paths[0], "path")
        except Exception:
            pass


class MeshConvertUi:
    def __new__(cls):
        """Get the Mesh Converter UI."""
        from uitk import Switchboard

        sb = Switchboard(ui_source="mesh_convert.ui", slot_source=MeshConvertSlots)
        ui = sb.loaded_ui.mesh_convert

        ui.set_attributes(WA_TranslucentBackground=True)
        ui.set_flags(FramelessWindowHint=True)
        ui.style.set(theme="dark", style_class="translucentBgWithBorder")
        ui.header.config_buttons("menu", "minimize", "hide")
        return ui


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    MeshConvertUi().show(pos="screen", app_exec=True)
