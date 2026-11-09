# !/usr/bin/python
# coding=utf-8
"""The Khronos glTF Validator: the reference check of a glTF or GLB file.

Adopted, not built (CODE_STANDARD section 15, "Interchange: open formats,
validated"): KhronosGroup/glTF-Validator's own release binary, pinned
(:attr:`GltfValidator.VERSION`), found on ``PATH`` or installed into the user's
tool folder by ``AppInstaller`` -- the way FBX2glTF is. Weighed: its npm
package (``gltf-validator``) needs Node, which the DCC hosts and the Python CI
legs do not carry; the release binary is one file and no runtime.

A report is the validator's own JSON (``issues.numErrors``,
``issues.numWarnings``, ``issues.messages``: each a ``code``, ``severity`` --
0 error, 1 warning, 2 info, 3 hint -- ``message`` and ``pointer``).
"""

import json
import logging
import os
import platform as _platform
import shutil
import subprocess
from typing import Any, Callable, Dict, List, Mapping, Optional, Union

logger = logging.getLogger(__name__)


class GltfValidator:
    """Run the Khronos glTF Validator on a file or a glTF document.

    >>> report = GltfValidator.validate("model.glb")
    >>> report["issues"]["numErrors"]
    0
    """

    TOOL_NAME = "gltf_validator"
    #: The KhronosGroup/glTF-Validator release this class downloads and drives.
    VERSION = "2.0.0-dev.3.10"
    _RELEASE = (
        "https://github.com/KhronosGroup/glTF-Validator/releases/download/"
        f"{VERSION}/gltf_validator-{VERSION}"
    )
    #: Per platform: the release archive and the executable inside it.
    PLATFORMS: Dict[str, Dict[str, str]] = {
        "windows": {
            "url": f"{_RELEASE}-win64.zip",
            "type": "zip",
            "executable": "gltf_validator.exe",
        },
        "linux": {
            "url": f"{_RELEASE}-linux64.tar.xz",
            "type": "tar.xz",
            "executable": "gltf_validator",
            "arch": "x86_64",
        },
        "darwin": {
            "url": f"{_RELEASE}-macos64.tar.xz",
            "type": "tar.xz",
            "executable": "gltf_validator",
        },
    }
    #: Seconds a validation may take.
    TIMEOUT = 120

    #: The validator's severities, by number.
    SEVERITIES = {0: "error", 1: "warning", 2: "info", 3: "hint"}

    @classmethod
    def resolve(
        cls,
        auto_install: bool = False,
        prompt: Union[bool, Callable[[str], bool]] = True,
    ) -> Optional[str]:
        """The validator's executable: on ``PATH``, else the managed install,
        else -- with *auto_install* -- downloaded (asked through *prompt*, as
        ``AppInstaller.consent`` asks; ``False`` installs unasked). ``None``
        when it is not installed and not installed now."""
        from pythontk.core_utils.app_installer import AppInstaller

        info = cls.PLATFORMS.get(_platform.system().lower())
        if info is None:
            return None
        exe = info["executable"]
        on_path = shutil.which(exe) or shutil.which(cls.TOOL_NAME)
        if on_path:
            return on_path
        managed = AppInstaller.get_path(cls.TOOL_NAME, executable=exe)
        if managed or not auto_install:
            return managed
        answer = AppInstaller.consent(
            prompt,
            f"The Khronos glTF Validator {cls.VERSION} is not installed. "
            "Download to ~/.pythontk/tools/ now?",
        )
        if not answer:
            return None
        try:
            return AppInstaller.ensure(
                cls.TOOL_NAME, platforms=cls.PLATFORMS, executable=exe, version=cls.VERSION
            )
        except (RuntimeError, OSError, LookupError) as error:
            logger.warning("glTF Validator install failed: %s", error)
            return None

    @classmethod
    def validate(cls, path: str, executable: Optional[str] = None) -> Dict[str, Any]:
        """The validator's report on the glTF or GLB at *path*, its external
        resources (an image ``uri``) read beside it.

        Raises:
            FileNotFoundError: The validator is not installed
                (:meth:`resolve`), or *path* is not a file.
            RuntimeError: The validator wrote no report.
        """
        exe = executable or cls.resolve()
        if not exe:
            raise FileNotFoundError(
                "The Khronos glTF Validator is not installed: "
                "GltfValidator.resolve(auto_install=True) installs it."
            )
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        run = subprocess.run(
            [exe, "--stdout", path],
            capture_output=True,
            text=True,
            timeout=cls.TIMEOUT,
        )
        start = run.stdout.find("{")
        if start < 0:
            raise RuntimeError(
                f"glTF Validator wrote no report (exit {run.returncode}): "
                f"{(run.stderr or run.stdout).strip()[:400]}"
            )
        return json.loads(run.stdout[start:])

    @classmethod
    def validate_document(
        cls, document: Mapping[str, Any], folder: str, name: str = "document.gltf"
    ) -> Dict[str, Any]:
        """The report on *document* (a glTF JSON document) written as *name*
        in *folder*, where the images its ``uri`` s name are read."""
        from pythontk.file_utils._file_utils import FileUtils

        path = os.path.join(folder, name)
        FileUtils.write_json(path, dict(document), indent=None)
        return cls.validate(path)

    @classmethod
    def messages(
        cls, report: Mapping[str, Any], worst: int = 1
    ) -> List[str]:
        """The report's messages at *worst* severity or worse (1: errors and
        warnings), one line each: ``severity code pointer: message``."""
        lines = []
        for item in (report.get("issues") or {}).get("messages") or ():
            severity = item.get("severity", 3)
            if severity <= worst:
                lines.append(
                    f"{cls.SEVERITIES.get(severity, severity)} {item.get('code')} "
                    f"{item.get('pointer', '')}: {item.get('message')}"
                )
        return lines
