# !/usr/bin/python
# coding=utf-8
"""The Scene Exporter panels' export-button contract, written once.

Both DCC Scene Exporters (mayatk / blendertk) describe their tasks and checks
as declarative *definitions* -- ``{name: {"widget_type", "object_name",
"setChecked", ...}}`` -- from which each panel builds its widgets, and the
export button reads those widgets back into the ``tasks`` dict
``perform_export`` takes. That read, and the coupled entries it derives (a
texture template arms its compatibility check, an optimize choice sets the
size cap and its check), is the same rule in both panels; this module is the
one copy of it.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Mapping


class ExportProfile:
    """Pure helpers over Scene Exporter task / check definitions."""

    CHECKBOX = "QCheckBox"

    # ------------------------------------------------------------------ naming
    @staticmethod
    def legal_name(name: str) -> str:
        """The switchboard's objectName rule: non-alphanumerics become ``_``.

        The one place the rule lives; ``uitk.SwitchboardNameMixin.
        convert_to_legal_name`` delegates here so the widget the panel builds
        from a definition and the key :meth:`read_values` looks it up by agree
        by construction.
        """
        return re.sub(r"[^0-9a-zA-Z]", "_", name)

    @classmethod
    def widget_key(cls, name: str, spec: Mapping[str, Any]) -> str:
        """The objectName the panel gives *name*'s widget (and the preset stores)."""
        return spec.get("object_name") or cls.legal_name(name)

    @classmethod
    def value_method(cls, spec: Mapping[str, Any]) -> str:
        """The read the export button performs on the widget (``b000``'s rule)."""
        method = spec.get("value_method")
        if method:
            return method
        widget_type = spec.get("widget_type", cls.CHECKBOX)
        return "isChecked" if widget_type == cls.CHECKBOX else "currentData"

    # ------------------------------------------------------------- run config
    @classmethod
    def run_config(
        cls,
        values: Mapping[str, Any],
        task_definitions: Mapping[str, Mapping[str, Any]],
        check_definitions: Mapping[str, Mapping[str, Any]],
        override_checks: bool = False,
        ignore_groups_case_sensitive: bool = False,
        default_export_mode: str = "visible",
    ) -> Dict[str, Any]:
        """The export button's contract: widget values -> ``perform_export`` inputs.

        Exactly what both panels' ``b000`` did inline: read each task / check by
        its widget key, derive the coupled entries (a texture template arms the
        material-compatibility check; an optimize choice sets the size cap and
        arms the optimization check), drop what is off, wrap ``ignore_groups``
        with its match mode, and lift the scope out of the tasks.

        Returns:
            ``{"tasks": {...}, "export_mode": str, "export_visible": bool}`` --
            ``tasks`` is the dict ``perform_export`` takes (without
            ``output_format``, which is a settings row the caller adds).
        """
        task_params: Dict[str, Any] = {}
        for name, spec in task_definitions.items():
            key = cls.widget_key(name, spec)
            if key in values:
                task_params[name] = values[key]
        check_params: Dict[str, Any] = {}
        for name, spec in check_definitions.items():
            key = cls.widget_key(name, spec)
            if key in values:
                check_params[name] = values[key]

        texture_template = task_params.get("convert_textures")
        if texture_template:
            check_params["check_material_compatibility"] = texture_template

        optimize_choice = task_params.get("optimize_textures")
        if optimize_choice:
            if optimize_choice is not True:
                task_params["texture_max_size"] = optimize_choice
            optimize_value = texture_template or True
            task_params["optimize_textures"] = optimize_value
            check_params["check_texture_optimization"] = optimize_value

        task_params = {k: v for k, v in task_params.items() if v}
        check_params = (
            {} if override_checks else {k: v for k, v in check_params.items() if v}
        )

        if "ignore_groups" in task_params:
            task_params["ignore_groups"] = {
                "names": task_params["ignore_groups"],
                "case_sensitive": bool(ignore_groups_case_sensitive),
            }

        export_mode = task_params.pop("export_visible_objects", default_export_mode)
        return {
            "tasks": {**task_params, **check_params},
            "export_mode": export_mode,
            "export_visible": export_mode != "selected",
        }

    @classmethod
    def read_values(
        cls,
        widgets: Mapping[str, Any],
        *tables: Mapping[str, Mapping[str, Any]],
    ) -> Dict[str, Any]:
        """Read the panel's live widgets into ``{objectName: value}``.

        *widgets* maps objectName to widget (the panel's ``ui``); each
        definition's widget is read with :meth:`value_method`, the button's
        own rule, so :meth:`run_config` reads what the screen shows.
        """
        values: Dict[str, Any] = {}
        for table in tables:
            for name, spec in table.items():
                key = cls.widget_key(name, spec)
                widget = widgets.get(key) if isinstance(widgets, Mapping) else None
                if widget is None:
                    widget = getattr(widgets, key, None)
                method = cls.value_method(spec)
                if widget is not None and hasattr(widget, method):
                    values[key] = getattr(widget, method)()
        return values
