# !/usr/bin/python
# coding=utf-8
"""The Scene Exporter panels' shared contract, written once.

Both DCC Scene Exporters (mayatk / blendertk) describe their tasks and checks
as declarative *definitions* -- ``{name: {"widget_type", "object_name",
"setChecked", ...}}`` -- from which each panel builds its widgets, and the
export button reads those widgets back into the ``tasks`` dict
``perform_export`` takes. That read, and the coupled entries it derives (a
texture template arms its compatibility check, an optimize choice sets the
size cap and its check), is the same rule in both panels; this module is the
one copy of it (:class:`ExportProfile`).

The same goes for what the two pipelines run and in what order -- the task
order, which tasks each check reads, the combo tables the panels offer
(:class:`ExportProfile`'s run tables) -- and for the per-run modes the export
button's dict carries beside the tasks, which :class:`ExportRun` parses out
of it and holds, frozen, for the length of the run.
"""

from __future__ import annotations

import dataclasses
import os
import re
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, Iterable, List, Mapping, Optional, Tuple

from pythontk.str_utils._str_utils import StrUtils


class ExportProfile:
    """Pure helpers over Scene Exporter task / check definitions."""

    CHECKBOX = "QCheckBox"

    # ------------------------------------------------------------------ naming
    @staticmethod
    def legal_name(name: str) -> str:
        """The switchboard's objectName rule, from its owner.

        Thin pass-through to :meth:`StrUtils.to_legal_name`, which is where a
        generic string rule belongs: this class is the export button's
        contract, not the home of the naming convention the whole switchboard
        derives objectNames with. Kept as a name because :meth:`widget_key`
        and the panels read it here.
        """
        return StrUtils.to_legal_name(name)

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

    # -------------------------------------------------------------- check rows
    @staticmethod
    def texture_size_limit_bytes(max_size_mb: Any) -> Optional[int]:
        """The Max Texture Size row's value as bytes; None when it is OFF.

        ``None``, ``0`` (the spin box's "OFF" position), ``""``, ``"OFF"``, a
        negative number and non-numeric text are all OFF; a number or numeric
        text (``"16"``, from a hand-edited template) is megabytes. One reading
        for a pre-write size check and for the post-write ``glb_image_bytes``
        gate given the same limit.

        Parameters:
            max_size_mb: The row's value.

        Returns:
            The limit in bytes, or None when the check is off.
        """
        if not max_size_mb or str(max_size_mb).strip().upper() == "OFF":
            return None
        try:
            limit = int(float(max_size_mb) * 1024 * 1024)
        except (TypeError, ValueError, OverflowError):
            return None
        return limit if limit > 0 else None

    # ------------------------------------------------------------ output name
    #: The bare token the Output Filename accepts as sugar for ``{name}``. The
    #: ONLY wildcard: a name resolves to ONE value, so a match-anything glob has
    #: nothing left to mean, and every other filename metacharacter is illegal
    #: on Windows anyway.
    NAME_WILDCARD = "*"

    #: The token the wildcard (and a blank field) stands for -- the ONE spelling
    #: of the exported name. It is the scene's own name, so it is called that:
    #: a second token meaning "the scene name, with the RegEx already applied"
    #: existed only while the RegEx lived in its own field, and an inline
    #: ``{scene:PATTERN->REPLACEMENT}`` says that explicitly.
    NAME_KEY = "scene"

    #: Retired spellings of :attr:`NAME_KEY`, honoured for one release. A
    #: consumer keeps them resolvable in its context; they are listed here so a
    #: folded modifier reaches a pattern that still uses one -- a saved
    #: ``{name}_x`` must keep getting the RegEx it has always had.
    NAME_KEY_ALIASES = ("name",)

    #: The Output Filename's version counter: ``{n}`` is the next version the
    #: name has in the output folder, so ``*_v{n:03d}`` versions every export.
    VERSION_TOKEN = "n"

    #: A version series' trailing ``_v<N>`` -- the tail a ``*_v{n:03d}`` name
    #: ends in, which each exporter strips to key the scene-data manifest every
    #: version of a series shares. The one copy: both DCC ``SceneDataSidecar``
    #: classes and ``ExportVerifier`` read it.
    VERSION_SUFFIX_RE = re.compile(r"_v\d+$", re.IGNORECASE)

    #: The files each output format ships, by ``output_format`` token: what a
    #: version counter must find taken (a GLB-only export leaves no ``.fbx``
    #: behind, so counting those numbered every export v001) and what the
    #: Output Filename preview names.
    OUTPUT_EXTENSIONS: Dict[str, Tuple[str, ...]] = {
        "fbx": (".fbx",),
        "glb": (".glb",),
        "fbx_glb": (".fbx", ".glb"),
        "usd": (".usd",),
    }

    @staticmethod
    def strip_deliverable_extension(name: Optional[str]) -> str:
        """*name* trimmed, without a trailing deliverable extension -- a whitelist
        strip (the carrier vocabulary), so a dotted version token survives and
        "asset.fbx" typed into a USD export lands as "asset.usd"."""
        from pythontk.core_utils.app_handoff import CARRIER_BY_EXTENSION

        return StrUtils.strip_suffix((name or "").strip(), tuple(CARRIER_BY_EXTENSION))

    @classmethod
    def fold_legacy_regex(cls, name_regex: Optional[str]) -> Optional[str]:
        """The retired free-standing RegEx field's text as an inline modifier spec.

        DEPRECATED input, honoured so a saved field keeps shaping the name it
        always did. The field grew its own three-delimiter grammar; the token
        system's modifier (:meth:`StrUtils.split_regex_modifier`) takes ``->`` /
        ``=>`` only, because ``|`` is regex ALTERNATION -- splitting on it makes
        ``(foo|bar)->baz`` unwritable. This folds the two legacy spellings the
        modifier does not accept into ones it does, so ONE regex implementation
        serves both:

        - ``A|B``      -> ``A->B``  (the old "replace A with B" shorthand)
        - ``PATTERN``  -> ``PATTERN->``  (a bare pattern deletes its match)

        Returns:
            (str | None) A spec :meth:`StrUtils.apply_regex_modifier` accepts;
            ``None`` when *name_regex* is blank.
        """
        spec = (name_regex or "").strip()
        if not spec:
            return None
        if StrUtils.split_regex_modifier(spec) is not None:
            return spec
        if "|" in spec:
            pattern, replacement = spec.split("|", 1)
            return f"{pattern.strip()}->{replacement.strip()}"
        return f"{spec}->"

    @classmethod
    def fold_legacy_naming(
        cls,
        pattern: Optional[str],
        version_format: str = "",
        timestamp: bool = False,
        name_regex: Optional[str] = None,
    ) -> Optional[str]:
        """Fold the retired Version pattern and Timestamp flag into a name pattern.

        DEPRECATED inputs, honoured for one release. *name_regex* is the retired
        free-standing RegEx field, which shaped the name token wherever the
        pattern used it; it folds to an inline modifier on that token
        (:meth:`StrUtils.attach_modifier`), so the rule survives the field and
        becomes visible in the field that states the name. The other two
        decorated the name the Output Filename produced -- the timestamp appended, then the Version
        pattern wrapping the result as ``{stem}`` -- and the field spells both
        itself now, so substituting its own placeholder form lands the same file.
        *pattern* comes back untouched when neither is given.
        """
        spec = cls.fold_legacy_regex(name_regex)
        if not (version_format or timestamp or spec):
            return pattern
        folded = StrUtils.expand_wildcard(
            cls.strip_deliverable_extension(pattern),
            key=cls.NAME_KEY,
            wildcard=cls.NAME_WILDCARD,
        )
        if spec:
            # Into the pattern, not onto the value: retiring the field must not
            # drop the rule, and the user can now SEE what it does. Every
            # spelling of the name token gets it, retired ones included --
            # the field shaped the NAME, not one way of writing it.
            for token in (cls.NAME_KEY, *cls.NAME_KEY_ALIASES):
                folded = StrUtils.attach_modifier(folded, token, spec)
        if timestamp:
            folded += "_{date}_{time}"
        if version_format:
            folded = version_format.replace("{stem}", folded)
        return folded

    @classmethod
    def resolve_output_path(
        cls,
        pattern: Optional[str],
        context: Mapping[str, Any],
        export_dir: str = "",
        output_format: str = "fbx",
        version_format: str = "",
        timestamp: bool = False,
        name_regex: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Resolve the Output Filename field into the file(s) an export writes.

        The pattern resolves through :meth:`StrUtils.resolve_name_pattern`
        (blank or ``*`` is ``context["name"]``, ``{tokens}`` fill from
        *context*) with the counter kept; the counter then takes the next
        version the name has among the files *output_format* ships in
        *export_dir* (:meth:`FileUtils.next_version_number`). A counter spec an
        int cannot take (``{n:s}``) numbers plainly. Diagnostics are returned,
        not logged -- :meth:`naming_report` words them.

        Parameters:
            pattern: The Output Filename text.
            context: token -> value (the DCC's scene tokens + the universal ones).
            export_dir: The folder written to; the counter scans it. Blank
                scans nothing and numbers from 1.
            output_format: An :attr:`OUTPUT_EXTENSIONS` key.
            version_format: DEPRECATED -- see :meth:`fold_legacy_naming`.
            timestamp: DEPRECATED -- see :meth:`fold_legacy_naming`.
            name_regex: DEPRECATED -- the retired free-standing RegEx field;
                write ``{scene:PATTERN->REPLACEMENT}`` into the pattern instead.

        Returns:
            ``resolve_name_pattern``'s dict plus ``"stem"`` (the final name),
            ``"path"`` (``.usd`` for USD, else ``.fbx`` -- a GLB-only run names
            its temp FBX after it), ``"paths"`` (every file *output_format*
            ships), ``"n"`` (the counter's value; None without one),
            ``"folded"`` (the pattern a retired input folded to, else None) and
            ``"counter_error"`` (why a counter spec was dropped, else None).
        """
        from pythontk.file_utils._file_utils import FileUtils

        folded = cls.fold_legacy_naming(pattern, version_format, timestamp, name_regex)
        result = StrUtils.resolve_name_pattern(
            cls.strip_deliverable_extension(folded),
            context,
            wildcard=cls.NAME_WILDCARD,
            key=cls.NAME_KEY,
            keep=(cls.VERSION_TOKEN,),
        )
        template, counter_error = result["template"], None
        stem, n = result["name"], None
        if result["kept"]:
            try:
                template.format(**{cls.VERSION_TOKEN: 1})
            except (ValueError, KeyError, IndexError) as e:
                counter_error, template = str(e), cls._plain_counter(template)
            suffixes = cls._output_extensions(output_format)
            n = (
                FileUtils.next_version_number(
                    export_dir,
                    template + "{ext}",
                    ext=suffixes[0],
                    extensions=suffixes[1:],
                )
                if export_dir
                else 1
            )
            stem = template.format(**{cls.VERSION_TOKEN: n})
        folder = export_dir or ""
        primary = ".usd" if output_format == "usd" else ".fbx"
        return {
            **result,
            "template": template,
            "stem": stem,
            "n": n,
            "path": os.path.join(folder, stem + primary),
            "paths": [
                os.path.join(folder, stem + suffix)
                for suffix in cls._output_extensions(output_format)
            ],
            "folded": folded if folded != pattern else None,
            "counter_error": counter_error,
        }

    @classmethod
    def naming_report(
        cls,
        resolved: Mapping[str, Any],
        tokens: Mapping[str, str],
        version_suffix=None,
    ) -> List[Tuple[str, str]]:
        """``[(level, message), ...]`` for what :meth:`resolve_output_path` hit.

        *level* is a logger method name (``"error"`` / ``"warning"``), so a panel
        reports through its own logger at the severity given. *tokens* is the
        vocabulary named when a token had no value.

        *version_suffix* is DEPRECATED and ignored. It used to warn that a
        versioned name not ending in ``_v<N>`` would lose its hierarchy diff
        baseline across versions -- true while that baseline was keyed by the
        output file's stem, which is exactly the coupling that made renaming an
        export reset its history. The baseline is per-SCENE now
        (``HierarchyBaseline``), so no filename can carry or lose it and the
        warning has nothing left to warn about.
        """
        report = []
        if resolved.get("folded"):
            report.append(
                (
                    "warning",
                    "The Version and Timestamp settings are retired: write them "
                    "into the Output Filename instead — this export resolves as "
                    f"{resolved['folded']!r}.",
                )
            )
        if resolved["error"]:
            report.append(
                (
                    "error",
                    f"Output filename {resolved['expanded']!r} is not a valid "
                    f"pattern: {resolved['error']}. Using it as literal text.",
                )
            )
        if resolved.get("counter_error"):
            report.append(
                (
                    "error",
                    "Output filename's version counter has an invalid format "
                    f"({resolved['counter_error']}); numbering it plainly.",
                )
            )
        if resolved["unresolved"]:
            report.append(
                (
                    "warning",
                    "Output filename has no value for "
                    + ", ".join("{" + n + "}" for n in resolved["unresolved"])
                    + " — left in the name as typed. Supported: "
                    + ", ".join("{" + n + "}" for n in tokens),
                )
            )
        if resolved["dropped"]:
            report.append(
                (
                    "warning",
                    f"Output filename dropped {' '.join(resolved['dropped'])} — "
                    "not valid in a filename.",
                )
            )
        return report

    @classmethod
    def _output_extensions(cls, output_format: str) -> Tuple[str, ...]:
        """The files *output_format* ships; an unknown token is FBX."""
        return cls.OUTPUT_EXTENSIONS.get(output_format) or cls.OUTPUT_EXTENSIONS["fbx"]

    @classmethod
    def _plain_counter(cls, template: str) -> str:
        """*template* with every counter field reduced to a bare ``{n}``."""
        import string

        out = []
        for literal, field, _spec, _conv in string.Formatter().parse(template):
            out.append(literal.replace("{", "{{").replace("}", "}}"))
            if field is not None:
                out.append("{" + cls.VERSION_TOKEN + "}")
        return "".join(out)

    # ------------------------------------------------------------ run tables
    # The Scene Exporter's declarative tables, written ONCE for both DCC
    # managers. Each ``TaskManager`` is decorated with :meth:`scoped_tables`,
    # which hands it these lists scoped to the tasks and checks it actually
    # implements -- so the two pipelines cannot disagree on order or on what a
    # check reads, and a task one DCC has not ported is a visible, declared
    # gap (:meth:`unimplemented`) rather than a silently different table.

    #: Execution order for the export tasks -- the phases and the reasons a
    #: task sits where it does. A task not listed here runs after the listed
    #: ones, alphabetically (``TaskFactory._order_tasks``).
    TASK_ORDER: List[str] = [
        # Phase 1 -- Environment setup
        "set_workspace",
        "set_linear_unit",
        # Phase 2 -- Object filtering
        "ignore_groups",
        "exclude_hdr",
        # Phase 2.5 -- Name hygiene (before checks/sidecar record the names)
        "conform_shape_names",
        # Phase 3 -- Material cleanup (reassign THEN resolve THEN relativize;
        # the texture-processing pair runs LAST: convert then optimize what
        # will actually ship, and their staged absolute paths must never be
        # seen by convert_to_relative_paths, which would copy them into the
        # project's texture folder).
        "reassign_duplicate_materials",
        "resolve_invalid_texture_paths",
        "convert_to_relative_paths",
        "convert_textures",
        "optimize_textures",
        # Phase 4 -- Animation (flatten THEN bake THEN optimize THEN snap/tie
        # THEN split THEN set range). Flatten first: smart_bake and the FBX
        # write must see the export-representable hierarchy.
        # set_bake_animation_range is LAST -- it owns the bake range, and it
        # can only honor its "never clip a declared take" rule once
        # apply_declared_takes has realized the takes it must cover.
        "flatten_sheared_chains",
        "smart_bake",
        "optimize_keys",
        "snap_keys_to_frame",
        "tie_all_keyframes",
        "export_data_node",
        "apply_declared_takes",
        "set_bake_animation_range",
    ]

    #: Tasks that REMOVE nodes from the export set, so every check reading the
    #: export set is downstream of them.
    OBJECT_SET_TASKS: Tuple[str, ...] = ("ignore_groups", "exclude_hdr")
    #: Tasks that rewrite what a texture node points at -- which materials own
    #: it (reassign), where the path resolves (resolve/relativize) and what
    #: file is actually there (convert/optimize).
    TEXTURE_PATH_TASKS: Tuple[str, ...] = (
        "reassign_duplicate_materials",
        "resolve_invalid_texture_paths",
        "convert_to_relative_paths",
        "convert_textures",
        "optimize_textures",
    )
    #: Tasks that edit anim curves. ``flatten_sheared_chains`` belongs here as
    #: well as in the hierarchy set: it re-wraps a chain's local matrices, and
    #: a sampled flatten writes keys.
    KEY_EDIT_TASKS: Tuple[str, ...] = (
        "flatten_sheared_chains",
        "smart_bake",
        "optimize_keys",
        "snap_keys_to_frame",
        "tie_all_keyframes",
    )

    #: For each check, the tasks whose execution can change its verdict --
    #: ``TaskFactory._schedule`` reads this to hoist each check above the
    #: tasks it does not read, so a gate that was always going to fail fails
    #: BEFORE the texture and animation phases have burned minutes on a
    #: deliverable that will not be written (and, for a check that depends on
    #: nothing enabled, before the scene is touched at all). ``TASK_ORDER``
    #: itself is never reordered: only the checks move.
    #:
    #: Adding a task means auditing this map. Over-declaring only costs an
    #: early abort; UNDER-declaring makes a check judge a scene the pipeline
    #: has not finished preparing, so when in doubt, declare the dependency.
    CHECK_DEPENDENCIES: Dict[str, Tuple[str, ...]] = {
        # --- General -----------------------------------------------------
        # Scans the scene's references; no task creates, imports or removes
        # one, so this is decidable before the first mutation.
        "check_referenced_objects": (),
        # Reads the destination files, which no task writes or touches -- and
        # the whole point is to fail before the pipeline spends anything.
        "check_output_writable": (),
        # Reads the scene time unit, but only once the export set is known to
        # carry keys at all -- which the filters can empty, and smart_bake can
        # fill (a constraint-driven node has no curves until it is baked).
        "check_framerate": OBJECT_SET_TASKS + ("smart_bake",),
        # --- Hierarchy & Naming ------------------------------------------
        "check_geometry_lod_suffix": OBJECT_SET_TASKS + ("conform_shape_names",),
        # Its widest scope ("Connected & Animated") selects by INCOMING
        # transform connections -- which flatten cuts and smart_bake creates --
        # and a reparent silently number-suffixes a name that collides under
        # its new parent, so both hierarchy tasks move this verdict.
        "check_duplicate_names": OBJECT_SET_TASKS
        + ("conform_shape_names", "flatten_sheared_chains", "smart_bake"),
        "check_duplicate_locator_names": OBJECT_SET_TASKS
        + ("conform_shape_names", "flatten_sheared_chains", "smart_bake"),
        "check_mangled_names": OBJECT_SET_TASKS
        + ("conform_shape_names", "flatten_sheared_chains"),
        # set_linear_unit rescales every translate the check reads against
        # identity; flatten can bake a root's local matrix away.
        "check_root_default_transforms": ("set_linear_unit",)
        + OBJECT_SET_TASKS
        + ("flatten_sheared_chains",),
        # flatten_sheared_chains exists to clear this one; smart_bake writes
        # the matrices it then samples.
        "check_sheared_local_transforms": OBJECT_SET_TASKS
        + ("flatten_sheared_chains", "smart_bake"),
        # Diffs the FULL export hierarchy against the sidecar baseline, so
        # every task that renames a node, re-parents one, or appends the
        # data_export carrier moves it.
        "check_hierarchy_vs_existing_fbx": OBJECT_SET_TASKS
        + (
            "conform_shape_names",
            "flatten_sheared_chains",
            "export_data_node",
            "apply_declared_takes",
        ),
        # --- Geometry ----------------------------------------------------
        # smart_bake bakes visibility, which is half of what this reads.
        "check_hidden_geometry": OBJECT_SET_TASKS + ("smart_bake",),
        "check_overlapping_duplicate_mesh": OBJECT_SET_TASKS,
        # Reads the export meshes' UV sets, which no task adds or removes.
        "check_uv_snapshots": OBJECT_SET_TASKS,
        # The floor tolerance is in SCENE UNITS and the bbox is world-space:
        # set_linear_unit rescales both sides, smart_bake can move the object.
        "check_objects_below_floor": ("set_linear_unit",)
        + OBJECT_SET_TASKS
        + ("smart_bake",),
        # --- Materials & Paths -------------------------------------------
        "check_default_materials": OBJECT_SET_TASKS + ("reassign_duplicate_materials",),
        "check_duplicate_materials": OBJECT_SET_TASKS
        + ("reassign_duplicate_materials",),
        "check_material_compatibility": OBJECT_SET_TASKS + TEXTURE_PATH_TASKS,
        "check_texture_optimization": OBJECT_SET_TASKS + TEXTURE_PATH_TASKS,
        # set_workspace is what a relative texture path resolves AGAINST, so
        # both path gates read its result.
        "check_path_length": ("set_workspace",) + OBJECT_SET_TASKS + TEXTURE_PATH_TASKS,
        "check_valid_paths": ("set_workspace",) + OBJECT_SET_TASKS + TEXTURE_PATH_TASKS,
        "check_texture_file_size": OBJECT_SET_TASKS + TEXTURE_PATH_TASKS,
        # --- Animation ---------------------------------------------------
        "check_untied_keyframes": OBJECT_SET_TASKS + KEY_EDIT_TASKS,
        "check_floating_point_keys": OBJECT_SET_TASKS + KEY_EDIT_TASKS,
    }

    @classmethod
    def scoped_tables(cls, manager: type) -> type:
        """Class decorator: give *manager* the shared tables, scoped to it.

        ``manager.TASK_ORDER`` becomes :attr:`TASK_ORDER` without the tasks
        the class does not implement, and ``manager.CHECK_DEPENDENCIES``
        becomes :attr:`CHECK_DEPENDENCIES` for the checks it implements, each
        dependency list likewise scoped (a task that does not exist can never
        run, so it is not a barrier). What was dropped is what
        :meth:`unimplemented` reports.
        """
        manager.TASK_ORDER = cls.task_order(manager)
        implemented = set(manager.TASK_ORDER)
        manager.CHECK_DEPENDENCIES = {
            check: tuple(task for task in tasks if task in implemented)
            for check, tasks in cls.CHECK_DEPENDENCIES.items()
            if callable(getattr(manager, check, None))
        }
        return manager

    @classmethod
    def task_order(cls, manager: Any) -> List[str]:
        """:attr:`TASK_ORDER` scoped to the tasks *manager* implements."""
        return [t for t in cls.TASK_ORDER if callable(getattr(manager, t, None))]

    @classmethod
    def unimplemented(cls, manager: Any) -> Dict[str, List[str]]:
        """The shared tables' names *manager* has no method for.

        Returns:
            ``{"tasks": [...], "checks": [...]}`` in table order -- the parity
            gap a mirror declares (and its tests pin) rather than hides.
        """
        return {
            "tasks": [
                t for t in cls.TASK_ORDER if not callable(getattr(manager, t, None))
            ],
            "checks": [
                c
                for c in cls.CHECK_DEPENDENCIES
                if not callable(getattr(manager, c, None))
            ],
        }

    # --------------------------------------------------------- combo tables
    # ``{label: value}`` for the panels' combo rows -- presentation shared by
    # both DCCs, so a label edit lands in each. Every combo persists by INDEX,
    # so a sentinel stays where it is and a new choice APPENDS.

    #: Output Format -- what the run writes. Values are the ``output_format``
    #: tokens :class:`ExportRun` reads.
    OUTPUT_FORMATS: Dict[str, str] = {
        "FBX": "fbx",
        "GLB": "glb",
        "FBX + GLB": "fbx_glb",
        "USD": "usd",
    }
    #: Export Scope -- which objects the run starts from.
    EXPORT_MODE_OPTIONS: Dict[str, str] = {
        "All Scene Objects": "all",
        "All Visible Objects": "visible",
        "Selected Objects Only": "selected",
    }
    #: Texture Output -- do the texture-processing tasks modify the scene's
    #: textures, or stage copies for the export and restore the scene after?
    #: The value is the ``texture_write_back`` mode.
    TEXTURE_OUTPUT_OPTIONS: Dict[str, bool] = {
        "Export Copies (Scene Untouched)": False,
        "Scene Files (In Place)": True,
    }
    #: Animation Output -- the same question for the tasks that edit KEYS
    #: (the ``animation_write_back`` mode). Export Copies by default: an
    #: export is an act of publishing, and until this gate existed it silently
    #: rewrote the artist's curves.
    ANIMATION_OUTPUT_OPTIONS: Dict[str, bool] = {
        "Export Copies (Scene Untouched)": False,
        "Scene Keys (In Place)": True,
    }
    #: Longest-edge ceilings Optimize Textures offers -- the Map Converter's
    #: clamp choices, minus 256 (a scene export never wants that small).
    TEXTURE_MAX_SIZES: Tuple[int, ...] = (512, 1024, 2048, 4096, 8192)
    #: Bake Range -- ONE dial owning the FBX bake range. OFF is index 0 and
    #: the falsy sentinel (the run config drops the task).
    BAKE_RANGE_OPTIONS: Dict[str, Optional[str]] = {
        "OFF": None,
        "Auto (Shots → Keyframes)": "auto",
        "Keyframe Extent": "keys",
        "Scene Animation Range": "scene",
    }
    #: Animation Clips -- WHAT the deliverable ships. Ordered least-to-most,
    #: ``both`` LAST so the historical shape keeps the highest index.
    ANIMATION_CLIPS_OPTIONS: Dict[str, str] = {
        "Full Sequence Only": "full",
        "Shots Only": "shots",
        "Shots + Full Sequence": "both",
    }
    #: Optimize Keys -- the pass switch and its aggressiveness in ONE combo.
    #: The SEMANTICS live in each DCC's ``AnimUtils.OPTIMIZE_LEVELS`` (one
    #: table, shared with SmartBake); these are the labels for them.
    OPTIMIZE_KEYS_OPTIONS: Dict[str, Optional[str]] = {
        "OFF": None,
        "Static Curves Only": "static",
        "Static + Flat Keys": "flat",
        "+ Simplify (lossy)": "simplify",
        "Reduce To Extremes": "extremes",
    }
    #: Secondary Map Size -- a LOWER longest-edge ceiling for the packed data
    #: maps (metallic-roughness / occlusion) of a GLB deliverable, under the
    #: primary ceiling Optimize Textures sets. Color keeps the primary (the
    #: perceptual detail), and so do normals (surface detail a resample
    #: visibly softens). 0 = the same ceiling as everything else. The value is
    #: the ``secondary_max_size`` mode.
    SECONDARY_MAX_SIZE_OPTIONS: Dict[str, int] = {
        "Same As Other Maps": 0,
        "Max 512": 512,
        "Max 1024": 1024,
        "Max 2048": 2048,
    }
    #: KTX2 UASTC RDO -- rate-distortion optimisation for the GLB's UASTC
    #: (normal / data) encodes: the blocks are steered toward what Zstandard
    #: compresses at a controlled quality cost. Measured on a 4K production
    #: set: ORM packs -30% at lambda 1 (PSNR 50/44/48 dB), a noisy normal map
    #: -3.5%, encode 3-4x slower; normals are capped at 0.75 whatever the dial
    #: says (toktx's own guidance). The value is the ``uastc_rdo`` mode.
    UASTC_RDO_OPTIONS: Dict[str, float] = {
        "OFF": 0,
        "Light (lambda 0.5)": 0.5,
        "Standard (lambda 1)": 1.0,
        "Strong (lambda 2)": 2.0,
    }
    #: GLB Key Tolerance -- the GLB half of Optimize Keys. FBX2glTF bakes a
    #: key on every frame, so the DCC-side optimisation never reaches the
    #: deliverable; its clips are reduced to this bound instead (the motion
    #: needs ~7% of the keys). The value is the ``glb_key_tolerance`` mode:
    #: the deviation any original sample may show, in scene units (meters)
    #: for translation / scale and quaternion components for rotation --
    #: 1e-4 is 0.1 mm / 0.006 degrees. Inert while Optimize Keys is OFF.
    GLB_KEY_REDUCTION_OPTIONS: Dict[str, float] = {
        "Keep Every Key": 0,
        "Within 1e-6 (near-exact)": 1e-6,
        "Within 1e-4 (0.1 mm)": 1e-4,
        "Within 1e-3 (1 mm)": 1e-3,
    }

    @classmethod
    def optimize_textures_options(cls) -> Dict[str, Any]:
        """Optimize Textures -- the pass switch and its size dial in ONE combo.

        Falsy 0 = OFF (the run config drops the task), True = optimize without
        resampling, an int = optimize + hard pixel ceiling, and the
        ``MapOptimizer.SIZE_CLAMP_TEMPLATE`` sentinel = optimize + enforce the
        selected template's own budget. OFF is index 0 and the sentinel LAST.
        """
        from pythontk.core_utils.engines.textures.map_optimizer import MapOptimizer

        return {
            "OFF": 0,
            "Optimize": True,
            **{f"Optimize + Max {s}": s for s in cls.TEXTURE_MAX_SIZES},
            "Optimize + Template Budget": MapOptimizer.SIZE_CLAMP_TEMPLATE,
        }

    @classmethod
    def texture_file_type_options(cls) -> Dict[str, Any]:
        """Texture File Type -- the container dial for EVERY texture the export
        ships (scene/FBX maps and a GLB's embedded copies alike).

        Built from the shared registry so a container added to ``ImgUtils``
        appears here and in the Map Converter's own Format menu without an
        edit, plus the two delivery-only entries a GLB deliverable can carry:
        KTX2 alone, and KTX2 + a core-readable twin of each map
        (:attr:`ExportRun.KTX2_WITH_FALLBACK`). "Original" is index 0 and the
        falsy sentinel: a TEMPLATE contract, so never insert above it.
        """
        from pythontk.core_utils.engines.textures.output_template import (
            OutputTemplates,
        )

        return {
            **dict(
                OutputTemplates.format_choices(sentinel="Original", sentinel_first=True)
            ),
            "KTX2": "ktx2",
            "KTX2 + PNG/JPEG": ExportRun.KTX2_WITH_FALLBACK,
        }

    @classmethod
    def frame_rate_options(cls) -> Dict[str, Optional[str]]:
        """Frame Rate check -- every ``VidUtils.FRAME_RATES`` entry labelled
        with its fps, behind an OFF sentinel."""
        from pythontk.iter_utils._iter_utils import IterUtils
        from pythontk.vid_utils._vid_utils import VidUtils

        rates = IterUtils.insert_into_dict(VidUtils.FRAME_RATES, "OFF", None)
        return {
            (
                f"{k}"
                if v is None
                else (
                    f"{v:g} fps" if any(c.isdigit() for c in k) else f"{k} ({v:g} fps)"
                )
            ): (k if v is not None else None)
            for k, v in rates.items()
        }


@dataclass(frozen=True)
class ExportRun:
    """The modes of ONE Scene Exporter run, decided before its pipeline runs.

    Both DCC exporters pop every UI-only setting out of the ``tasks`` dict
    their export button hands them (:meth:`from_tasks`), resolve the output
    path, and hand the result to their task manager (``TaskManager.begin_run``)
    -- the one per-run reset. Tasks and checks then read ``self.run.<mode>``;
    nothing writes a mode mid-run (the object is frozen), and a reader never
    needs a ``getattr`` default because every mode is declared here with the
    value a run that never named it has.

    The three modes derived from dispatched tasks (:meth:`with_tasks`) are
    re-read by ``TaskManager.run_tasks`` off the FULL task dict, so a caller
    that drives the manager directly gets them too -- and an override that
    resumes a subset of tasks never re-derives them from that subset.
    """

    #: The file the run writes (``.fbx``, or ``.usd``); a GLB-only run names
    #: its temp FBX after it. Empty until an exporter resolves one, which is
    #: what a manager driven directly reads.
    export_path: str = ""
    #: An :attr:`OUTPUT_FORMATS` token.
    output_format: str = "fbx"
    #: The retired Version pattern, folded into the name by
    #: :meth:`ExportProfile.fold_legacy_naming`.
    version_format: str = ""
    #: The Texture File Type dial -- the container every texture the export
    #: ships is written in (each destination clamps what it cannot carry).
    #: None is "Original".
    texture_file_type: Optional[str] = None
    #: KTX2 + a core-readable PNG/JPEG twin of every map in the GLB.
    ktx2_fallback: bool = False
    #: Texture Output: True writes the texture passes back over the scene's
    #: own files; False stages copies for the write and restores the scene.
    texture_write_back: bool = False
    #: Animation Output: True leaves the key edits in the scene; False captures
    #: the curves and restores them after the write.
    animation_write_back: bool = False
    #: The Optimize Textures size dial: None/0 = never resample, an int pixel
    #: ceiling, or ``MapOptimizer.SIZE_CLAMP_TEMPLATE``.
    texture_max_size: Any = None
    #: The texture template the passes convert to (the Texture Template combo,
    #: or the optimize row's own template), None for "as authored".
    texture_template: Optional[str] = None
    #: Whether the optimize_textures task runs this pass -- read by the GLB
    #: half after the pipeline, so it is a mode as well as a task.
    optimize_textures: bool = False
    #: The Optimize Keys level the run asked for, handed to smart_bake so it
    #: optimizes its own output at the same level (derived: :meth:`with_tasks`).
    optimize_keys_level: Any = False
    #: Whether convert_to_relative_paths runs, so a write-back texture
    #: conversion relativizes what it rewired (derived: :meth:`with_tasks`).
    relative_paths: bool = False
    #: The Animation Clips row's own value (``full`` / ``shots`` / ``both``, or
    #: the boolean a pre-combo preset stored), unresolved -- the exporter
    #: normalises it once, and hands it to the shots producer as INPUT through
    #: the export context, so the shot record is declared, never patched
    #: (derived: :meth:`with_tasks`; ``"both"`` when the row is absent, the
    #: mode a run that never realizes takes still converts its GLB under).
    animation_clips_mode: Any = "both"
    #: The write splits the scene's declared takes: the Animation Clips row ran
    #: in a shot-bearing mode (derived: :meth:`with_tasks`). Whether the scene
    #: declares any takes is the DCC's read, off its carrier.
    splits_takes: bool = False
    #: The Output Filename carried a version counter, so the sidecar routes
    #: through the base stem and a series shares one manifest.
    versioned: bool = False
    #: The Verify The Written File row: re-open the deliverables after the
    #: write and run the file-level gates.
    verify_deliverables: bool = False
    #: The Exclude Rig Helpers row: the written FBX -- and so the GLB built
    #: from it -- drops the apparatus of the rig it baked (the DCC's census,
    #: removed by ``FbxMedia.drop_apparatus``). FBX only: a USD layer is
    #: written as the scene samples.
    drop_rig_apparatus: bool = False
    #: The written FBX carries its own texture copies (embedded media, or a
    #: path mode that copies them beside it), so staged textures are temp.
    fbx_media_selfcontained: bool = False
    #: Secondary Map Size: a lower pixel ceiling for a GLB's packed data maps
    #: (:attr:`ExportProfile.SECONDARY_MAX_SIZE_OPTIONS`); None is "the same
    #: as Optimize Textures".
    secondary_max_size: Optional[int] = None
    #: KTX2 UASTC RDO lambda for the GLB's UASTC encodes
    #: (:attr:`ExportProfile.UASTC_RDO_OPTIONS`); None is off.
    uastc_rdo: Optional[float] = None
    #: GLB Key Tolerance: the deviation bound for the deliverable's key reduction
    #: (:attr:`ExportProfile.GLB_KEY_REDUCTION_OPTIONS`); None is off.
    glb_key_tolerance: Optional[float] = None

    #: The ``output_format`` tokens a run accepts.
    OUTPUT_FORMATS: ClassVar[Tuple[str, ...]] = ("fbx", "glb", "fbx_glb", "usd")
    #: Texture File Type token for KTX2 PLUS a core-readable PNG/JPEG twin of
    #: every map. :meth:`from_tasks` parses it into the ``ktx2`` container and
    #: the :attr:`ktx2_fallback` flag, so no other consumer ever compares it.
    KTX2_WITH_FALLBACK: ClassVar[str] = "ktx2+fallback"
    #: The FBX-side key hygiene a GLB-only run drops: the FBX is then a temp
    #: intermediate FBX2glTF resamples on every frame, so nothing these do
    #: reaches the deliverable (Optimize Keys reaches it as the GLB key
    #: reduction instead), and Optimize Keys alone costs minutes of a
    #: production run.
    FBX_KEY_HYGIENE: ClassVar[Tuple[str, ...]] = (
        "optimize_keys",
        "snap_keys_to_frame",
        "tie_all_keyframes",
        "check_floating_point_keys",
        "check_untied_keyframes",
    )
    #: The ``tasks`` keys that are modes, not tasks -- what :meth:`from_tasks`
    #: pops (legacy spellings included) so none reaches the dispatcher as an
    #: unknown task.
    MODE_KEYS: ClassVar[Tuple[str, ...]] = (
        "version",
        "output_format",
        "create_glb",
        "texture_file_type",
        "glb_texture_format",
        "glb_optimize_textures",
        "texture_write_back",
        "optimize_textures_write_back",
        "animation_write_back",
        "verify_deliverables",
        "drop_rig_apparatus",
        "texture_max_size",
        "secondary_max_size",
        "uastc_rdo",
        "glb_key_tolerance",
    )

    @staticmethod
    def _dial(value: Any, cast: type) -> Any:
        """A numeric dial's value, or None when it is off (falsy / unreadable)."""
        try:
            number = cast(value) if value else None
        except (TypeError, ValueError):
            return None
        return number or None

    @property
    def glb_only(self) -> bool:
        """The GLB is the deliverable; the FBX is a temp intermediate."""
        return self.output_format == "glb"

    @property
    def create_glb(self) -> bool:
        """A ``.glb`` is written this run (alone, or beside the FBX)."""
        return self.output_format in ("glb", "fbx_glb")

    @property
    def usd(self) -> bool:
        """The deliverable is a USD layer."""
        return self.output_format == "usd"

    def replace(self, **changes: Any) -> "ExportRun":
        """A copy with *changes* applied (``dataclasses.replace``)."""
        return dataclasses.replace(self, **changes)

    @staticmethod
    def clip_mode(value: Any) -> str:
        """An Animation Clips row value as one of its modes (``full`` /
        ``shots`` / ``both``) -- the one resolver both DCC exporters use.

        Accepts the boolean a pre-combo preset stored: a stored preset is a
        contract, and a widget-type change must not silently re-point it at a
        different deliverable. ``True`` kept the whole-timeline stack beside
        the split takes, so it is ``both``; every FALSY value (the unticked
        box, and the ``None`` a headless caller passes for OFF) split nothing
        and shipped the sequence alone, so it is ``full``.

        Raises:
            ValueError: An unknown mode.
        """
        if not value or isinstance(value, bool):
            return "both" if value else "full"
        resolved = str(value).strip().lower()
        modes = tuple(ExportProfile.ANIMATION_CLIPS_OPTIONS.values())
        if resolved not in modes:
            raise ValueError(
                f"Unknown animation clips mode {value!r}; expected one of "
                f"{', '.join(modes)}."
            )
        return resolved

    def with_tasks(self, tasks: Mapping[str, Any]) -> "ExportRun":
        """A copy carrying the modes derived from the dispatched *tasks*.

        ``optimize_keys_level`` is the Optimize Keys row's own value, passed
        through unresolved (each DCC's SmartBake resolves the token against
        its ``AnimUtils.OPTIMIZE_LEVELS``, so there is one table); a
        write-back texture conversion relativizes its rewired paths only when
        ``convert_to_relative_paths`` is on; ``splits_takes`` is whether the
        Animation Clips row runs in a shot-bearing mode (an unknown mode reads
        False here -- the task itself raises, naming it).
        """
        try:
            splits = "apply_declared_takes" in tasks and (
                self.clip_mode(tasks["apply_declared_takes"]) != "full"
            )
        except ValueError:
            splits = False
        return self.replace(
            optimize_keys_level=tasks.get("optimize_keys", False),
            relative_paths=bool(tasks.get("convert_to_relative_paths", False)),
            animation_clips_mode=tasks.get("apply_declared_takes", "both"),
            splits_takes=splits,
        )

    @classmethod
    def from_tasks(
        cls,
        tasks: Optional[Mapping[str, Any]],
        texture_file_types: Iterable[Any] = (),
    ) -> Tuple["ExportRun", Dict[str, Any], List[Tuple[str, str]]]:
        """Pop the per-run modes out of a ``perform_export`` *tasks* dict.

        The one copy of the rule both DCC exporters applied inline: the
        output format (a legacy ``create_glb`` flag maps to ``fbx_glb``; the
        format wins when both are present), the Texture File Type (a legacy
        ``glb_texture_format`` is read when the new key is absent; the
        redundant ``glb_optimize_textures`` flag is dropped; the KTX2 +
        fallback token becomes ``ktx2`` plus :attr:`ktx2_fallback`; KTX2
        without a GLB to carry it is inert), the two write-back flags (a
        preset saved before the rename carries
        ``optimize_textures_write_back``), the verification row, the rig-helper
        row (inert on a USD run, with a note), the size
        dial and the three GLB optimisation dials (secondary map size, UASTC
        RDO, key reduction -- each None when off; the key reduction rides
        Optimize Keys and is inert without it). A GLB-only format drops
        :attr:`FBX_KEY_HYGIENE` with a note: the FBX is then a temp
        intermediate the converter resamples per frame. ``optimize_textures``
        and ``convert_textures`` are READ, not
        popped -- they are real tasks -- for the template and the pass flag
        the GLB half resolves after the pipeline.

        Parameters:
            tasks: The export button's dict. Not modified.
            texture_file_types: The tokens the Texture File Type dial offers
                (:meth:`ExportProfile.texture_file_type_options` values); a
                value outside them is a config error. Empty skips the gate.

        Returns:
            ``(run, tasks, notes)`` -- the run (``export_path`` and
            ``versioned`` still unset: the exporter resolves the path), the
            tasks dict with every mode key removed, and ``[(level, message)]``
            for the exporter's logger (``"error"`` means the run must not
            proceed; the exporter reports and aborts).
        """
        tasks = dict(tasks or {})
        notes: List[Tuple[str, str]] = []

        version_format = tasks.pop("version", "") or ""
        output_format = str(tasks.pop("output_format", "") or "").lower()
        if not output_format:
            output_format = "fbx_glb" if tasks.pop("create_glb", False) else "fbx"
        else:
            tasks.pop("create_glb", None)  # the format wins over a legacy flag
        if output_format not in cls.OUTPUT_FORMATS:
            notes.append(
                (
                    "warning",
                    f"Unknown output_format {output_format!r} (expected one of "
                    f"{', '.join(cls.OUTPUT_FORMATS)}); exporting FBX.",
                )
            )
            output_format = "fbx"
        create_glb = output_format in ("glb", "fbx_glb")

        texture_file_type = str(tasks.pop("texture_file_type", "") or "").lower()
        legacy_glb_format = str(tasks.pop("glb_texture_format", "") or "").lower()
        tasks.pop("glb_optimize_textures", None)  # redundant: Optimize Textures
        if not texture_file_type and legacy_glb_format:
            texture_file_type = legacy_glb_format
            notes.append(
                (
                    "debug",
                    f"Legacy 'glb_texture_format' {legacy_glb_format!r} read as "
                    "'texture_file_type'.",
                )
            )
        texture_file_type = texture_file_type.lstrip(".") or None
        known = {str(t).lower().lstrip(".") for t in texture_file_types if t}
        if texture_file_type and known and texture_file_type not in known:
            # A hand-edited template / headless caller can send anything; an
            # unknown value is a config error and aborts loudly -- discovered
            # at encode time it would fail per-image and ship an effectively
            # unencoded texture set behind warning noise.
            notes.append(
                (
                    "error",
                    f"Export aborted: unknown texture_file_type "
                    f"{texture_file_type!r} (expected one of "
                    f"{', '.join(sorted(known))}, or empty for Original).",
                )
            )
        ktx2_fallback = texture_file_type == cls.KTX2_WITH_FALLBACK
        if ktx2_fallback:
            texture_file_type = "ktx2"
        if texture_file_type == "ktx2" and not create_glb:
            # KTX2 is a delivery-only container: no scene texture node or FBX
            # importer reads it, so with no GLB to carry it the choice has
            # nowhere to land. Inert, not an error.
            notes.append(
                (
                    "info",
                    "Texture File Type 'KTX2' ignored: it can only ship inside "
                    "a GLB, and the output format produces none.",
                )
            )
            texture_file_type, ktx2_fallback = None, False

        write_back = tasks.pop("texture_write_back", None)
        if write_back is None:
            write_back = tasks.pop("optimize_textures_write_back", False)
        else:
            tasks.pop("optimize_textures_write_back", None)  # new key wins

        optimize = tasks.get("optimize_textures")
        template = tasks.get("convert_textures")
        # GLB Key Tolerance rides Optimize Keys: the converter resamples every
        # frame, so the DCC-side optimisation never reaches the GLB and the
        # tolerance is the GLB half of ONE choice. Read BEFORE the GLB-only
        # gate below pops the task.
        key_tolerance = cls._dial(tasks.pop("glb_key_tolerance", None), float)
        if key_tolerance and not tasks.get("optimize_keys"):
            notes.append(
                (
                    "info",
                    "GLB Key Tolerance ignored: it rides Optimize Keys, which is "
                    "OFF -- the deliverable keeps the converter's per-frame keys.",
                )
            )
            key_tolerance = None
        drop_rig_apparatus = bool(tasks.pop("drop_rig_apparatus", False))
        if drop_rig_apparatus and output_format == "usd":
            # The pass edits a written FBX; a USD layer has none, and its
            # writer samples the live scene, so nothing reaches it. Inert.
            notes.append(
                (
                    "info",
                    "Exclude Rig Helpers ignored: it edits the written FBX, and "
                    "the output format is USD.",
                )
            )
            drop_rig_apparatus = False
        if output_format == "glb":
            skipped = [key for key in cls.FBX_KEY_HYGIENE if tasks.pop(key, None)]
            if skipped:
                notes.append(
                    (
                        "info",
                        f"GLB only: {', '.join(skipped)} skipped -- the FBX is a "
                        "temp intermediate the converter resamples on every "
                        "frame, so its key hygiene never reaches the "
                        "deliverable; Optimize Keys reaches it as the GLB key "
                        "reduction instead.",
                    )
                )
        run = cls(
            output_format=output_format,
            version_format=str(version_format),
            texture_file_type=texture_file_type,
            ktx2_fallback=ktx2_fallback,
            texture_write_back=bool(write_back),
            animation_write_back=bool(tasks.pop("animation_write_back", False)),
            texture_max_size=tasks.pop("texture_max_size", None),
            secondary_max_size=cls._dial(tasks.pop("secondary_max_size", None), int),
            uastc_rdo=cls._dial(tasks.pop("uastc_rdo", None), float),
            glb_key_tolerance=key_tolerance,
            texture_template=(
                template
                if isinstance(template, str)
                else (optimize if isinstance(optimize, str) else None)
            ),
            optimize_textures=bool(optimize),
            verify_deliverables=bool(tasks.pop("verify_deliverables", False)),
            drop_rig_apparatus=drop_rig_apparatus,
        )
        return run.with_tasks(tasks), tasks, notes
