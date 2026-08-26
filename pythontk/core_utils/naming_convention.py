# !/usr/bin/python
# coding=utf-8
"""Qt-free, zero-dependency **naming convention** — the ecosystem's one answer to
"what affix marks a mesh / a material / a group?".

Before this module every tool answered that question for itself: mayatk's and
blendertk's ``Naming.SUFFIX_TYPES`` held two verbatim copies of a 19-row table,
``rig_utils`` hardcoded ``"_GRP"`` / ``"_LOC"`` / ``"_GEO"`` in its signatures,
``image_to_plane`` defaulted to ``"_MAT"``, and the Naming panel persisted the
user's *real* convention onto nineteen ``QLineEdit`` objectNames where nothing
else could read it. Renaming ``_GEO`` to ``_MSH`` meant editing every one of
those places and finding the ones you missed at the wrong moment.

A convention entry is an **affix**, not a suffix: a spelling plus the side of the
name it lands on (see :class:`AffixRule`). That is the same three-mode grammar
the option-box picker exposes (``auto`` / ``suffix`` / ``prefix``) over the same
:py:meth:`pythontk.StrUtils.split_affix` primitive, so a studio that writes
``GEO_body`` rather than ``body_GEO`` changes the convention, not the tools.

Why here and not in the naming module: uitk's ``AffixOption`` resolves a field
against this table and cannot import a DCC toolkit; ``extapps`` has no DCC at
all; and the table itself is plain data with nothing Maya- or Blender-specific
in it. What *is* host-specific — resolving a live scene node to a type key —
stays in ``mayatk`` / ``blendertk`` ``Naming.type_key``, which owns that binding
in both directions.

Storage follows :class:`UserConfig`: the shipped :attr:`NamingConvention.DEFAULTS`
are the baseline and only the entries a user actually changed are written to
``<user_config_root>/pythontk/naming_convention.json`` — so a default that moves
in a later release still reaches everyone who never overrode it.

    >>> NamingConvention.affix_parts("mesh")
    ('', '_GEO')
    >>> NamingConvention.apply("body", "mesh")
    'body_GEO'
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from pythontk.core_utils.user_config import UserConfig

logger = logging.getLogger(__name__)

#: Config doc name / owning package — resolves to
#: ``<user_config_root>/pythontk/naming_convention.json``.
CONFIG_NAME = "naming_convention"
CONFIG_PACKAGE = "pythontk"

#: Env var pointing at an explicit convention file (a project or studio share),
#: honoured ahead of the per-user doc by :class:`UserConfig`'s discovery.
CONFIG_ENV_VAR = "PYTHONTK_NAMING_CONVENTION"

#: The affix placement modes, in the option-box picker's cycle order.
AFFIX_MODES: Tuple[str, str, str] = ("auto", "suffix", "prefix")


@dataclass(frozen=True)
class AffixRule:
    """One convention entry: a spelling plus where it lands on a base name.

    Attributes:
        text: The affix spelling (``"_GEO"``, ``"MAT_"``). Empty disables the
            entry — a type with no affix is left alone rather than renamed.
        mode: ``"auto"`` / ``"suffix"`` / ``"prefix"``. ``"auto"`` infers from
            the delimiter (``"_GEO"`` -> suffix, ``"GEO_"`` -> prefix), which is
            why every shipped default can stay ``"auto"`` and still behave as
            the suffix it reads as.
        label: Human title for the type (``"Nurbs Curve"``), used by the panels
            that render one editor row per entry.
    """

    text: str = ""
    mode: str = "auto"
    label: str = ""

    def parts(self, *, default: str = "suffix") -> Tuple[str, str]:
        """``(prefix, suffix)`` for this rule — at most one is non-empty.

        Parameters:
            default: Placement used when :attr:`mode` is ``"auto"`` and
                :attr:`text` carries no boundary delimiter. Suffix, because
                every convention this ships with is one.
        """
        from pythontk.str_utils._str_utils import StrUtils

        return StrUtils.split_affix(self.text, mode=self.mode, default=default)

    def apply(self, name: str, *, default: str = "suffix") -> str:
        """Idempotently put this rule's affix on *name* (no-op for empty text).

        Safe on a name that IS its own affix: ``apply_affix`` refuses a
        de-duplication strip that would consume the whole name, so a camera called
        ``Cam`` under a ``_CAM`` rule becomes ``Cam_CAM`` rather than a bare
        ``_CAM`` with the name gone.
        """
        from pythontk.str_utils._str_utils import StrUtils

        prefix, suffix = self.parts(default=default)
        return StrUtils.apply_affix(name, prefix=prefix, suffix=suffix)

    def as_dict(self) -> Dict[str, str]:
        """JSON-shaped form (``label`` omitted — it ships with the defaults)."""
        return {"text": self.text, "mode": self.mode}


class _NamingConventionInternal(object):
    """Internal helpers for :class:`NamingConvention`."""

    #: Cached merged table; invalidated by any write and by ``reload()``.
    _cache: Optional[Dict[str, AffixRule]] = None

    @staticmethod
    def _coerce(key: str, value, label: str = "") -> AffixRule:
        """Build an :class:`AffixRule` from a rule, a dict, or a bare string.

        A bare string is the spelling with mode ``"auto"`` — the shape every
        pre-existing caller passes (``mesh_suffix="_GEO"``) and the shape a
        hand-edited config file most naturally takes.
        """
        if isinstance(value, AffixRule):
            return AffixRule(value.text, value.mode, value.label or label or key)
        if isinstance(value, dict):
            text = value.get("text", value.get("affix", ""))
            mode = value.get("mode", "auto")
            label = value.get("label") or label
        else:
            text, mode = value, "auto"
        mode = str(mode or "auto").lower()
        if mode not in AFFIX_MODES:
            logger.warning(
                "[naming_convention] unknown mode %r for %r; using 'auto'.", mode, key
            )
            mode = "auto"
        return AffixRule("" if text is None else str(text), mode, label or key)


class NamingConvention(_NamingConventionInternal):
    """The ecosystem's single source of truth for type-based name affixes.

    Read it through the class namespace — :meth:`get`, :meth:`affix_parts`,
    :meth:`apply` — and edit it through :meth:`set` / :meth:`update`, which
    persist only the entries that differ from :attr:`DEFAULTS`.

    The keys are a **shared vocabulary**, not any one DCC's node types: mayatk
    and blendertk each map their own scene types onto them (``Naming.type_key``),
    so a tentacle slot asking for ``"mesh"`` gets the right answer in either
    host. Anything not listed can still be added — an unknown key is simply an
    entry with no shipped default.
    """

    #: The shipped baseline. ``mode="auto"`` throughout: every spelling here
    #: leads with ``_`` and therefore reads as a suffix, while a user who
    #: rewrites one as ``"GEO_"`` gets a prefix without also touching the mode.
    DEFAULTS: Dict[str, AffixRule] = {
        "group": AffixRule("_GRP", "auto", "Group"),
        "locator": AffixRule("_LOC", "auto", "Locator"),
        "joint": AffixRule("_JNT", "auto", "Joint"),
        "mesh": AffixRule("_GEO", "auto", "Mesh"),
        "nurbsCurve": AffixRule("_CRV", "auto", "Nurbs Curve"),
        "camera": AffixRule("_CAM", "auto", "Camera"),
        "light": AffixRule("_LGT", "auto", "Light"),
        "displayLayer": AffixRule("_LYR", "auto", "Display Layer"),
        "ikHandle": AffixRule("_IKH", "auto", "IK Handle"),
        "nurbsSurface": AffixRule("_SRF", "auto", "Nurbs Surface"),
        "cluster": AffixRule("_CLS", "auto", "Cluster"),
        "lattice": AffixRule("_LAT", "auto", "Lattice"),
        "skinCluster": AffixRule("_SKN", "auto", "Skin Cluster"),
        "blendShape": AffixRule("_BS", "auto", "Blend Shape"),
        "constraint": AffixRule("_CON", "auto", "Constraint"),
        "material": AffixRule("_MAT", "auto", "Material"),
        "shadingEngine": AffixRule("_SG", "auto", "Shading Group"),
        "texture": AffixRule("_TEX", "auto", "Texture"),
        "objectSet": AffixRule("_SET", "auto", "Set"),
        # Artifact entries: not scene NODE types, so no host binds them and
        # "Suffix By Type" never applies them -- but they are conventions all
        # the same (a studio decides lightmaps are "_LM", not "_Lightmap"), and
        # the tools that write these files ask for them by name.
        "lightmap": AffixRule("_Lightmap", "auto", "Lightmap"),
        "packedTexture": AffixRule("_Packed", "auto", "Packed Texture"),
        # A rig control IS a nurbsCurve, but it carries its own marker -- a rig
        # that suffixed its controls "_CRV" would be unreadable -- so it is its
        # own entry rather than a second meaning for the curve one.
        "control": AffixRule("_CTRL", "auto", "Control"),
    }

    #: Entries with no scene-node counterpart of their own. Panels that drive a
    #: *rename by node type* skip these (nothing resolves TO them); panels that
    #: EDIT the convention still show them.
    ARTIFACT_KEYS: Tuple[str, ...] = ("lightmap", "packedTexture", "control")

    # ------------------------------------------------------------------ read
    @classmethod
    def resolve(cls, *, refresh: bool = False) -> Dict[str, AffixRule]:
        """The full table: three layers, each overlaying the one before it.

        ``DEFAULTS`` < the studio/project doc named by
        ``$PYTHONTK_NAMING_CONVENTION`` < the per-user doc. A studio can
        therefore deploy a shared convention without freezing it: an artist
        who overrides one entry keeps that entry and still tracks the share
        for everything else, and the write path only ever touches the doc
        they own (see :meth:`_load_overrides`).

        Cached — every tool reads this on each operation while the doc only
        changes when someone edits it. Pass ``refresh=True`` (or call
        :meth:`reload`) to re-read after an external edit.
        """
        if cls._cache is not None and not refresh:
            return cls._cache

        table = dict(cls.DEFAULTS)
        for layer in (cls._studio_overrides(), cls._load_overrides()):
            for key, value in (layer or {}).items():
                label = table[key].label if key in table else ""
                table[key] = cls._coerce(key, value, label)
        cls._cache = table
        return table

    @classmethod
    def reload(cls) -> Dict[str, AffixRule]:
        """Drop the cache and re-read the config doc."""
        return cls.resolve(refresh=True)

    @classmethod
    def keys(cls) -> List[str]:
        """Every type key, in the shipped order (user-added keys appended)."""
        table = cls.resolve()
        extra = sorted(k for k in table if k not in cls.DEFAULTS)
        return [k for k in cls.DEFAULTS if k in table] + extra

    @classmethod
    def items(cls) -> List[Tuple[str, AffixRule]]:
        """``(key, rule)`` pairs in :meth:`keys` order."""
        table = cls.resolve()
        return [(k, table[k]) for k in cls.keys()]

    @classmethod
    def get(cls, key: str, fallback: Optional[AffixRule] = None) -> AffixRule:
        """The rule for *key*, or *fallback* (an empty, no-op rule by default)."""
        rule = cls.resolve().get(key)
        if rule is not None:
            return rule
        return fallback if fallback is not None else AffixRule(label=key)

    @classmethod
    def label(cls, key: str) -> str:
        """Human title for *key* (the key itself when it has no shipped label)."""
        return cls.get(key).label or key

    @classmethod
    def affix(cls, key: str) -> str:
        """The affix spelling for *key* (``""`` when the type is unset)."""
        return cls.get(key).text

    @classmethod
    def mode(cls, key: str) -> str:
        """The placement mode for *key* (``"auto"`` when unset)."""
        return cls.get(key).mode

    @classmethod
    def affix_parts(cls, key: str, *, default: str = "suffix") -> Tuple[str, str]:
        """``(prefix, suffix)`` for *key* — the pair tools actually apply."""
        return cls.get(key).parts(default=default)

    @classmethod
    def apply(cls, name: str, key: str, *, default: str = "suffix") -> str:
        """Idempotently apply *key*'s affix to *name*.

        Safe on an already-conventional name: the underlying
        :py:meth:`pythontk.StrUtils.apply_affix` strips a pre-existing copy
        first, so ``"body_GEO"`` stays ``"body_GEO"`` rather than doubling.
        """
        return cls.get(key).apply(name, default=default)

    @classmethod
    def all_affixes(cls) -> List[str]:
        """Every non-empty spelling, longest first.

        Longest-first is load-bearing for callers that strip a *wrong* affix
        before applying the right one: ``"_SG"`` is a tail of ``"_LSG"``, and
        testing the short one first would eat half of the long one.
        """
        # Alphabetical tie-break, not bare length: ordering a set by length
        # alone leaves ties to set-iteration order, which Python randomizes per
        # process — the caller's "strip one wrong affix" would then pick a
        # different one from session to session.
        return sorted(
            {r.text for r in cls.resolve().values() if r.text},
            key=lambda a: (-len(a), a),
        )

    @classmethod
    def bind(
        cls,
        bindings: Iterable[Tuple[str, str, str]],
        overrides: Optional[Dict[str, str]] = None,
        modes: Optional[Dict[str, str]] = None,
    ) -> Dict[str, AffixRule]:
        """Join this table to a host's type vocabulary: ``{host key: AffixRule}``.

        The one operation every type-driven rename runs, and the only part of it
        that is host-specific is the *bindings* table each toolkit ships — which
        of ITS node types each convention entry names. Keeping the join here is
        what stops mayatk and blendertk from carrying two copies of it (they
        did, verbatim, along with the 19-row table this replaced).

        Each entry starts as the shared rule and is then overridden per call, so
        a tool can deviate for one run without editing — or being surprised by —
        the stored convention.

        Parameters:
            bindings: Iterable of ``(keyword, convention key, host type key)``.
                The keyword is the engine's public parameter name; it exists so
                a caller can pass its own signature's kwargs straight through.
            overrides: ``{keyword or host type key: affix spelling}``. An empty
                string disables that entry (the type is left alone).
            modes: ``{keyword or host type key: "auto"/"suffix"/"prefix"}`` --
                placement overrides for the same entries.

        Returns:
            One rule per binding, keyed by HOST type key, ready to
            :py:meth:`AffixRule.apply`. Labels come from this table, so a panel
            rendering the result still has its row titles.
        """
        overrides = overrides or {}
        modes = modes or {}
        rules: Dict[str, AffixRule] = {}
        for keyword, convention_key, host_key in bindings:
            rule = cls.get(convention_key)
            # Keyword first: a caller forwarding its own kwargs is the common
            # case, and a host key that happens to equal a keyword must not
            # shadow it.
            text = overrides.get(keyword, overrides.get(host_key, rule.text))
            mode = modes.get(keyword, modes.get(host_key, rule.mode))
            rules[host_key] = AffixRule(
                "" if text is None else str(text), mode, rule.label
            )
        return rules

    # ----------------------------------------------------------------- write
    @classmethod
    def set(cls, key: str, text: str, mode: str = "auto", label: str = "") -> AffixRule:
        """Set one entry and persist it. Returns the stored rule."""
        return cls.update({key: {"text": text, "mode": mode, "label": label}})[key]

    @classmethod
    def update(cls, mapping: Dict[str, object]) -> Dict[str, AffixRule]:
        """Set several entries at once and persist. Returns the merged table.

        *mapping* values may be :class:`AffixRule`, ``{"text", "mode"}`` dicts,
        or bare spelling strings.
        """
        table = dict(cls.resolve())
        for key, value in mapping.items():
            label = table[key].label if key in table else ""
            table[key] = cls._coerce(key, value, label)
        cls._write(table)
        cls._cache = table
        return table

    @classmethod
    def reset(cls, key: Optional[str] = None) -> Dict[str, AffixRule]:
        """Drop the user's override for *key* (or every override when ``None``).

        Drops to the layer below, which is the studio doc when one is deployed
        and the shipped default otherwise — never to the default THROUGH a
        studio value the artist never set.
        """
        overrides = dict(cls._load_overrides() or {})
        if key is None:
            overrides.clear()
        else:
            overrides.pop(key, None)
        cls._save_overrides(overrides)
        return cls.reload()

    # -------------------------------------------------------------- internal
    @classmethod
    def config_path(cls):
        """Where the user's overrides are written."""
        return UserConfig.path_for(CONFIG_NAME, CONFIG_PACKAGE)

    @classmethod
    def _studio_overrides(cls) -> dict:
        """The shared doc named by ``$PYTHONTK_NAMING_CONVENTION`` (``{}`` when unset).

        Read-only, always: it is one file behind many artists, so a panel edit
        must never land here. It sits between the shipped defaults and the
        user's own doc in :meth:`resolve`.
        """
        import os

        raw = os.environ.get(CONFIG_ENV_VAR)
        if not raw:
            return {}
        doc = UserConfig.load_file(os.path.expanduser(os.path.expandvars(raw)))
        return doc if isinstance(doc, dict) else {}

    @classmethod
    def _load_overrides(cls) -> dict:
        """The user's OWN override doc (``{}`` when absent or unreadable).

        Deliberately NOT ``UserConfig.resolve``'s first-match-wins discovery.
        This is the doc :meth:`_save_overrides` writes, and read and write have
        to name the same file: when they disagreed, a studio deploying
        ``$PYTHONTK_NAMING_CONVENTION`` made every panel edit vanish on the next
        reload, and :meth:`reset` round-tripped the share's entries into the
        user's doc while dropping the personal keys that were already there.
        """
        path = cls.config_path()
        if not path.is_file():
            return {}
        doc = UserConfig.load_file(path)
        return doc if isinstance(doc, dict) else {}

    @classmethod
    def _baseline(cls) -> Dict[str, AffixRule]:
        """What a rule is compared against to decide it is worth persisting.

        The defaults with the studio doc overlaid — i.e. every layer below the
        user's own. Diffing against ``DEFAULTS`` alone would write a bogus
        personal override for every entry the studio share had already changed.
        """
        baseline = dict(cls.DEFAULTS)
        for key, value in (cls._studio_overrides() or {}).items():
            label = baseline[key].label if key in baseline else ""
            baseline[key] = cls._coerce(key, value, label)
        return baseline

    @classmethod
    def _write(cls, table: Dict[str, AffixRule]) -> None:
        """Persist only what differs from the layers below the user's doc.

        Storing the whole table would freeze today's defaults into every user's
        config: a later release renaming ``"_BS"`` to ``"_BLS"`` would reach
        nobody, because every user would be carrying an override they never made.
        The same argument applies one layer up, which is why the comparison is
        against :meth:`_baseline` and not ``DEFAULTS``.
        """
        baseline = cls._baseline()
        overrides = {}
        for key, rule in table.items():
            shipped = baseline.get(key)
            if shipped is not None and (rule.text, rule.mode) == (
                shipped.text,
                shipped.mode,
            ):
                continue
            overrides[key] = rule.as_dict()
        cls._save_overrides(overrides)

    @classmethod
    def _save_overrides(cls, overrides: dict) -> None:
        import json

        path = cls.config_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            from pythontk.file_utils._file_utils import FileUtils

            FileUtils.atomic_write_text(path, json.dumps(overrides, indent=4))
        except OSError as e:
            logger.warning("[naming_convention] could not write %s: %s", path, e)
        cls._cache = None
