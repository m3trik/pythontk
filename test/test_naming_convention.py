# !/usr/bin/python
# coding=utf-8
"""Tests for pythontk.NamingConvention — the ecosystem's type-affix SSoT.

Covers the read surface every tool uses (:meth:`get` / :meth:`affix_parts` /
:meth:`apply`), the write surface the Naming panel edits through, and the two
properties the whole design rests on: that a rule is an *affix* (a prefix reads
as a prefix), and that persisting an override never freezes today's shipped
defaults into a user's config.

Run standalone: python -m test.test_naming_convention
"""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from pythontk.core_utils.naming_convention import (
    AffixRule,
    NamingConvention,
    CONFIG_ENV_VAR,
)
from pythontk.core_utils.user_config import CONFIG_ROOT_ENV_VAR


class _SandboxedConventionTest(unittest.TestCase):
    """Redirects the config root at a temp dir so no test touches the real doc."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self._prev_root = os.environ.get(CONFIG_ROOT_ENV_VAR)
        self._prev_env = os.environ.get(CONFIG_ENV_VAR)
        os.environ[CONFIG_ROOT_ENV_VAR] = self.tmp
        os.environ.pop(CONFIG_ENV_VAR, None)
        NamingConvention.reload()

    def tearDown(self):
        for var, prev in (
            (CONFIG_ROOT_ENV_VAR, self._prev_root),
            (CONFIG_ENV_VAR, self._prev_env),
        ):
            if prev is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = prev
        shutil.rmtree(self.tmp, ignore_errors=True)
        NamingConvention.reload()

    def _overrides(self) -> dict:
        path = NamingConvention.config_path()
        return json.loads(path.read_text()) if path.exists() else {}


class AffixRuleTest(unittest.TestCase):
    """The rule itself — an affix, not a suffix."""

    def test_leading_delimiter_reads_as_a_suffix(self):
        self.assertEqual(AffixRule("_GEO").parts(), ("", "_GEO"))
        self.assertEqual(AffixRule("_GEO").apply("body"), "body_GEO")

    def test_trailing_delimiter_reads_as_a_prefix(self):
        """The whole point of storing affixes: GEO_ must land on the front."""
        self.assertEqual(AffixRule("GEO_").parts(), ("GEO_", ""))
        self.assertEqual(AffixRule("GEO_").apply("body"), "GEO_body")

    def test_explicit_mode_overrides_the_delimiter(self):
        self.assertEqual(AffixRule("_GEO", "prefix").parts(), ("_GEO", ""))
        self.assertEqual(AffixRule("GEO_", "suffix").parts(), ("", "GEO_"))

    def test_bare_text_falls_back_to_suffix(self):
        """No delimiter to infer from — conventions here are suffixes."""
        self.assertEqual(AffixRule("GEO").parts(), ("", "GEO"))

    def test_empty_text_is_a_no_op(self):
        self.assertEqual(AffixRule("").parts(), ("", ""))
        self.assertEqual(AffixRule("").apply("body"), "body")

    def test_apply_is_idempotent(self):
        """Tools re-run over already-named scenes; doubling would be a bug."""
        rule = AffixRule("_GEO")
        self.assertEqual(rule.apply(rule.apply("body")), "body_GEO")

    def test_as_dict_drops_the_label(self):
        self.assertEqual(
            AffixRule("_GEO", "auto", "Mesh").as_dict(),
            {"text": "_GEO", "mode": "auto"},
        )


class ConventionReadTest(_SandboxedConventionTest):
    """The surface every consuming tool calls."""

    def test_shipped_defaults_resolve(self):
        self.assertEqual(NamingConvention.affix("mesh"), "_GEO")
        self.assertEqual(NamingConvention.affix("material"), "_MAT")
        self.assertEqual(NamingConvention.affix_parts("mesh"), ("", "_GEO"))
        self.assertEqual(NamingConvention.apply("body", "mesh"), "body_GEO")

    def test_labels_come_from_the_table(self):
        self.assertEqual(NamingConvention.label("nurbsCurve"), "Nurbs Curve")

    def test_unknown_key_is_an_inert_rule_not_an_error(self):
        """A tool naming a type this table has never heard of must not crash."""
        rule = NamingConvention.get("no_such_type")
        self.assertEqual(rule.text, "")
        self.assertEqual(NamingConvention.apply("body", "no_such_type"), "body")

    def test_keys_are_in_shipped_order(self):
        keys = NamingConvention.keys()
        self.assertEqual(keys[:4], ["group", "locator", "joint", "mesh"])
        self.assertEqual(len(keys), len(NamingConvention.DEFAULTS))

    def test_all_affixes_is_longest_first(self):
        """Callers strip a wrong affix before applying the right one: '_SG' is a
        tail of '_LSG', so testing the short one first would eat half the long."""
        affixes = NamingConvention.all_affixes()
        self.assertEqual(affixes, sorted(affixes, key=len, reverse=True))
        self.assertIn("_GEO", affixes)
        self.assertNotIn("", affixes)


class ConventionWriteTest(_SandboxedConventionTest):
    """The surface the Naming panel edits through."""

    def test_set_changes_what_every_reader_sees(self):
        NamingConvention.set("mesh", "_MSH")
        self.assertEqual(NamingConvention.affix("mesh"), "_MSH")
        self.assertEqual(NamingConvention.apply("body", "mesh"), "body_MSH")

    def test_override_survives_a_reload(self):
        NamingConvention.set("mesh", "_MSH")
        NamingConvention.reload()
        self.assertEqual(NamingConvention.affix("mesh"), "_MSH")

    def test_a_prefix_override_lands_on_the_front(self):
        NamingConvention.set("mesh", "GEO_")
        self.assertEqual(NamingConvention.apply("body", "mesh"), "GEO_body")

    def test_only_real_overrides_are_persisted(self):
        """Storing the whole table would freeze today's defaults into every
        user's config, so a later release that moves one would reach nobody."""
        NamingConvention.set("mesh", "_MSH")
        NamingConvention.set("material", "_MAT")  # identical to the default
        self.assertEqual(list(self._overrides()), ["mesh"])

    def test_update_writes_several_at_once(self):
        NamingConvention.update({"mesh": "_MSH", "material": {"text": "MTL_"}})
        self.assertEqual(NamingConvention.affix("mesh"), "_MSH")
        self.assertEqual(NamingConvention.affix_parts("material"), ("MTL_", ""))

    def test_reset_one_restores_the_default(self):
        NamingConvention.set("mesh", "_MSH")
        NamingConvention.reset("mesh")
        self.assertEqual(NamingConvention.affix("mesh"), "_GEO")
        self.assertEqual(self._overrides(), {})

    def test_reset_all_clears_every_override(self):
        NamingConvention.update({"mesh": "_MSH", "material": "MTL_"})
        NamingConvention.reset()
        self.assertEqual(NamingConvention.affix("mesh"), "_GEO")
        self.assertEqual(NamingConvention.affix("material"), "_MAT")
        self.assertEqual(self._overrides(), {})

    def test_an_empty_affix_disables_the_type(self):
        """A user who wants meshes left alone clears the field, not the tool."""
        NamingConvention.set("mesh", "")
        self.assertEqual(NamingConvention.apply("body", "mesh"), "body")
        self.assertNotIn("", NamingConvention.all_affixes())

    def test_a_new_key_can_be_added(self):
        NamingConvention.set("gpencil", "_GP", label="Grease Pencil")
        NamingConvention.reload()
        self.assertEqual(NamingConvention.affix("gpencil"), "_GP")
        self.assertIn("gpencil", NamingConvention.keys())

    def test_labels_survive_an_override(self):
        """Editing a suffix must not cost the panel its row title."""
        NamingConvention.set("nurbsCurve", "_CURVE")
        self.assertEqual(NamingConvention.label("nurbsCurve"), "Nurbs Curve")

    def test_unknown_mode_degrades_to_auto(self):
        NamingConvention.set("mesh", "_MSH", mode="sideways")
        self.assertEqual(NamingConvention.mode("mesh"), "auto")


class ConventionBindTest(_SandboxedConventionTest):
    """``bind`` — the join every type-driven rename runs.

    Lives here rather than in each toolkit: only the BINDINGS table is
    host-specific, and mayatk/blendertk carried two verbatim copies of the join
    before this existed.
    """

    BINDINGS = (
        ("mesh_suffix", "mesh", "MESH"),
        ("material_suffix", "material", "material"),
    )

    def test_keys_by_host_type_not_convention_key(self):
        rules = NamingConvention.bind(self.BINDINGS)
        self.assertEqual(set(rules), {"MESH", "material"})
        self.assertEqual(rules["MESH"].text, "_GEO")

    def test_labels_come_from_the_convention(self):
        """A panel rendering the result still has its row titles."""
        self.assertEqual(NamingConvention.bind(self.BINDINGS)["MESH"].label, "Mesh")

    def test_follows_a_convention_edit(self):
        NamingConvention.set("mesh", "_MSH")
        self.assertEqual(NamingConvention.bind(self.BINDINGS)["MESH"].text, "_MSH")

    def test_override_by_keyword_or_by_host_key(self):
        by_kw = NamingConvention.bind(self.BINDINGS, {"mesh_suffix": "_A"})
        by_host = NamingConvention.bind(self.BINDINGS, {"MESH": "_A"})
        self.assertEqual((by_kw["MESH"].text, by_host["MESH"].text), ("_A", "_A"))

    def test_keyword_wins_over_host_key(self):
        """A caller forwarding its own kwargs is the common case."""
        rules = NamingConvention.bind(
            self.BINDINGS, {"mesh_suffix": "_KW", "MESH": "_HOST"}
        )
        self.assertEqual(rules["MESH"].text, "_KW")

    def test_mode_override_changes_placement(self):
        rules = NamingConvention.bind(
            self.BINDINGS, {"mesh_suffix": "GEO_"}, {"mesh_suffix": "prefix"}
        )
        self.assertEqual(rules["MESH"].apply("body"), "GEO_body")

    def test_empty_override_disables_the_entry(self):
        rules = NamingConvention.bind(self.BINDINGS, {"mesh_suffix": ""})
        self.assertEqual(rules["MESH"].apply("body"), "body")

    def test_a_binding_naming_an_unknown_entry_is_inert_not_fatal(self):
        rules = NamingConvention.bind((("x_suffix", "no_such_key", "X"),))
        self.assertEqual(rules["X"].apply("body"), "body")


class ConventionEnvOverrideTest(_SandboxedConventionTest):
    """A project or studio share can point the whole toolset at one doc."""

    def test_env_var_doc_is_honoured(self):
        doc = Path(self.tmp) / "studio_convention.json"
        doc.write_text(json.dumps({"mesh": {"text": "GEO_", "mode": "prefix"}}))
        os.environ[CONFIG_ENV_VAR] = str(doc)
        NamingConvention.reload()
        self.assertEqual(NamingConvention.apply("body", "mesh"), "GEO_body")

    def _deploy_studio_doc(self, entries: dict):
        """Stand up a shared doc and point the env var at it."""
        doc = Path(self.tmp) / "studio_convention.json"
        doc.write_text(json.dumps(entries))
        os.environ[CONFIG_ENV_VAR] = str(doc)
        NamingConvention.reload()
        return doc

    def test_a_panel_edit_survives_a_reload_under_a_studio_doc(self):
        """The read and write paths have to name the same file.

        They did not: the reader preferred the env-pointed share while every
        write went to the per-user doc, so an artist's edit applied for the
        rest of the session and was silently gone at the next launch.
        """
        self._deploy_studio_doc({"mesh": {"text": "_GEO_STUDIO", "mode": "suffix"}})
        self.assertEqual(NamingConvention.affix("mesh"), "_GEO_STUDIO")

        NamingConvention.set("mesh", "_MSH", "suffix")
        NamingConvention.reload()
        self.assertEqual(
            NamingConvention.affix("mesh"),
            "_MSH",
            "the artist's override was discarded on reload",
        )

    def test_reset_under_a_studio_doc_drops_to_the_share(self):
        """reset() has to read the file it writes, or it is a no-op."""
        self._deploy_studio_doc({"material": {"text": "MTL_", "mode": "prefix"}})
        NamingConvention.set("material", "_MAT2", "suffix")
        self.assertEqual(NamingConvention.affix("material"), "_MAT2")

        NamingConvention.reset("material")
        NamingConvention.reload()
        self.assertEqual(
            NamingConvention.affix("material"),
            "MTL_",
            "reset did not restore the studio value",
        )

    def test_reset_leaves_unrelated_personal_keys_alone(self):
        """reset() used to round-trip the SHARE into the personal doc.

        It loaded the studio entries as if they were the user's, popped one,
        and wrote the rest back over the personal doc -- destroying every
        personal override that was in it.
        """
        self._deploy_studio_doc({"material": {"text": "MTL_", "mode": "prefix"}})
        NamingConvention.set("joint", "_MY_JNT", "suffix")
        NamingConvention.set("material", "_MAT2", "suffix")

        NamingConvention.reset("material")
        NamingConvention.reload()
        self.assertEqual(
            NamingConvention.affix("joint"),
            "_MY_JNT",
            "an unrelated personal override was wiped by reset()",
        )

    def test_the_personal_doc_never_absorbs_the_share(self):
        """A value equal to the share is not a personal override."""
        self._deploy_studio_doc(
            {
                "mesh": {"text": "_GEO_STUDIO", "mode": "suffix"},
                "material": {"text": "MTL_", "mode": "prefix"},
            }
        )
        NamingConvention.set("joint", "_MY_JNT", "suffix")
        self.assertEqual(
            sorted(self._overrides()),
            ["joint"],
            "the studio doc was copied into the user's own file",
        )

    def test_a_studio_doc_pointing_nowhere_is_inert(self):
        """A share on an unmounted drive must not break the toolset."""
        os.environ[CONFIG_ENV_VAR] = str(Path(self.tmp) / "does_not_exist.json")
        NamingConvention.reload()
        self.assertEqual(NamingConvention.affix("mesh"), "_GEO")
        NamingConvention.set("mesh", "_MSH", "suffix")
        NamingConvention.reload()
        self.assertEqual(NamingConvention.affix("mesh"), "_MSH")


if __name__ == "__main__":
    unittest.main(verbosity=2)
