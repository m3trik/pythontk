# !/usr/bin/python
# coding=utf-8
"""Tests for :class:`pythontk.HandoffScope` -- the scope words and their precedence."""

import unittest

import pythontk as ptk
from pythontk.core_utils.handoff.handoff_scope import HandoffScope


class _Calls:
    """Lookups that record what was asked, returning canned answers."""

    def __init__(self, **answers):
        self.answers = answers
        self.asked = []

    def __getattr__(self, name):
        if name.startswith("_") or name not in self.answers:
            raise AttributeError(name)

        def lookup():
            self.asked.append(name)
            return self.answers[name]

        return lookup


class TestWords(unittest.TestCase):
    def test_root_exposes_the_class(self):
        self.assertIs(ptk.HandoffScope, HandoffScope)

    def test_vocabulary_is_the_bridge_panels_own(self):
        """The order is the combo's: selected leads (default), then all, visible."""
        self.assertEqual(HandoffScope.PARAM, "SCOPE")
        self.assertEqual(HandoffScope.WORDS, ("selected", "all", "visible"))

    def test_known_words_pass_through(self):
        for word in HandoffScope.WORDS:
            self.assertEqual(HandoffScope.word(word), word)

    def test_scene_is_a_synonym_for_all(self):
        self.assertEqual(HandoffScope.word("scene"), HandoffScope.ALL)

    def test_unknown_values_never_widen(self):
        for value in ("bogus", "", None, 3, ["all"], "ALL", " all"):
            self.assertEqual(HandoffScope.word(value), HandoffScope.SELECTED, value)


class TestResolve(unittest.TestCase):
    def test_selected_reads_only_the_selection(self):
        calls = _Calls(sel=["a"], scene=["a", "b"], vis=["b"])
        got = HandoffScope.resolve(
            "selected", selected=calls.sel, all=calls.scene, visible=calls.vis
        )
        self.assertEqual(got, ["a"])
        self.assertEqual(calls.asked, ["sel"])

    def test_widening_words_use_their_lookup(self):
        calls = _Calls(sel=["a"], scene=["a", "b"], vis=["b"])
        kw = dict(selected=calls.sel, all=calls.scene, visible=calls.vis)
        self.assertEqual(HandoffScope.resolve("all", **kw), ["a", "b"])
        self.assertEqual(HandoffScope.resolve("scene", **kw), ["a", "b"])
        self.assertEqual(HandoffScope.resolve("visible", **kw), ["b"])
        self.assertNotIn("sel", calls.asked)

    def test_an_empty_answer_is_an_answer(self):
        """``[]`` means "nothing there" -- it must not fall back to the selection."""
        calls = _Calls(sel=["a"], scene=[])
        self.assertEqual(
            HandoffScope.resolve("all", selected=calls.sel, all=calls.scene), []
        )
        self.assertEqual(calls.asked, ["scene"])

    def test_no_answer_falls_back_to_the_selection(self):
        """``None`` (the host cannot enumerate) and a missing lookup both narrow."""
        calls = _Calls(sel=["a"], scene=None)
        self.assertEqual(
            HandoffScope.resolve("all", selected=calls.sel, all=calls.scene), ["a"]
        )
        self.assertEqual(HandoffScope.resolve("visible", selected=calls.sel), ["a"])

    def test_a_chain_is_tried_in_order(self):
        calls = _Calls(sel=["a"], hook=None, meshes=["m"])
        got = HandoffScope.resolve(
            "all", selected=calls.sel, all=(calls.hook, calls.meshes)
        )
        self.assertEqual(got, ["m"])
        self.assertEqual(calls.asked, ["hook", "meshes"])

    def test_the_first_answer_wins(self):
        calls = _Calls(sel=["a"], hook=["root"], meshes=["m"])
        got = HandoffScope.resolve(
            "all", selected=calls.sel, all=[calls.hook, calls.meshes]
        )
        self.assertEqual(got, ["root"])
        self.assertEqual(calls.asked, ["hook"])

    def test_unknown_scope_resolves_as_selected(self):
        calls = _Calls(sel=["a"], scene=["a", "b"])
        self.assertEqual(
            HandoffScope.resolve("bogus", selected=calls.sel, all=calls.scene), ["a"]
        )
        self.assertEqual(calls.asked, ["sel"])

    def test_selection_answer_passes_through_unchanged(self):
        """``None`` from *selected* is the hand-off skeleton's "the selection"."""
        self.assertIsNone(HandoffScope.resolve("selected", selected=lambda: None))

    def test_a_misspelt_lookup_raises(self):
        with self.assertRaises(TypeError):
            HandoffScope.resolve("all", selected=list, visibel=list)
        with self.assertRaises(TypeError):
            HandoffScope.resolve("all", selected=list, selected_too=list)


if __name__ == "__main__":
    unittest.main()
