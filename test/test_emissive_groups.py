#!/usr/bin/python
# coding=utf-8
"""The Emissive Groups model, host-free: an in-memory host stands in for a
host's scene (each host library tests its own half in an instance the tooling
launches).

A group is materials: its faces wear a copy of their material made for the
group, one per material and group, and the copy's emission is keyed by the
group's weight.
"""

import contextlib
import json
import unittest
from typing import Any, Dict, List, Optional

import pythontk as ptk
from pythontk.core_utils.engines.scene_export.scene_store import SceneStoreBase
from pythontk.core_utils.engines.scene_export.scene_records import Scope

from pythontk.core_utils.engines.textures.emissive_groups import (
    EmissiveGroupsHost,
    EmissiveGroupsModel,
    PureEmissiveGroupsHost,
)


def _bound(host):
    """The model over *host*, bound as a host library binds it: a subclass whose
    ``HOST`` is the host's half."""
    return type("EmissiveGroups", (EmissiveGroupsModel,), {"HOST": host})


class _Store(SceneStoreBase):
    """One dict per scope (pythontk's ``DictStore`` test double)."""

    data: Dict[Scope, Dict[str, Any]] = {}

    @classmethod
    def read(cls, scope, key):
        value = cls.data.get(Scope(scope), {}).get(key)
        return value if isinstance(value, str) and value else None

    @classmethod
    def write(cls, scope, key, text):
        scope = Scope(scope)
        if not text:
            group = cls.data.get(scope)
            if group is None or key not in group:
                return None
            del group[key]
            return cls.name(scope)
        cls.data.setdefault(scope, {})[key] = text
        return cls.name(scope)

    @classmethod
    def values(cls, scope):
        return dict(cls.data.get(Scope(scope), {}))


class _Host(PureEmissiveGroupsHost):
    """An in-memory scene: meshes whose faces each wear a material, materials
    with an emission scalar (``None``: the shader has none), a colour (``None``:
    mapped), other values and keys; the old tool's stores and weights."""

    data_nodes = _Store

    def __init__(self):
        self.faces_of: Dict[str, List[str]] = {
            "hull": ["metal"] * 6,
            "panel": ["glass"] * 4,
        }
        self.mats: Dict[str, Dict[str, Any]] = {
            "metal": {"emission": 2.0, "color": None, "values": {"rough": 0.4}, "keys": []},
            "glass": {"emission": 1.0, "color": [0.2, 0.6, 1.0], "values": {"rough": 0.1}, "keys": []},
            "lambert1": {"emission": None, "color": [0, 0, 0], "values": {}, "keys": []},
        }
        self.tags: Dict[str, Dict[str, Any]] = {}
        self.selection: Dict[str, List[int]] = {}
        self.selected: Optional[dict] = None
        self.proxies: Dict[str, Any] = {}
        self.legacy_stores: Dict[str, Dict[str, List[int]]] = {}
        self.legacy_weights: Dict[str, List] = {}
        self.legacy_color_sets: List[str] = []
        self.steps: List[str] = []
        self.frame = 1.0

    # faces and materials
    def faces(self, faces=None):
        source = self.selection if faces is None else dict(faces)
        return {
            o: sorted(i) or list(range(len(self.faces_of[o])))
            for o, i in source.items()
            if o in self.faces_of
        }

    def face_materials(self, face_map):
        out: Dict[str, Dict[str, List[int]]] = {}
        for obj, indices in face_map.items():
            for i in indices:
                out.setdefault(self.faces_of[obj][i], {}).setdefault(obj, []).append(i)
        return out

    def material_faces(self, material):
        out = {}
        for obj, mats in self.faces_of.items():
            hit = [i for i, m in enumerate(mats) if m == material]
            if hit:
                out[obj] = hit
        return out

    def material(self, name):
        return name if name in self.mats else None

    def assign(self, material, face_map):
        for obj, indices in face_map.items():
            for i in indices:
                self.faces_of[obj][i] = material

    def select(self, face_map):
        self.selected = face_map

    # copies
    def copies(self):
        return {
            name: {
                "group": tag["group"],
                "base": tag["base"] if tag["base"] in self.mats else None,
                "strength": tag["strength"],
            }
            for name, tag in self.tags.items()
            if name in self.mats
        }

    def make_copy(self, base, name):
        while name in self.mats:
            name += "1"
        src = self.mats[base]
        self.mats[name] = {
            "emission": src["emission"],
            "color": None if src["color"] is None else list(src["color"]),
            "values": dict(src["values"]),
            "keys": [],
        }
        return name

    def tag(self, copy, group, base, strength):
        self.tags[copy] = {"group": group, "base": base, "strength": strength}

    def delete_material(self, material):
        self.mats.pop(material, None)
        self.tags.pop(material, None)

    def material_state(self, material):
        mat = self.mats[material]
        return {"color": mat["color"], **mat["values"]}

    def sync_copy(self, copy, base):
        self.mats[copy]["values"] = dict(self.mats[base]["values"])
        self.mats[copy]["color"] = self.mats[base]["color"]

    # emission
    def emission(self, material):
        return self.mats[material]["emission"]

    def emission_color(self, material):
        return self.mats[material]["color"]

    def set_emission(self, material, value):
        self.mats[material]["emission"] = value

    def emission_keys(self, material):
        return list(self.mats[material]["keys"])

    def key_emission(self, material, value, frame=None):
        frame = self.frame if frame is None else frame
        keys = [k for k in self.mats[material]["keys"] if k[0] != frame]
        self.mats[material]["keys"] = sorted(keys + [(frame, value)])
        self.mats[material]["emission"] = value

    def clear_emission_keys(self, material):
        self.mats[material]["keys"] = []

    def copy_emission(self, src, dst, factor):
        self.mats[dst]["keys"] = [(f, v * factor) for f, v in self.mats[src]["keys"]]
        self.mats[dst]["emission"] = self.mats[src]["emission"] * factor

    def scale_emission(self, material, factor):
        self.copy_emission(material, material, factor)

    # export
    def stage_emission_proxy(self, material, name, factor):
        if name in self.proxies:
            return None
        self.proxies[name] = [(f, v * factor) for f, v in self.mats[material]["keys"]]
        return name

    def emission_proxies(self):
        return sorted(self.proxies)

    def remove_emission_proxies(self, names=None):
        removed = sorted(self.proxies if names is None else set(names) & set(self.proxies))
        for name in removed:
            del self.proxies[name]
        return removed

    @contextlib.contextmanager
    def undo_step(self, label):
        """One step per outermost call: a host's nested steps fold into it."""
        self.depth = getattr(self, "depth", 0) + 1
        if self.depth == 1:
            self.steps.append(label)
        try:
            yield
        finally:
            self.depth -= 1

    # the old tool's data
    def legacy_members(self, store):
        found = self.legacy_stores.get(store)
        return None if found is None else {o: list(i) for o, i in found.items()}

    def legacy_store_names(self, prefix):
        return [s[len(prefix):] for s in self.legacy_stores if s.startswith(prefix)]

    def adopt_legacy_weight(self, attr, copy, factor):
        keys = self.legacy_weights.get(attr)
        if not keys:
            return False
        self.mats[copy]["keys"] = [(f, v * factor) for f, v in keys]
        return True

    def remove_legacy(self, stores, attrs, color_set):
        removed = [s for s in stores if self.legacy_stores.pop(s, None) is not None]
        removed += [a for a in attrs if self.legacy_weights.pop(a, None) is not None]
        if color_set in self.legacy_color_sets:
            self.legacy_color_sets.remove(color_set)
            removed.append(color_set)
        return removed


class _ModelCase(unittest.TestCase):
    def setUp(self):
        _Store.data = {}
        self.host = _Host()
        self.model = _bound(self.host)

    def material_of(self, obj, index):
        return self.host.faces_of[obj][index]


class TestGroupsAreMaterials(_ModelCase):
    def test_a_group_gives_its_faces_a_copy_per_material(self):
        copies = self.model.add_group("signal", {"hull": [0, 1], "panel": [0]}, weight=0.5)
        self.assertEqual(sorted(copies), ["glass_signal", "metal_signal"])
        self.assertEqual(self.material_of("hull", 0), "metal_signal")
        self.assertEqual(self.material_of("hull", 2), "metal")
        self.assertEqual(self.material_of("panel", 0), "glass_signal")
        # The copy shares its base's values; its emission is the base's times the weight.
        self.assertEqual(self.host.mats["metal_signal"]["values"], {"rough": 0.4})
        self.assertAlmostEqual(self.host.emission("metal_signal"), 1.0)
        self.assertAlmostEqual(self.host.emission("glass_signal"), 0.5)
        self.assertEqual(
            self.host.copies()["metal_signal"],
            {"group": "signal", "base": "metal", "strength": 2.0},
        )
        self.assertEqual(self.host.steps, ["Emissive Groups: add signal"])

    def test_extending_reuses_the_copy_and_keeps_the_weight(self):
        self.model.add_group("signal", {"hull": [0]}, weight=0.25)
        self.model.add_group("signal", {"hull": [1]})
        self.assertEqual(self.material_of("hull", 1), "metal_signal")
        self.assertEqual(sorted(self.host.copies()), ["metal_signal"])
        self.assertAlmostEqual(self.host.emission("metal_signal"), 0.5)

    def test_faces_move_between_groups_and_an_emptied_copy_goes(self):
        self.model.add_group("a", {"hull": [0]})
        self.model.add_group("b", {"hull": [0]})
        self.assertEqual(self.material_of("hull", 0), "metal_b")
        self.assertNotIn("metal_a", self.host.mats, "a copy left with no face is deleted")
        self.assertEqual(self.host.copies()["metal_b"]["base"], "metal")

    def test_none_takes_the_selection_and_nothing_raises(self):
        with self.assertRaises(ValueError):
            self.model.add_group("x")
        self.host.selection = {"panel": [1]}
        self.model.add_group("x")
        self.assertEqual(self.material_of("panel", 1), "glass_x")

    def test_a_shader_without_an_emission_weight_is_refused_with_its_fix(self):
        self.host.assign("lambert1", {"panel": [3]})
        with self.assertRaises(ValueError) as raised:
            self.model.add_group("x", {"panel": [2, 3]})
        self.assertIn("lambert1", str(raised.exception))
        self.assertIn("emission", str(raised.exception))
        self.assertEqual(self.material_of("panel", 2), "glass", "refused before any change")

    def test_a_base_that_does_not_glow_is_refused(self):
        self.host.set_emission("glass", 0.0)
        with self.assertRaises(ValueError) as raised:
            self.model.add_group("x", {"panel": [0]})
        self.assertIn("glass", str(raised.exception))

    def test_removing_a_group_restores_its_faces_base_and_deletes_its_copies(self):
        self.model.add_group("signal", {"hull": [0], "panel": [0, 1]})
        self.model.remove_group("signal")
        self.assertEqual(self.host.faces_of["hull"][0], "metal")
        self.assertEqual(self.host.faces_of["panel"][:2], ["glass", "glass"])
        self.assertEqual(self.host.copies(), {})
        self.assertNotIn("glass_signal", self.host.mats)

    def test_removing_faces_restores_their_base(self):
        self.model.add_group("signal", {"hull": [0, 1]})
        self.model.remove_faces({"hull": [1]})
        self.assertEqual(self.host.faces_of["hull"][:2], ["metal_signal", "metal"])
        self.model.remove_faces({"hull": [0]})
        self.assertEqual(self.host.copies(), {}, "the emptied copy goes")

    def test_list_and_select(self):
        self.model.add_group("signal", {"hull": [0, 1], "panel": [0]}, weight=0.5)
        listed = self.model.list_groups()
        self.assertEqual(list(listed), ["signal"])
        self.assertEqual(listed["signal"]["faces"], 3)
        self.assertEqual(listed["signal"]["materials"], ["glass_signal", "metal_signal"])
        self.assertAlmostEqual(listed["signal"]["weight"], 0.5)
        self.assertFalse(listed["signal"]["keyed"])
        self.model.select_group("signal")
        self.assertEqual(self.host.selected, {"hull": [0, 1], "panel": [0]})

    def test_a_dry_run_lists_the_changes_and_makes_none(self):
        ctx = ptk.OpContext(dry_run=True)
        self.model.add_group("signal", {"hull": [0]}, ctx=ctx)
        self.assertEqual(self.host.copies(), {})
        self.assertEqual(self.material_of("hull", 0), "metal")
        self.assertIn(("create", "metal_signal"), [(c.kind, c.target) for c in ctx.changes])


class TestWeights(_ModelCase):
    def test_an_unkeyed_weight_is_set_and_a_keyed_one_left(self):
        self.model.add_group("signal", {"hull": [0], "panel": [0]})
        self.assertEqual(self.model.set_weight("signal", 2.0), 1.0, "clamped")
        self.model.set_weight("signal", 0.25)
        self.assertAlmostEqual(self.host.emission("metal_signal"), 0.5)
        self.model.key_weight("signal", 1.0, frame=10)
        self.model.set_weight("signal", 0.0)
        self.assertAlmostEqual(self.host.emission("metal_signal"), 2.0, msg="keys own it")

    def test_keying_keys_every_copy_by_its_strength(self):
        self.model.add_group("signal", {"hull": [0], "panel": [0]})
        self.model.key_weight("signal", 0.0, frame=1)
        self.model.key_weight("signal", 1.0, frame=24)
        self.assertEqual(self.host.emission_keys("metal_signal"), [(1, 0.0), (24, 2.0)])
        self.assertEqual(self.host.emission_keys("glass_signal"), [(1, 0.0), (24, 1.0)])
        self.assertTrue(self.model.list_groups()["signal"]["keyed"])

    def test_a_copy_joining_a_keyed_group_plays_its_curve(self):
        self.model.add_group("signal", {"hull": [0]})
        self.model.key_weight("signal", 0.0, frame=1)
        self.model.key_weight("signal", 1.0, frame=24)
        self.model.add_group("signal", {"panel": [0]})
        self.assertEqual(self.host.emission_keys("glass_signal"), [(1, 0.0), (24, 1.0)], "weight 1 at its own strength")

    def test_removing_keys_keeps_the_value(self):
        self.model.add_group("signal", {"hull": [0]})
        self.model.key_weight("signal", 0.5, frame=3)
        self.assertEqual(self.model.remove_keys("signal"), ["metal_signal"])
        self.assertEqual(self.host.emission_keys("metal_signal"), [])
        self.assertAlmostEqual(self.host.emission("metal_signal"), 1.0)

    def test_unknown_groups_are_refused(self):
        for call in (
            lambda: self.model.set_weight("nope", 1.0),
            lambda: self.model.key_weight("nope"),
            lambda: self.model.remove_group("nope"),
            lambda: self.model.select_group("nope"),
        ):
            with self.assertRaises(ValueError):
                call()


class TestInStep(_ModelCase):
    def test_a_changed_base_value_reaches_its_copies_on_sync(self):
        self.model.add_group("signal", {"hull": [0]})
        self.host.mats["metal"]["values"]["rough"] = 0.9
        found = self.model.sync(check_only=True)
        self.assertEqual(len(found), 1)
        self.assertIn("metal_signal", found[0])
        self.assertIn("rough", found[0])
        self.assertEqual(self.host.mats["metal_signal"]["values"]["rough"], 0.4, "a check changes nothing")
        self.model.sync()
        self.assertEqual(self.host.mats["metal_signal"]["values"]["rough"], 0.9)
        self.assertEqual(self.model.sync(check_only=True), [])

    def test_a_changed_base_strength_rescales_the_copy(self):
        self.model.add_group("signal", {"hull": [0]}, weight=0.5)
        self.model.key_weight("signal", 1.0, frame=5)
        self.host.set_emission("metal", 4.0)
        self.model.sync()
        self.assertEqual(self.host.emission_keys("metal_signal"), [(5, 4.0)])
        self.assertEqual(self.host.copies()["metal_signal"]["strength"], 4.0)

    def test_a_lost_base_is_reported_never_guessed(self):
        self.model.add_group("signal", {"hull": [0]})
        self.host.delete_material("metal")
        found = self.model.sync()
        self.assertTrue(any("base" in f and "metal_signal" in f for f in found), found)
        self.assertIn("metal_signal", self.host.mats)

    def test_validate_reports_drift_and_old_style_data(self):
        self.assertEqual(self.model.validate(), [])
        self.model.add_group("signal", {"hull": [0]})
        self.host.mats["metal"]["values"]["rough"] = 0.9
        _Store.data = {Scope.PRIVATE: {"emissive_registry": json.dumps({"groups": {"old": {"slot": 0}}})}}
        warnings = self.model.validate()
        self.assertTrue(any("rough" in w for w in warnings), warnings)
        self.assertTrue(any("convert_legacy" in w for w in warnings), warnings)


class TestExport(_ModelCase):
    def test_each_animated_copy_is_a_render_effects_material(self):
        self.model.add_group("signal", {"hull": [0], "panel": [0]})
        self.model.add_group("still", {"hull": [1]}, weight=0.5)
        self.model.key_weight("signal", 0.0, frame=1)
        self.model.key_weight("signal", 1.0, frame=24)
        entries = self.model.render_effect_materials()
        self.assertEqual(
            entries,
            [
                {"material": "glass_signal", "channels": ["emission"], "emission_color": [0.2, 0.6, 1.0], "emission_strength": 1.0},
                {"material": "metal_signal", "channels": ["emission"], "emission_color": [1.0, 1.0, 1.0], "emission_strength": 2.0},
            ],
        )
        record = ptk.MeshConvert.build_visibility_tracks([], entries)
        self.assertEqual(ptk.SceneRecords.problems(ptk.SceneRecords.VISIBILITY.key, record), [])

    def test_each_animated_copys_weight_curve_is_stated_as_its_proxy_carries_it(self):
        """Bug: a WebXR push with animation off played no emissive group --
        its copies' weights ride their ``<copy>__emission`` proxies, in takes
        such a file has none of. Fixed: 2026-11-07 -- ``weight_curves`` states
        each animated copy's weight curve in the scene's frames, as its proxy
        would carry it (the host's reader's keys, each segment's
        interpolation kept, over the copy's strength), for the push's
        overlay (``RenderEffects.effect_curves``)."""
        self.model.add_group("signal", {"hull": [0], "panel": [0]})
        self.model.add_group("still", {"hull": [1]}, weight=0.5)
        self.model.key_weight("signal", 0.0, frame=1)
        self.model.key_weight("signal", 1.0, frame=24)
        read = []

        def keys_of(copy):
            read.append(copy)
            return [[f, v, "linear"] for f, v in self.host.emission_keys(copy)]

        curves = self.model.weight_curves(keys_of)
        self.assertEqual(read, ["glass_signal", "metal_signal"], "the animated copies alone")
        self.assertEqual(
            curves,
            [
                {"target": "glass_signal", "channel": "emission", "keys": [[1.0, 0.0, "linear"], [24.0, 1.0, "linear"]]},
                {"target": "metal_signal", "channel": "emission", "keys": [[1.0, 0.0, "linear"], [24.0, 1.0, "linear"]]},
            ],
        )

    def _two_props(self):
        """A group on each of two props, each copy keyed."""
        self.model.add_group("signal", {"hull": [0], "panel": [0]})
        self.model.key_weight("signal", 0.0, frame=1)
        self.model.key_weight("signal", 1.0, frame=24)

    def test_a_scoped_write_names_and_stages_only_the_copies_it_ships(self):
        """A push of one prop carries the copies its faces wear and no other
        (the record's entries and the proxies alike: 2026-10-31, a push of one
        prop shipped every group in the scene)."""
        self._two_props()
        ctx = ptk.ExportContext(nodes=frozenset({"panel"}))
        self.assertEqual([e["material"] for e in self.model.render_effect_materials(ctx)], ["glass_signal"])
        self.assertEqual(self.model.stage_export_proxies(ctx), ["glass_signal__emission"])
        self.assertEqual(sorted(self.host.proxies), ["glass_signal__emission"])
        unscoped = ptk.ExportContext()
        self.assertEqual(len(self.model.render_effect_materials(unscoped)), 2, "no scope: every copy")

    def test_staging_made_before_the_scope_narrows_to_it(self):
        """A writer that stages before it publishes (the publish carries the
        scope): the staged proxies of copies the write does not ship go."""
        self._two_props()
        self.model.stage_export_proxies()
        self.assertEqual(sorted(self.host.proxies), ["glass_signal__emission", "metal_signal__emission"])
        removed = self.model.scope_export_proxies(ptk.ExportContext(nodes=frozenset({"hull"})))
        self.assertEqual(removed, ["glass_signal__emission"])
        self.assertEqual(sorted(self.host.proxies), ["metal_signal__emission"])
        self.assertEqual(self.model.scope_export_proxies(ptk.ExportContext()), [], "no scope: nothing goes")

    def test_a_base_whose_emission_colour_is_black_is_refused(self):
        """No emission at all: a black uniform colour glows at no weight."""
        self.host.mats["glass"]["color"] = [0.0, 0.0, 0.0]
        with self.assertRaises(ValueError) as raised:
            self.model.add_group("x", {"panel": [0]})
        self.assertIn("glass", str(raised.exception))
        self.assertIn("black", str(raised.exception))

    def test_the_curves_without_a_take_are_the_weight(self):
        """``weight_curves`` states the weight, 0..1, as the proxy does: the
        keys over the strength and over what the host's keyed plug is scaled
        by (a colour channel's reference)."""
        self.model.add_group("signal", {"hull": [0]})
        self.model.key_weight("signal", 1.0, frame=24)
        self.host.emission_scale = lambda material: 0.5
        keys_of = lambda copy: [[f, v * 0.5, "linear"] for f, v in self.host.emission_keys(copy)]
        (curve,) = self.model.weight_curves(keys_of)
        self.assertEqual(curve["target"], "metal_signal")
        self.assertEqual(curve["keys"], [[24.0, 1.0, "linear"]])

    def test_a_gate_shaders_copy_ships_its_base_glow_not_its_scalar(self):
        """A shader whose emission scalar only gates its emission (OpenPBR's
        luminance, in nits) is stated by the delivery at its family's
        published weight, 1: the ``materials`` record's emission weight
        (``EmissiveGroupsHost.published_strength``). So the copy's entry
        states that strength, weight 1 its base's glow, while the copy keeps
        the nits as what its weight scales -- the proxies and the curves
        stay the weight, 0..1. Measured before: the entry stated the
        luminance (1000) as the strength, a copy 1000 times its base."""
        self.host.mats["metal"]["emission"] = 1000.0
        self.host.published_strength = lambda material, strength: 1.0 if strength > 0 else 0.0
        self.model.add_group("signal", {"hull": [0], "panel": [0]})
        self.model.key_weight("signal", 0.0, frame=1)
        self.model.key_weight("signal", 1.0, frame=24)
        self.assertEqual(self.host.mats["metal_signal"]["emission"], 1000.0, "the copy keys its nits")
        strengths = {e["material"]: e["emission_strength"] for e in self.model.render_effect_materials()}
        self.assertEqual(strengths, {"metal_signal": 1.0, "glass_signal": 1.0})
        curves = self.model.weight_curves(
            lambda copy: [[f, v, "linear"] for f, v in self.host.emission_keys(copy)]
        )
        self.assertEqual(
            {c["target"]: c["keys"] for c in curves}["metal_signal"],
            [[1.0, 0.0, "linear"], [24.0, 1.0, "linear"]],
            "the weight, 0..1, never the nits",
        )
        self.model.stage_export_proxies()
        self.assertEqual(self.host.proxies["metal_signal__emission"], [(1, 0.0), (24, 1.0)])
        self.assertEqual(PureEmissiveGroupsHost().published_strength("m", 2.5), 2.5, "a weight that scales: itself")

    def test_a_copy_no_face_wears_plays_nowhere(self):
        """Its object deleted, a keyed copy is left with no face: it is no
        material of any file, so it is neither published nor staged (found
        2026-11-07: a deleted object's copy shipped a second proxy and a
        second take in a host's export)."""
        self.model.add_group("signal", {"hull": [0]})
        self.model.key_weight("signal", 1.0, frame=5)
        del self.host.faces_of["hull"]
        self.assertEqual(self.model.render_effect_materials(), [])
        self.assertEqual(self.model.stage_export_proxies(), [])

    def test_the_stager_stages_each_animated_copys_weight_on_its_proxy(self):
        self.model.add_group("signal", {"hull": [0]})
        self.model.key_weight("signal", 0.0, frame=1)
        self.model.key_weight("signal", 1.0, frame=24)
        self.model.add_group("still", {"hull": [1]})
        self.host.mats["metal"]["values"]["rough"] = 0.9
        from unittest import mock

        with mock.patch.object(self.model.logger, "warning") as warned:
            staged = self.model.stage_export_proxies()
        logged = [str(call.args[0]) for call in warned.call_args_list]
        self.assertEqual(staged, ["metal_signal__emission"])
        self.assertEqual(self.host.proxies["metal_signal__emission"], [(1, 0.0), (24, 1.0)], "the weight, 0..1")
        self.assertTrue(any("rough" in line for line in logged), "out of step: said, never swapped")
        self.assertEqual(self.host.mats["metal_signal"]["values"]["rough"], 0.4)
        self.assertEqual(self.model.remove_export_proxies(), ["metal_signal__emission"])


class TestConvertLegacy(_ModelCase):
    def _old_scene(self, encoding="vertex-color"):
        registry = {
            "schema": 1,
            "encoding": "channels" if encoding == "mask" else "vertex-color",
            "groups": {
                "headlights": {"slot": 0, "default": 1.0},
                "signal": {"slot": 1, "default": 0.5, "attr": "emissiveGroup_signal"},
            },
            "retired_slots": [],
        }
        if encoding == "mask":
            registry["mask"] = "scene_EMask.png"
        _Store.data = {
            Scope.PRIVATE: {"emissive_registry": json.dumps(registry)},
            Scope.DELIVERABLE: {"emissive_groups": json.dumps({"version": 2, "groups": []})},
        }
        self.host.legacy_stores = {
            "emissiveGroup_headlights": {"hull": [0, 1, 2]},
            "emissiveGroup_signal": {"hull": [2, 3], "panel": [0, 1]},
        }
        self.host.legacy_weights = {"emissiveGroup_signal": [(1, 0.0), (24, 1.0)]}
        self.host.legacy_color_sets = ["emissiveGroups"]

    def test_face_level_groups_convert_exactly_in_one_action(self):
        self._old_scene()
        report = self.model.convert_legacy()
        self.assertEqual(report["converted"], {"headlights": 3, "signal": 3})
        self.assertEqual(self.host.faces_of["hull"][:4], ["metal_headlights"] * 3 + ["metal_signal"])
        self.assertEqual(self.host.faces_of["panel"][:2], ["glass_signal"] * 2)
        self.assertAlmostEqual(self.host.emission("metal_headlights"), 2.0)
        self.assertEqual(self.host.emission_keys("metal_signal"), [(1, 0.0), (24, 2.0)])
        self.assertEqual(self.host.emission_keys("glass_signal"), [(1, 0.0), (24, 1.0)])
        self.assertEqual(self.host.steps, ["Emissive Groups: convert"], "one undo step")
        # The overlap is reported by name, never guessed.
        self.assertTrue(any("signal" in r and "headlights" in r and "hull" in r for r in report["reported"]), report)
        # The old data is gone, and a second run has nothing to do.
        self.assertEqual(self.host.legacy_stores, {})
        self.assertEqual(self.host.legacy_weights, {})
        self.assertEqual(self.host.legacy_color_sets, [])
        self.assertIsNone(_Store.read(Scope.PRIVATE, "emissive_registry"))
        self.assertIsNone(_Store.read(Scope.DELIVERABLE, "emissive_groups"))
        self.assertEqual(self.model.convert_legacy()["converted"], {})

    def test_mask_groups_are_reported_with_their_fix(self):
        self._old_scene("mask")
        report = self.model.convert_legacy()
        self.assertEqual(report["converted"], {"headlights": 3, "signal": 3})
        masked = [r for r in report["reported"] if "scene_EMask.png" in r]
        self.assertEqual(len(masked), 1, report)
        self.assertIn("headlights", masked[0])
        self.assertIn("cut", masked[0])

    def test_a_dry_run_converts_nothing(self):
        self._old_scene()
        ctx = ptk.OpContext(dry_run=True)
        self.model.convert_legacy(ctx=ctx)
        self.assertEqual(self.host.copies(), {})
        self.assertIn("emissiveGroup_signal", self.host.legacy_stores)
        self.assertTrue(ctx.changes)


class TestCrossings(_ModelCase):
    def test_a_crossing_restores_the_groups_on_the_far_side(self):
        self.model.add_group("signal", {"hull": [0]})
        self.model.key_weight("signal", 1.0, frame=4)
        payload = self.model.transfer_out(ptk.TransferContext())
        self.assertEqual(
            payload,
            {"copies": {"metal_signal": {"group": "signal", "base": "metal", "strength": 2.0, "keys": [[4, 2.0]]}}},
        )
        far = _Host()
        far.mats["metal_signal"] = dict(far.mats["metal"], keys=[])
        model = _bound(far)
        ctx = ptk.TransferContext()
        model.transfer_in(payload, ctx)
        self.assertEqual(far.copies()["metal_signal"]["group"], "signal")
        self.assertEqual(far.emission_keys("metal_signal"), [(4, 2.0)])


class TestPureHost(unittest.TestCase):
    def setUp(self):
        _Store.data = {}

    def test_an_unbound_model_reads_nothing_and_refuses_to_write(self):
        self.assertEqual(EmissiveGroupsModel.list_groups(), {})
        self.assertEqual(EmissiveGroupsModel.render_effect_materials(), [])
        with self.assertRaises(RuntimeError):
            EmissiveGroupsModel.add_group("x", {"cube": [0]})

    def test_the_pure_host_and_the_fake_satisfy_the_protocol(self):
        self.assertIsInstance(PureEmissiveGroupsHost(), EmissiveGroupsHost)
        self.assertIsInstance(_Host(), EmissiveGroupsHost)


class TestRetired(unittest.TestCase):
    def test_the_emissive_groups_record_is_retired_with_its_fix(self):
        self.assertFalse(hasattr(ptk.SceneRecords, "EMISSIVE_GROUPS"))
        self.assertIn("emissive_groups", ptk.SceneRecords.RETIRED)
        self.assertIn("convert", ptk.SceneRecords.RETIRED_FIXES["emissive_groups"])


if __name__ == "__main__":
    unittest.main()
