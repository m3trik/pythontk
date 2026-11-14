# !/usr/bin/python
# coding=utf-8
"""Tests for pythontk.geo_utils.uv.transfer_run (TransferRun -- a Transfer run).

The Texture Combine panel's Transfer mode -- its decisions (the passes a
Source / Transfer pair runs, Auto's pick, the refusals) and the run over the
host's operations -- written once for both hosts. A fake host stands in for
mayatk's / blendertk's adapter and records what the run asks of it; each
library's ``test_uv_texture_transfer.py`` pins its own adapter, and tentacle's
``test_texture_combine_panel.py`` how the panel words a result.
"""

import contextlib
import types
import unittest
from unittest import mock

import pythontk as ptk

TransferRun = ptk.TransferRun


class _Engine:
    """A texture-transfer engine double: records its calls, answers *maps*
    (or raises *error*), and logs *warnings* while running."""

    def __init__(self, maps=None, error=None, warnings=()):
        self.calls = []
        self.maps = (
            {"m": {"baseColor": "/out/hero_BaseColor.png"}} if maps is None else maps
        )
        self.error = error
        self.warnings = list(warnings)
        self.buffer = None
        self.logger = types.SimpleNamespace(
            enable_log_buffer=self._enable, disable_log_buffer=self._disable
        )

    def _enable(self, level=None):
        self.buffer = types.SimpleNamespace(buffer=[])
        return self.buffer

    def _disable(self):
        self.buffer = None

    def transfer(self, targets, source=None, **options):
        self.calls.append((list(targets or []), source, options))
        for text in self.warnings:
            if self.buffer is not None:
                self.buffer.buffer.append(
                    types.SimpleNamespace(getMessage=lambda t=text: t)
                )
        if self.error:
            raise ValueError(self.error)
        return self.maps


class _Host:
    """A :class:`pythontk.TransferHost` double over plain names."""

    def __init__(self, sides=("first", ["src"], ["tgt"]), **answers):
        self.sides = sides
        self.calls = []
        self.undo = []
        self.engine = answers.pop("engine", _Engine())
        self.pairs = answers.pop("pairs", None)
        self.uvs = answers.pop("uvs", None)
        self.similar = answers.pop("similar", [])
        self.materials = answers.pop("materials", {})
        self.lightmapped = answers.pop("lightmapped", set())
        self.lightmaps = answers.pop("lightmap_results", {})
        self.errors = answers.pop("errors", {})

    def _raise(self, name):
        if name in self.errors:
            raise ValueError(self.errors[name])

    def selection_sides(self, mode, stored):
        self.calls.append(("selection_sides", mode, stored))
        return self.sides

    def face_materials(self, mesh):
        return list(self.materials.get(mesh, {})), None

    def material_maps(self, material):
        for maps in self.materials.values():
            if material in maps:
                return dict(maps[material])
        return {}

    def pair_sources(self, targets, sources):
        self.calls.append(("pair_sources", list(targets), list(sources)))
        self._raise("pair_sources")
        if self.pairs is not None:
            return dict(self.pairs)
        return dict(zip(targets, sources))

    def default_name(self, nodes):
        self.calls.append(("default_name", list(nodes)))
        return "crate"

    def transfer_uvs(self, sources, targets):
        self.calls.append(("transfer_uvs", list(sources), list(targets)))
        self._raise("transfer_uvs")
        if self.uvs is not None:
            return list(self.uvs)
        return [(s, t, "topology") for s, t in zip(sources, targets)]

    def transfer_uvs_to_similar(self, source, candidates, tolerance):
        self.calls.append(("transfer_uvs_to_similar", source, candidates, tolerance))
        return list(self.similar)

    def texture_transfer(self):
        return self.engine

    def has_lightmap(self, mesh):
        return mesh in self.lightmapped

    def transfer_lightmaps(self, targets, source, **options):
        self.calls.append(("transfer_lightmaps", list(targets), list(source), options))
        self._raise("transfer_lightmaps")
        return dict(self.lightmaps)

    @contextlib.contextmanager
    def undo_chunk(self, name):
        self.undo.append(name)
        yield

    def called(self, name):
        return [c for c in self.calls if c[0] == name]


def _run(host, **options):
    return TransferRun(host).run(**options)


class TestDecisions(unittest.TestCase):
    def test_the_passes(self):
        passes = TransferRun.passes
        self.assertEqual(passes("first", "auto"), (True, True))
        self.assertEqual(passes("uvset", "auto"), (False, True))
        self.assertEqual(passes("first", "textures"), (False, True))
        self.assertEqual(passes("first", "uvs"), (True, False))
        self.assertEqual(passes("uvset", "uvs"), (False, False))
        self.assertEqual(passes("first", None), (True, False))

    def test_auto_picks_textures_when_a_source_material_has_maps(self):
        host = _Host(materials={"src": {"m": {"baseColor": "/a.png"}}})
        run = TransferRun(host)
        self.assertEqual(run.resolve_auto(["src"]), "textures")
        host.materials = {"src": {"m": {}}}
        self.assertEqual(run.resolve_auto(["src"]), "uvs")
        self.assertEqual(
            run.resolve_auto(["src"], has_lightmap=lambda m: True), "textures"
        )

    def test_an_unreadable_mesh_counts_as_no_map(self):
        host = _Host()
        host.face_materials = mock.Mock(side_effect=RuntimeError("gone"))
        broken = mock.Mock(side_effect=RuntimeError("marker"))
        self.assertEqual(TransferRun(host).resolve_auto(["src"], broken), "uvs")

    def test_every_refusal_is_a_named_one(self):
        host = _Host(sides=("first", [], []))
        for source in ("first", "last", "stored", "uvset"):
            host.sides = (source, [], [])
            refusal = _run(host, source=source, transfer="textures").refusal
            self.assertIn(refusal, TransferRun.REFUSALS, source)

    def test_a_refusal_opens_no_undo_step(self):
        """Decided before the undo chunk opens: Blender pushes a step on its
        exit whatever ran, so a refused run's Ctrl+Z undid nothing visible."""
        host = _Host(sides=("first", [], []))
        self.assertEqual(
            _run(host, source="first", transfer="uvs").refusal, "pick_first"
        )
        self.assertEqual(host.undo, [])

    def test_with_no_dcc_the_pure_host_runs_and_refuses(self):
        self.assertIsInstance(ptk.PureTransferHost(), ptk.TransferHost)
        self.assertIsInstance(_Host(), ptk.TransferHost)
        result = TransferRun().run()
        self.assertEqual((result.mode, result.refusal), ("first", "pick_first"))
        result = TransferRun().run(source="uvset", transfer="textures")
        self.assertEqual(result.refusal, "nothing_selected")

    def test_an_engine_the_host_cannot_give_is_the_texture_passs_error(self):
        host = _Host()
        host.texture_transfer = ptk.PureTransferHost().texture_transfer
        result = _run(host, source="first", transfer="textures", assign_from="in_place")
        self.assertEqual(result.texture_error, "No scene to transfer textures in.")
        self.assertIsNone(result.maps)


class TestTheRun(unittest.TestCase):
    def test_the_run_is_one_undo_step_over_the_hosts_sides(self):
        host = _Host()
        result = _run(host, source="first", transfer="uvs", stored=("|a",))
        self.assertEqual(host.undo, ["Texture Transfer"])
        self.assertEqual(
            host.called("selection_sides"), [("selection_sides", "first", ["|a"])]
        )
        self.assertEqual(
            host.called("pair_sources"), [("pair_sources", ["tgt"], ["src"])]
        )
        self.assertIsNone(result.refusal)

    def test_a_pair_transfers_uvs_without_re_vetting(self):
        host = _Host(pairs={"tgt": "src"})
        result = _run(host, source="first", transfer="uvs")
        self.assertEqual(
            host.called("transfer_uvs"), [("transfer_uvs", ["src"], ["tgt"])]
        )
        self.assertEqual(result.uvs, [("src", "tgt", "topology")])
        self.assertEqual((result.do_uvs, result.do_textures), (True, False))
        self.assertIsNone(result.maps)

    def test_no_selection_refuses_with_the_mode_it_read(self):
        for mode, refusal in (("first", "pick_first"), ("last", "pick_last")):
            host = _Host(sides=(mode, [], []))
            result = _run(host, source=mode, transfer="uvs")
            self.assertEqual(result.refusal, refusal)
            self.assertEqual(host.called("transfer_uvs"), [])
        host = _Host(sides=("stored", [], ["tgt"]))
        self.assertEqual(_run(host, source="stored").refusal, "no_stored")
        host = _Host(sides=("stored", ["src"], []))
        self.assertEqual(_run(host, source="stored").refusal, "no_targets")

    def test_the_same_mesh_source_needs_a_texture_pass_and_a_mesh(self):
        host = _Host(sides=("uvset", None, ["m"]))
        self.assertEqual(
            _run(host, source="uvset", transfer="uvs").refusal, "nothing_to_transfer"
        )
        host = _Host(sides=("uvset", None, []))
        self.assertEqual(
            _run(host, source="uvset", transfer="textures").refusal, "nothing_selected"
        )
        host = _Host(sides=("uvset", None, ["m"]))
        result = _run(host, source="uvset", transfer="textures")
        self.assertIsNone(result.refusal)
        self.assertEqual(host.engine.calls[0][:2], (["m"], None))

    def test_the_uv_pass_refuses_a_combined_target(self):
        host = _Host(sides=("last", ["a", "b"], ["c"]), pairs={"c": ("a", "b")})
        result = _run(host, source="last", transfer="uvs")
        self.assertEqual(result.refusal, "one_source_per_target")
        self.assertEqual(host.called("transfer_uvs"), [])
        # Its textures can travel.
        result = _run(host, source="last", transfer="textures")
        self.assertIsNone(result.refusal)
        self.assertEqual(host.engine.calls[0][:2], (["c"], ["a", "b"]))

    def test_the_hosts_refusals_carry_its_message(self):
        host = _Host(errors={"pair_sources": "cannot pair 2 with 3"})
        result = _run(host, source="first", transfer="uvs")
        self.assertEqual(
            (result.refusal, result.error), ("pair_error", "cannot pair 2 with 3")
        )
        host = _Host(errors={"transfer_uvs": "no mesh"})
        result = _run(host, source="first", transfer="uvs")
        self.assertEqual((result.refusal, result.error), ("uv_error", "no mesh"))

    def test_the_similar_scopes_fan_out_through_the_host(self):
        host = _Host(sides=("first", ["src"], ["t1", "t2"]), similar=["t1"])
        result = _run(
            host, source="first", transfer="uvs", scope="selection", similarity=0.8
        )
        self.assertEqual(
            host.called("transfer_uvs_to_similar"),
            [("transfer_uvs_to_similar", "src", ["t1", "t2"], 0.8)],
        )
        self.assertEqual(
            (result.targets, result.uvs), (["t1"], [("src", "t1", "topology")])
        )
        host = _Host(sides=("first", ["src"], []), similar=["far"])
        _run(host, source="first", transfer="uvs", scope="scene")
        self.assertIsNone(host.called("transfer_uvs_to_similar")[0][2])
        host = _Host(sides=("first", ["src"], ["t1"]), similar=[])
        self.assertEqual(
            _run(host, source="first", transfer="uvs", scope="scene").refusal,
            "no_similar",
        )

    def test_the_similar_scopes_need_the_uv_pass(self):
        host = _Host(sides=("first", ["src"], ["t1"]))
        result = _run(host, source="first", transfer="textures", scope="scene")
        self.assertEqual(result.refusal, "similar_needs_uvs")
        host = _Host(sides=("first", [], ["t1"]))
        result = _run(host, source="first", transfer="uvs", scope="selection")
        self.assertEqual(result.refusal, "pick_first")

    def test_auto_reads_the_selection_and_rest_of_selection(self):
        host = _Host(sides=("last", ["a", "b"], ["c"]), pairs={"c": ("a", "b")})
        result = _run(host, source="auto", transfer="textures", scope="scene")
        self.assertEqual(host.called("selection_sides")[0][1], "auto")
        self.assertEqual(
            (result.source_auto, result.mode, result.scope), (True, "last", "order")
        )

    def test_auto_transfer_resolves_off_the_source(self):
        host = _Host(materials={"src": {"m": {"baseColor": "/a.png"}}})
        result = _run(host, source="first", transfer="auto")
        self.assertEqual((result.transfer_auto, result.do_textures), (True, True))
        host = _Host(lightmapped={"src"})
        result = _run(host, source="first", transfer="auto", lightmaps=True)
        self.assertEqual(
            (result.do_uvs, result.do_textures, result.lightmaps), (False, True, True)
        )
        result = _run(_Host(), source="first", transfer="auto")
        self.assertEqual((result.do_uvs, result.do_textures), (True, False))

    def test_each_pass_reports_its_start(self):
        stages = []
        _run(_Host(), source="first", transfer="uvs", progress=stages.append)
        _run(_Host(), source="first", transfer="textures", progress=stages.append)
        self.assertEqual(stages, ["uvs", "textures"])


class TestTexturePass(unittest.TestCase):
    def _call(self, host=None, **options):
        host = host or _Host()
        options = {
            "source": "first",
            "transfer": "textures",
            "output_name": "hero",
            **options,
        }
        result = _run(host, **options)
        self.assertEqual(len(host.engine.calls), 1, result)
        return host, result, host.engine.calls[0]

    def test_the_rows_reach_the_hosts_transfer(self):
        host, result, (targets, source, kw) = self._call(
            supersample=3,
            padding=4,
            output_dir="",
            size=0,
            source_uv_set="",
            target_uv_set="lm",
        )
        self.assertEqual((targets, source), (["tgt"], ["src"]))
        self.assertEqual(
            (kw["output_name"], kw["supersample"], kw["padding"], kw["size"]),
            ("hero", 3, 4, None),
        )
        self.assertEqual(
            (kw["output_dir"], kw["source_uv_set"], kw["target_uv_set"]),
            (None, None, "lm"),
        )
        self.assertEqual((kw["assign_prefix"], kw["assign_suffix"]), ("", None))
        self.assertEqual((kw["assign"], kw["assign_from"]), (True, "source"))
        self.assertNotIn("assign_shader_type", kw)  # Blender's transfer has none
        self.assertEqual(
            (result.typed, result.output_name, result.map_count), (True, "hero", 1)
        )
        self.assertEqual(result.maps_folder, "/out")

    def test_a_shader_type_reaches_the_transfer(self):
        kw = self._call(assign_shader_type="stingray")[2][2]
        self.assertEqual(
            (kw["assign_shader_type"], kw["assign_from"]), ("stingray", "source")
        )

    def test_a_blank_name_is_the_hosts_default_name_for_the_source(self):
        host, result, call = self._call(output_name="  ")
        self.assertEqual(host.called("default_name"), [("default_name", ["src"])])
        self.assertEqual((call[2]["output_name"], result.typed), ("crate", False))
        host = _Host(sides=("uvset", None, ["mesh"]))
        self._call(host, source="uvset", output_name="")
        self.assertEqual(host.called("default_name"), [("default_name", ["mesh"])])

    def test_one_material_one_stem_names_the_outputs(self):
        engine = _Engine(maps={"m": {"baseColor": "/o/held_BaseColor.png"}})
        result = self._call(_Host(engine=engine))[1]
        self.assertEqual((result.output_name, result.named), ("hero", "held"))

    def test_a_refused_texture_pass_says_why(self):
        result = self._call(_Host(engine=_Engine(error="no maps")))[1]
        self.assertEqual((result.maps, result.texture_error), (None, "no maps"))
        self.assertFalse(result.produced)

    def test_overwrite_target_reports_what_it_rewrote_and_left(self):
        engine = _Engine(
            maps={"m": {"baseColor": "/maps/own.png", "normal": "/maps/new.png"}},
            warnings=["kept x"],
        )
        host = _Host(
            engine=engine, materials={"tgt": {"m": {"baseColor": "/maps/own.png"}}}
        )
        result = self._call(host, assign_from="in_place")[1]
        self.assertTrue(result.in_place)
        self.assertEqual(
            (result.overwritten, result.added), (["/maps/own.png"], ["/maps/new.png"])
        )
        self.assertEqual(result.warnings, ["kept x"])
        self.assertIsNone(engine.buffer)  # the ring buffer is dropped
        self.assertEqual(result.named, "hero")  # a rewritten map keeps its name
        # Assign off: nothing is re-used in place.
        result = self._call(
            _Host(engine=_Engine()), assign_from="in_place", assign=False
        )[1]
        self.assertFalse(result.in_place)

    def test_the_lightmap_pass_rides_the_texture_pass_and_finds_its_home(self):
        host = _Host(
            lightmap_results={
                "tgt": {"how": "resampled", "path": "/session/hero_Lightmap.exr"}
            }
        )
        session = mock.MagicMock()
        session.home.return_value = "/proj/lightmaps/hero_Lightmap.exr"
        with mock.patch.object(ptk.PendingFiles, "session", return_value=session):
            result = self._call(host, lightmaps=True, supersample=3, padding=2)[1]
        session.home.assert_called_once_with("/session/hero_Lightmap.exr")
        call = host.called("transfer_lightmaps")[0]
        self.assertEqual(call[1:3], (["tgt"], ["src"]))
        self.assertEqual(
            call[3],
            {"output_dir": None, "output_name": "hero", "supersample": 3, "padding": 2},
        )
        self.assertEqual(result.resampled, ["/session/hero_Lightmap.exr"])
        self.assertEqual(result.lightmap_folder, "/proj/lightmaps")
        self.assertTrue(result.produced)

    def test_the_lightmap_pass_needs_mesh_sources_and_can_refuse(self):
        host = _Host(sides=("uvset", None, ["m"]))
        self._call(host, source="uvset", lightmaps=True)
        self.assertEqual(host.called("transfer_lightmaps"), [])
        host = _Host(errors={"transfer_lightmaps": "no lightmap UVs"})
        result = self._call(host, lightmaps=True)[1]
        self.assertEqual(
            (result.lightmap_results, result.lightmap_error), (None, "no lightmap UVs")
        )
        result = self._call(_Host(lightmap_results={}), lightmaps=True)[1]
        self.assertEqual(result.lightmap_results, {})


if __name__ == "__main__":
    unittest.main()
