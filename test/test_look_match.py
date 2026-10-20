# !/usr/bin/python
# coding=utf-8
"""Tests for pythontk.core_utils.engines.lighting.look_match._look_match: the facade and its command line."""

import contextlib
import io
import json
import math
import os
import shutil
import tempfile
import unittest

import numpy as np
from PIL import Image

from pythontk.core_utils.engines.lighting.look_match._look_match import LookMatch
from pythontk.core_utils.engines.lighting.light_layers import LightLayers, LightMix
from pythontk import ImgUtils
from pythontk.core_utils.engines.lighting.light_predictor import LightPredictor
from pythontk.core_utils.engines.lighting.look_match.target import LookRegion, LookTarget

M709_TO_AP1 = np.array(
    [
        [0.61319, 0.33951, 0.04737],
        [0.07021, 0.91634, 0.01345],
        [0.02062, 0.10957, 0.86961],
    ]
)


class TestLookMatch(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="look_match_")
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        h, w = 24, 32
        xx = np.mgrid[0:h, 0:w][1] / w
        store = LightLayers(os.path.join(self.dir, "layers"))
        store.add(
            "key",
            np.repeat(np.exp(-((xx - 0.3) ** 2) / 0.03)[..., None], 3, 2).astype(
                np.float32
            ),
            spec={
                "kelvin": 3200,
                "lights": [
                    {
                        "name": "L_KEY",
                        "type": "spot",
                        "pos": [0, 200, 100],
                        "aim": [0, 90, 200],
                        "cone": 40,
                        "intensity": 2.0,
                    }
                ],
            },
        )
        store.add(
            "fill",
            np.full((h, w, 3), 0.2, np.float32),
            spec={
                "lights": [
                    {
                        "name": "L_FILL",
                        "type": "area",
                        "pos": [0, 300, 0],
                        "rot": [-90, 0, 0],
                    }
                ]
            },
        )
        store.add("panel", np.full((h, w, 3), 0.05, np.float32), kind="panel")
        store.add(
            "glow",
            np.full((h, w, 3), 0.1, np.float32),
            kind="emission",
            target={"node": "MAT_PANELS", "attr": "emissive_intensity"},
        )
        store.set_stamp(existing={"OFFICE_LIGHT": {"intensity": 0.0}})
        self.truth = LightMix(
            {"key": {"weight": 1.0, "kelvin": 3200}, "fill": {"weight": 0.5}}
        )
        ref = (LightPredictor(store).frame(self.truth) * 255 + 0.5).astype(np.uint8)
        Image.fromarray(ref).save(os.path.join(self.dir, "ref.png"))
        LookTarget(
            os.path.join(self.dir, "ref.png"),
            [
                LookRegion(
                    "left", "area", {"rect": [0.1, 0.2, 0.5, 0.8]}, measures=["mean"]
                ),
                LookRegion(
                    "right", "area", {"rect": [0.6, 0.2, 0.95, 0.8]}, measures=["mean"]
                ),
            ],
        ).save(os.path.join(self.dir, "target.json"))
        self.truth.with_weights({"key": 0.4}).save(os.path.join(self.dir, "mix.json"))
        self.session = os.path.join(self.dir, "session.json")
        with open(self.session, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "version": 1,
                    "layers": "layers",
                    "target": "target.json",
                    "mix": "mix.json",
                },
                f,
            )

    def test_a_session_opens_from_one_file(self):
        match = LookMatch.open(self.session)
        self.assertEqual(match.layers.names(), ["fill", "glow", "key", "panel"])
        self.assertEqual(match.mix.weight("key"), 0.4)
        self.assertEqual(match.predict().shape, (24, 32, 3))
        self.assertLess(match.score().region("left")["mean"].score, -0.1)

    def test_a_session_keeps_its_derived_data_in_the_user_cache(self):
        """``cache:`` paths: the authored files beside the session (a project),
        the captured layers in its cache folder (off the synced drive)."""
        from pythontk.core_utils.user_config import UserConfig

        cache = os.path.join(self.dir, "user_cache")
        prev = os.environ.get(UserConfig.CACHE_ROOT_ENV_VAR)
        os.environ[UserConfig.CACHE_ROOT_ENV_VAR] = cache
        self.addCleanup(
            lambda: os.environ.pop(UserConfig.CACHE_ROOT_ENV_VAR, None)
            if prev is None
            else os.environ.__setitem__(UserConfig.CACHE_ROOT_ENV_VAR, prev)
        )
        shutil.move(
            os.path.join(self.dir, "layers"),
            os.path.join(cache, "look_match", "room", "layers"),
        )
        with open(self.session, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "version": 1,
                    "id": "room",
                    "layers": "cache:layers",
                    "target": "target.json",
                    "mix": "mix.json",
                },
                f,
            )
        match = LookMatch.open(self.session)
        self.assertEqual(match.layers.names(), ["fill", "glow", "key", "panel"])
        self.assertEqual(match.mix.weight("key"), 0.4)

    def test_fit_moves_the_mix_toward_the_reference(self):
        match = LookMatch.open(self.session)
        result = match.fit(["key"], tolerance=0.0, regularize=0.0)
        self.assertAlmostEqual(result.mix.weight("key"), 1.0, delta=0.1)
        self.assertIs(match.mix, result.mix)  # the session moves on to the fitted mix

    def test_contributions_name_the_lights_of_a_region(self):
        shares = LookMatch.open(self.session).contributions("left")
        self.assertAlmostEqual(sum(shares.values()), 1.0, places=5)
        self.assertIn("key", shares)

    def test_a_rig_outside_the_payload_contract_is_refused(self):
        """The writer holds itself to LightRigPayload: a role's light spelled
        outside it never reaches a host as a half-built rig."""
        match = LookMatch.open(self.session)
        frame = match.layers.load("key")
        match.layers.add(
            "laser", frame, spec={"lights": [{"name": "L_X", "type": "laser"}]}
        )
        with self.assertRaises(ValueError) as ctx:
            match.to_rig(LightMix({"laser": {"weight": 1.0}}))
        self.assertIn("laser", str(ctx.exception))

    def test_a_rig_carries_each_light_at_its_weight_gain_and_colour(self):
        match = LookMatch.open(self.session)
        rig = match.to_rig(
            LightMix(
                {
                    "key": {"weight": 0.5, "kelvin": 3200},
                    "fill": {"weight": 2.0},
                    "panel": {"weight": 1.0},
                    "glow": {"weight": 0.25},
                }
            ),
            gain=math.pi,
        )
        lights = {light["name"]: light for light in rig["lights"]}
        self.assertEqual(set(lights), {"L_KEY", "L_FILL"})
        ap1 = M709_TO_AP1 @ np.asarray(ImgUtils.kelvin_tint(3200), np.float32)
        expected = (
            2.0 * math.pi * 0.5 * ap1.max()
        )  # the matrix here is a 5-digit constant
        self.assertAlmostEqual(
            lights["L_KEY"]["intensity"], expected, delta=1e-3 * expected
        )
        np.testing.assert_allclose(lights["L_KEY"]["color"], ap1 / ap1.max(), atol=1e-3)
        self.assertAlmostEqual(
            lights["L_FILL"]["intensity"], math.pi * 2.0, places=4
        )  # white: AP1 white is white
        self.assertEqual(rig["materials"], {"MAT_PANELS": {"emissive_intensity": 0.25}})
        self.assertEqual(rig["existing"], {"OFFICE_LIGHT": {"intensity": 0.0}})
        self.assertEqual(rig["source"]["viewer_only"], ["panel"])
        self.assertAlmostEqual(rig["bake_gain"], math.pi)

    def test_a_rig_light_carries_its_layers_specular_scale(self):
        rig = LookMatch.open(self.session).to_rig(
            LightMix({"key": {"weight": 1.0, "specular": 0.5}, "fill": {"weight": 1.0}})
        )
        lights = {light["name"]: light for light in rig["lights"]}
        self.assertEqual(lights["L_KEY"]["specular"], 0.5)  # LightRig: aiSpecular
        self.assertNotIn("specular", lights["L_FILL"])  # the light's own level

    def test_a_rig_carries_the_material_state_its_layers_were_baked_under(self):
        """Portable: applied to a scene that differs (an office lit where the
        layers were baked dark), the rig sets the bake's own state."""
        match = LookMatch.open(self.session)
        match.layers.set_stamp(
            materials={
                "MAT_ROOM_ai": {"emission": 0.0},
                "MAT_PANELS": {"emissive_intensity": 0.0},
            }
        )
        rig = match.to_rig(LightMix({"key": {"weight": 1.0}, "glow": {"weight": 0.25}}))
        self.assertEqual(rig["materials"]["MAT_ROOM_ai"], {"emission": 0.0})
        self.assertEqual(  # an emission layer's own level wins on its target
            rig["materials"]["MAT_PANELS"], {"emissive_intensity": 0.25}
        )
        self.assertEqual(
            match.layers.stamp["materials"]["MAT_PANELS"], {"emissive_intensity": 0.0}
        )

    def test_the_sheet_is_written(self):
        out = os.path.join(self.dir, "sheet.png")
        LookMatch.open(self.session).sheet(out)
        self.assertTrue(os.path.isfile(out))

    # ------------------------------------------------------------------ reading a look
    def test_a_readout_gives_each_region_its_colour_beside_the_references(self):
        match = LookMatch.open(self.session)
        read = match.readout({"mix": match.mix, "truth": self.truth})
        self.assertEqual(sorted(read), ["left", "right"])
        left = read["left"]
        self.assertEqual(list(left), ["mix", "truth", "reference"])
        np.testing.assert_allclose(left["truth"], left["reference"], atol=1.0)
        self.assertLess(left["mix"][3], left["reference"][3] - 5)  # luma: too dark
        frame = os.path.join(self.dir, "ref.png")  # a frame on disk reads the same
        np.testing.assert_allclose(
            match.readout({"shot": frame})["left"]["shot"],
            left["reference"],
            atol=1.0,
        )

    def test_owners_rank_the_lights_by_how_selectively_they_light_a_region(self):
        owners = LookMatch.open(self.session).owners("left")
        self.assertEqual([name for name, _ in owners], ["key", "fill"])  # lights only
        self.assertGreater(owners[0][1], 1.5)  # the key's pool sits on the left
        self.assertAlmostEqual(
            owners[1][1], 1.0, places=3
        )  # an even fill: no more there

    def test_solve_finds_the_weight_that_brings_a_region_to_the_reference(self):
        match = LookMatch.open(self.session)
        weight = match.solve("key", "left", self.truth.with_weights({"key": 0.0}))
        self.assertAlmostEqual(weight, 1.0, delta=0.05)
        with self.assertRaises(ValueError):  # already too bright without it
            match.solve("fill", "left", self.truth.with_weights({"key": 3.0}))

    def test_a_layer_alone_is_exposed_to_be_seen(self):
        weight, frame = LookMatch.open(self.session).exposed("key", level=0.6)
        luma = frame @ np.array([0.2126, 0.7152, 0.0722])
        self.assertAlmostEqual(float(np.percentile(luma, 99.5)), 0.6, delta=0.01)
        self.assertGreater(weight, 0)

    # ------------------------------------------------------------------ command line
    def run_cli(self, *argv):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = LookMatch.cli(list(argv))
        return code, buf.getvalue()

    def test_cli_score_prints_a_table(self):
        code, out = self.run_cli("score", self.session)
        self.assertEqual(code, 0)
        self.assertIn("left", out)
        self.assertIn("mean", out)

    def test_cli_score_json_is_machine_readable(self):
        code, out = self.run_cli("score", self.session, "--json")
        rows = json.loads(out)
        self.assertTrue(any(r["region"] == "left" for r in rows))

    def test_cli_fit_writes_the_mix(self):
        out = os.path.join(self.dir, "fitted.json")
        code, _ = self.run_cli(
            "fit",
            self.session,
            "--free",
            "key",
            "--tolerance",
            "0",
            "--regularize",
            "0",
            "--out",
            out,
        )
        self.assertEqual(code, 0)
        self.assertAlmostEqual(LightMix.load(out).weight("key"), 1.0, delta=0.1)

    def test_cli_fit_stops_at_max_iter(self):
        out = os.path.join(self.dir, "fitted.json")
        code, text = self.run_cli(
            "fit",
            self.session,
            "--free",
            "key,fill",
            "--max-iter",
            "1",
            "--tolerance",
            "0",
            "--regularize",
            "0",
            "--out",
            out,
        )
        self.assertEqual(code, 0)
        steps = int(text.split(" in ")[1].split(" steps")[0])
        self.assertLessEqual(steps, 1, text)

    def test_a_frame_scores_like_the_mix_it_shows(self):
        """An exact capture is judged by the same measures as a prediction."""
        match = LookMatch.open(self.session)
        full = match.layers.size
        shown = match.predictor.frame(match.mix, full)
        path = os.path.join(self.dir, "exact.png")
        Image.fromarray((np.clip(shown, 0, 1) * 255 + 0.5).astype(np.uint8)).save(path)
        by_mix = {
            (r["region"], r["measure"]): r["ours"]
            for r in match.score(size=full).rows()
        }
        by_frame = match.score(size=full, frame=path).rows()
        self.assertTrue(by_frame)
        for row in by_frame:
            self.assertAlmostEqual(
                row["ours"], by_mix[(row["region"], row["measure"])], delta=0.02
            )
        code, out = self.run_cli(
            "score", self.session, "--frame", path, "--size", f"{full[0]}x{full[1]}"
        )
        self.assertEqual(code, 0)
        sheet = os.path.join(self.dir, "framed.png")
        self.assertEqual(
            self.run_cli("sheet", self.session, "--frame", path, "--out", sheet)[0], 0
        )
        self.assertTrue(os.path.isfile(sheet))

    def test_cli_rig_and_sheet_write_files(self):
        rig = os.path.join(self.dir, "rig.json")
        sheet = os.path.join(self.dir, "s.png")
        self.assertEqual(self.run_cli("rig", self.session, "--out", rig)[0], 0)
        with open(rig, encoding="utf-8") as f:
            self.assertEqual(json.load(f)["version"], 1)
        self.assertEqual(self.run_cli("sheet", self.session, "--out", sheet)[0], 0)
        self.assertTrue(os.path.isfile(sheet))

    def test_cli_readout_owners_solve_and_tiles(self):
        truth = os.path.join(self.dir, "truth.json")
        self.truth.save(truth)
        code, out = self.run_cli("readout", self.session, "--compare", f"truth={truth}")
        self.assertEqual(code, 0)
        for word in ("left", "mix", "truth", "reference"):
            self.assertIn(word, out)
        code, out = self.run_cli("owners", self.session, "--region", "left")
        self.assertEqual(code, 0)
        self.assertLess(out.index("key"), out.index("fill"))
        code, out = self.run_cli(
            "solve", self.session, "--layer", "key", "--region", "left"
        )
        self.assertEqual(code, 0)
        self.assertIn("key", out)
        tiles = os.path.join(self.dir, "tiles.png")
        code, _ = self.run_cli(
            "tiles",
            self.session,
            "--out",
            tiles,
            "--solo",
            "key,fill",
            "--compare",
            f"truth={truth}",
        )
        self.assertEqual(code, 0)
        with Image.open(tiles) as img:  # mix, truth, key, fill, reference: 3 rows of 2
            self.assertEqual(img.size[0], 1640)
            self.assertGreater(img.size[1], img.size[0])

    def test_cli_layers_lists_the_store(self):
        code, out = self.run_cli("layers", self.session)
        self.assertEqual(code, 0)
        self.assertIn("panel", out)
        self.assertIn("viewer-only", out)


if __name__ == "__main__":
    unittest.main(exit=False)
