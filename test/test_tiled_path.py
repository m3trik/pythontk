# !/usr/bin/python
# coding=utf-8
"""Tests for :class:`pythontk.TiledPath` -- the one tile/frame token vocabulary."""

import os
import shutil
import unittest

from pythontk.file_utils.tiled_path import TiledPath

HERE = os.path.dirname(os.path.abspath(__file__))


class TiledPathCase(unittest.TestCase):
    def setUp(self):
        self.dir = os.path.join(HERE, "temp_tests", "tiled_path")
        shutil.rmtree(self.dir, ignore_errors=True)
        os.makedirs(self.dir)

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def touch(self, *names):
        for name in names:
            with open(os.path.join(self.dir, name), "wb") as fh:
                fh.write(b"x")

    def path(self, name):
        return os.path.join(self.dir, name).replace("\\", "/")


class TestDetect(unittest.TestCase):
    def test_every_token_of_the_table_is_a_set(self):
        for name in (
            "a.<UDIM>.png",
            "a.<uvtile>.png",
            "a.<u>_<v>.png",
            "a.<f>.exr",
            "a.<FRAME>.exr",
        ):
            self.assertTrue(TiledPath.has_token(name), name)
        for name in ("a.1001.png", "a.png", "", None, "a.<udimx>.png"):
            self.assertFalse(TiledPath.has_token(name), name)

    def test_only_a_token_with_no_stand_in_counts_frames(self):
        self.assertTrue(TiledPath.is_frame_sequence("seq.<f>.exr"))
        self.assertTrue(TiledPath.is_frame_sequence("seq.<Frame>.exr"))
        self.assertFalse(TiledPath.is_frame_sequence("rock.<UDIM>.png"))
        self.assertFalse(TiledPath.is_frame_sequence("rock.<u>_<v>.png"))
        self.assertFalse(TiledPath.is_frame_sequence(None))

    def test_a_composite_token_is_never_shadowed_by_its_prefix(self):
        # "<uvtile>" must not read as "<u>" + "vtile>".
        self.assertEqual(TiledPath.wildcard("a.<uvtile>.png", "#"), "a.#.png")

    def test_scheme_names_placeholders_and_concrete_tiles(self):
        cases = {
            "<UDIM>": "udim",
            "1001": "udim",
            "<uvtile>": "uvtile",
            "<u>_<v>": "uvtile",
            "u1_v1": "uvtile",
            "U2_V3": "uvtile",
            "<f>": "frame",
            "<frame>": "frame",
            "rock": None,
            "": None,
        }
        for spelling, scheme in cases.items():
            self.assertEqual(TiledPath.scheme(spelling), scheme, spelling)

    def test_placeholder_pattern_matches_tile_tokens_only(self):
        import re

        rx = re.compile("(?:%s)$" % TiledPath.tile_token_pattern(), re.IGNORECASE)
        for token in ("<UDIM>", "<uvtile>", "<u>_<v>"):
            self.assertTrue(rx.search("rock." + token), token)
        for token in ("<f>", "<frame>", "<u>", "1001"):
            self.assertFalse(rx.search("rock." + token), token)


class TestWildcard(unittest.TestCase):
    def test_plain_wildcard(self):
        self.assertEqual(TiledPath.wildcard("rock.<UDIM>.png"), "rock.*.png")
        self.assertEqual(TiledPath.wildcard("rock.<u>_<v>.png"), "rock.*_*.png")
        self.assertEqual(TiledPath.wildcard(None), "")

    def test_strict_glob_escapes_literals_and_keeps_each_vocabulary(self):
        self.assertEqual(
            TiledPath.wildcard("sh[ot]/rock.<UDIM>.png", None),
            "sh[[]ot]/rock.[0-9][0-9][0-9][0-9].png",
        )
        self.assertEqual(TiledPath.wildcard("r.<uvtile>.png", None), "r.u*_v*.png")


class TestSpell(unittest.TestCase):
    def test_a_udim_number_spelled_in_each_tile_vocabulary(self):
        self.assertEqual(TiledPath.spell("r.<UDIM>.png", 1012), "r.1012.png")
        self.assertEqual(TiledPath.spell("r.<UVTILE>.png", 1001), "r.u1_v1.png")
        self.assertEqual(TiledPath.spell("r.<UVTILE>.png", 1012), "r.u2_v2.png")
        self.assertEqual(TiledPath.spell("r.<u>_<v>.png", 1002), "r.u2_v1.png")

    def test_frame_tokens_and_plain_paths_pass_through(self):
        self.assertEqual(TiledPath.spell("s.<f>.exr", 1001), "s.<f>.exr")
        self.assertEqual(TiledPath.spell("plain.png"), "plain.png")
        self.assertEqual(TiledPath.spell(""), "")


class TestTiles(TiledPathCase):
    def test_each_token_globs_by_its_own_vocabulary(self):
        self.touch("rock.1001.png", "rock.1002.png", "rock.u1_v1.png", "rock.thumb.png")
        self.assertEqual(
            TiledPath.tiles(self.path("rock.<UDIM>.png")),
            [self.path("rock.1001.png"), self.path("rock.1002.png")],
        )
        self.assertEqual(
            TiledPath.tiles(self.path("rock.<uvtile>.png")),
            [self.path("rock.u1_v1.png")],
        )
        self.assertEqual(
            TiledPath.tiles(self.path("rock.<u>_<v>.png")),
            [self.path("rock.u1_v1.png")],
        )

    def test_a_token_free_path_is_its_own_existence_verdict(self):
        self.touch("plain.png")
        self.assertEqual(
            TiledPath.tiles(self.path("plain.png")), [self.path("plain.png")]
        )
        self.assertEqual(TiledPath.tiles(self.path("gone.png")), [])
        self.assertEqual(TiledPath.tiles(""), [])

    def test_a_glob_special_folder_cannot_swallow_the_match(self):
        folder = os.path.join(self.dir, "sh[ot]_01")
        os.makedirs(folder)
        with open(os.path.join(folder, "t.1001.png"), "wb") as fh:
            fh.write(b"x")
        pattern = os.path.join(folder, "t.<UDIM>.png")
        self.assertEqual(len(TiledPath.tiles(pattern)), 1)


class TestRepresentative(TiledPathCase):
    def test_a_fixed_token_prefers_its_stand_in(self):
        self.touch("rock.1001.png", "rock.1002.png")
        self.assertEqual(
            TiledPath.representative(self.path("rock.<UDIM>.png")),
            self.path("rock.1001.png"),
        )

    def test_a_set_not_starting_at_the_stand_in_falls_back_to_its_first_tile(self):
        self.touch("rock.1003.png", "rock.1002.png")
        self.assertEqual(
            TiledPath.representative(self.path("rock.<UDIM>.png")),
            self.path("rock.1002.png"),
        )

    def test_nothing_on_disk_is_the_stand_in_where_it_was_looked_for(self):
        self.assertEqual(
            TiledPath.representative(self.path("rock.<UDIM>.png")),
            self.path("rock.1001.png"),
        )
        self.assertEqual(
            TiledPath.representative(self.path("rock.<u>_<v>.png")),
            self.path("rock.u1_v1.png"),
        )

    def test_a_frame_token_globs_or_reports_none(self):
        self.touch("seq.0007.exr", "seq.0008.exr")
        for token in ("<f>", "<frame>", "<FRAME>"):
            hit = TiledPath.representative(self.path("seq.%s.exr" % token))
            self.assertEqual(
                os.path.normcase(os.path.normpath(hit or "")),
                os.path.normcase(os.path.normpath(self.path("seq.0007.exr"))),
                token,
            )
        self.assertIsNone(TiledPath.representative(self.path("none.<f>.exr")))

    def test_token_free_and_empty(self):
        self.assertEqual(TiledPath.representative("x/plain.png"), "x/plain.png")
        self.assertIsNone(TiledPath.representative(""))
        self.assertIsNone(TiledPath.representative(None))


class TestRootExport(unittest.TestCase):
    def test_the_root_serves_the_class(self):
        import pythontk as ptk

        self.assertIs(ptk.TiledPath, TiledPath)


if __name__ == "__main__":
    unittest.main(verbosity=2)
