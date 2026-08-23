#!/usr/bin/python
# coding=utf-8
"""
Unit tests for pythontk StrUtils.

Comprehensive edge case coverage for:
- set_case
- get_mangled_name
- get_text_between_delimiters
- get_matching_hierarchy_items
- split_at_delimiter
- insert / rreplace
- truncate
- get_trailing_integers
- find_str / find_str_and_format
- format_suffix

Run with:
    python -m pytest test_str.py -v
    python test_str.py
"""
import unittest

from pythontk import StrUtils

from conftest import BaseTestCase


class StrTest(BaseTestCase):
    """String utilities test class with comprehensive edge case coverage."""

    # -------------------------------------------------------------------------
    # set_case Tests
    # -------------------------------------------------------------------------

    def test_set_case_basic(self):
        """Test set_case converts strings to various cases."""
        self.assertEqual(StrUtils.set_case("xxx", "upper"), "XXX")
        self.assertEqual(StrUtils.set_case("XXX", "lower"), "xxx")
        self.assertEqual(StrUtils.set_case("xxx", "capitalize"), "Xxx")
        self.assertEqual(StrUtils.set_case("xxX", "swapcase"), "XXx")
        self.assertEqual(StrUtils.set_case("xxx XXX", "title"), "Xxx Xxx")
        self.assertEqual(StrUtils.set_case("xXx", "pascal"), "XXx")
        self.assertEqual(StrUtils.set_case("xXx", "camel"), "xXx")

    def test_set_case_list_input(self):
        """Test set_case with list input."""
        self.assertEqual(StrUtils.set_case(["xXx"], "camel"), ["xXx"])
        self.assertEqual(StrUtils.set_case(["abc", "def"], "upper"), ["ABC", "DEF"])

    def test_set_case_none_and_empty(self):
        """Test set_case with None and empty string."""
        self.assertEqual(StrUtils.set_case(None, "camel"), "")
        self.assertEqual(StrUtils.set_case("", "camel"), "")
        self.assertEqual(StrUtils.set_case("", "upper"), "")

    def test_set_case_unicode(self):
        """Test set_case with unicode characters."""
        self.assertEqual(StrUtils.set_case("ñoño", "upper"), "ÑOÑO")
        self.assertEqual(StrUtils.set_case("CAFÉ", "lower"), "café")
        self.assertEqual(StrUtils.set_case("über", "capitalize"), "Über")

    def test_set_case_single_char(self):
        """Test set_case with single character."""
        self.assertEqual(StrUtils.set_case("a", "upper"), "A")
        self.assertEqual(StrUtils.set_case("Z", "lower"), "z")

    def test_set_case_numbers_and_symbols(self):
        """Test set_case with numbers and symbols."""
        self.assertEqual(StrUtils.set_case("123abc", "upper"), "123ABC")
        self.assertEqual(StrUtils.set_case("!@#ABC", "lower"), "!@#abc")

    def test_set_case_whitespace(self):
        """Test set_case with whitespace strings."""
        self.assertEqual(StrUtils.set_case("   ", "upper"), "   ")
        self.assertEqual(StrUtils.set_case("\t\n", "lower"), "\t\n")

    # -------------------------------------------------------------------------
    # get_mangled_name Tests
    # -------------------------------------------------------------------------

    def test_get_mangled_name_with_class_string(self):
        """Test get_mangled_name with class name as string."""
        self.assertEqual(
            StrUtils.get_mangled_name("DummyClass", "__my_attribute"),
            "_DummyClass__my_attribute",
        )

    def test_get_mangled_name_with_class(self):
        """Test get_mangled_name with class object."""

        class DummyClass:
            pass

        self.assertEqual(
            StrUtils.get_mangled_name(DummyClass, "__my_attribute"),
            "_DummyClass__my_attribute",
        )

    def test_get_mangled_name_with_instance(self):
        """Test get_mangled_name with class instance."""

        class DummyClass:
            pass

        dummy_instance = DummyClass()
        self.assertEqual(
            StrUtils.get_mangled_name(dummy_instance, "__my_attribute"),
            "_DummyClass__my_attribute",
        )

    def test_get_mangled_name_invalid_attr_type(self):
        """Test get_mangled_name raises TypeError for non-string attribute."""
        with self.assertRaises(TypeError):
            StrUtils.get_mangled_name("DummyClass", 123)

    def test_get_mangled_name_invalid_attr_prefix(self):
        """Test get_mangled_name raises ValueError for non-dunder attribute."""
        with self.assertRaises(ValueError):
            StrUtils.get_mangled_name("DummyClass", "my_attribute")

    def test_get_mangled_name_single_underscore(self):
        """Test get_mangled_name with single underscore prefix."""
        with self.assertRaises(ValueError):
            StrUtils.get_mangled_name("MyClass", "_single")

    # -------------------------------------------------------------------------
    # get_text_between_delimiters Tests
    # -------------------------------------------------------------------------

    def test_get_text_between_delimiters_basic(self):
        """Test get_text_between_delimiters extracts text between markers."""
        input_string = (
            "Here is the <!-- start -->first match<!-- end --> and "
            "here is the <!-- start -->second match<!-- end -->"
        )
        result = StrUtils.get_text_between_delimiters(
            input_string, "<!-- start -->", "<!-- end -->", as_string=True
        )
        self.assertEqual(result, "first match second match")

    def test_get_text_between_delimiters_no_match(self):
        """Test get_text_between_delimiters with no matches."""
        result = StrUtils.get_text_between_delimiters(
            "no delimiters here", "[start]", "[end]", as_string=True
        )
        self.assertEqual(result, "")

    def test_get_text_between_delimiters_empty_string(self):
        """Test get_text_between_delimiters with empty input."""
        result = StrUtils.get_text_between_delimiters(
            "", "[start]", "[end]", as_string=True
        )
        self.assertEqual(result, "")

    def test_get_text_between_delimiters_nested(self):
        """Test get_text_between_delimiters with adjacent delimiters."""
        input_string = "(a)(b)(c)"
        result = list(StrUtils.get_text_between_delimiters(input_string, "(", ")"))
        self.assertEqual(result, ["a", "b", "c"])

    def test_get_text_between_delimiters_empty_content(self):
        """Test get_text_between_delimiters with empty content between delimiters."""
        result = list(
            StrUtils.get_text_between_delimiters("[start][end]", "[start]", "[end]")
        )
        self.assertEqual(result, [""])

    # -------------------------------------------------------------------------
    # get_matching_hierarchy_items Tests
    # -------------------------------------------------------------------------

    def test_get_matching_hierarchy_items_upstream(self):
        """Test get_matching_hierarchy_items finds upstream items."""
        hierarchy_items = [
            "polygons|mesh#submenu",
            "polygons|submenu",
            "polygons",
            "polygons|mesh",
            "polygons|other",
            "polygons|mesh|other",
            "other",
        ]
        target = "polygons|mesh"
        self.assertEqual(
            StrUtils.get_matching_hierarchy_items(
                hierarchy_items, target, upstream=True
            ),
            ["polygons"],
        )

    def test_get_matching_hierarchy_items_downstream(self):
        """Test get_matching_hierarchy_items finds downstream items."""
        hierarchy_items = [
            "polygons|mesh#submenu",
            "polygons|submenu",
            "polygons",
            "polygons|mesh",
            "polygons|other",
            "polygons|mesh|other",
            "other",
        ]
        target = "polygons|mesh"
        self.assertEqual(
            StrUtils.get_matching_hierarchy_items(
                hierarchy_items, target, downstream=True, delimiters=["|", "#"]
            ),
            ["polygons|mesh|other", "polygons|mesh#submenu"],
        )

    def test_get_matching_hierarchy_items_reversed(self):
        """Test get_matching_hierarchy_items with reverse option."""
        hierarchy_items = [
            "polygons|mesh#submenu",
            "polygons|mesh|other",
        ]
        target = "polygons|mesh"
        self.assertEqual(
            StrUtils.get_matching_hierarchy_items(
                hierarchy_items,
                target,
                downstream=True,
                delimiters=["|", "#"],
                reverse=True,
            ),
            ["polygons|mesh#submenu", "polygons|mesh|other"],
        )

    def test_get_matching_hierarchy_items_exact(self):
        """Test get_matching_hierarchy_items with exact option."""
        hierarchy_items = ["polygons", "polygons|mesh"]
        target = "polygons|mesh"
        self.assertEqual(
            StrUtils.get_matching_hierarchy_items(
                hierarchy_items, target, upstream=True, exact=True
            ),
            ["polygons", "polygons|mesh"],
        )

    def test_get_matching_hierarchy_items_empty_list(self):
        """Test get_matching_hierarchy_items with empty list."""
        self.assertEqual(
            StrUtils.get_matching_hierarchy_items([], "target", upstream=True), []
        )

    # -------------------------------------------------------------------------
    # split_delimited_string Tests
    # -------------------------------------------------------------------------

    def test_split_delimited_string_basic(self):
        """Test split_delimited_string splits strings correctly."""
        # Test list output (default)
        self.assertEqual(
            StrUtils.split_delimited_string("str|ing"),
            ["str", "ing"],
        )
        # Test tuple output (occurrence specified)
        self.assertEqual(
            StrUtils.split_delimited_string("str|ing", occurrence=-1),
            ("str", "ing"),
        )
        # Test list input (vectorized)
        self.assertEqual(
            StrUtils.split_delimited_string(["str|ing", "string"], occurrence=-1),
            [("str", "ing"), ("string", "")],
        )

    def test_split_delimited_string_with_occurrence(self):
        """Test split_delimited_string with specific occurrence."""
        self.assertEqual(
            StrUtils.split_delimited_string("aCHARScCHARSd", "CHARS", occurrence=0),
            ("", "a"),
        )

    def test_split_delimited_string_empty_string(self):
        """Test split_delimited_string with empty string."""
        # List mode
        self.assertEqual(StrUtils.split_delimited_string(""), [])
        # Tuple mode
        self.assertEqual(StrUtils.split_delimited_string("", occurrence=-1), ("", ""))

    def test_split_delimited_string_no_delimiter(self):
        """Test split_delimited_string when delimiter not found."""
        # List mode
        self.assertEqual(StrUtils.split_delimited_string("hello", "|"), ["hello"])
        # Tuple mode
        self.assertEqual(
            StrUtils.split_delimited_string("hello", "|", occurrence=-1), ("hello", "")
        )

    def test_split_delimited_string_delimiter_at_start(self):
        """Test split_delimited_string with delimiter at start."""
        self.assertEqual(
            StrUtils.split_delimited_string("|hello", occurrence=-1), ("", "hello")
        )

    def test_split_delimited_string_delimiter_at_end(self):
        """Test split_delimited_string with delimiter at end."""
        self.assertEqual(
            StrUtils.split_delimited_string("hello|", occurrence=-1), ("hello", "")
        )

    def test_split_delimited_string_multiple_delimiters(self):
        """Test split_delimited_string with multiple occurrences."""
        # Default splits all
        self.assertEqual(StrUtils.split_delimited_string("a|b|c"), ["a", "b", "c"])
        # Split at last
        self.assertEqual(
            StrUtils.split_delimited_string("a|b|c", occurrence=-1), ("a|b", "c")
        )

    # -------------------------------------------------------------------------
    # insert Tests
    # -------------------------------------------------------------------------

    def test_insert_basic(self):
        """Test insert adds substrings at specified positions."""
        self.assertEqual(
            StrUtils.insert("ins into str", "substr ", " "),
            "ins substr into str",
        )

    def test_insert_from_end(self):
        """Test insert from end of string."""
        self.assertEqual(
            StrUtils.insert("ins into str", " end of", " ", -1, True),
            "ins into end of str",
        )

    def test_insert_no_delimiter(self):
        """Test insert when delimiter not found."""
        self.assertEqual(
            StrUtils.insert("ins into str", "insert this", "atCharsThatDontExist"),
            "ins into str",
        )

    def test_insert_at_index(self):
        """Test insert at numeric index."""
        self.assertEqual(StrUtils.insert("ins into str", 666, 0), "666ins into str")

    def test_insert_empty_string(self):
        """Test insert into empty string."""
        self.assertEqual(StrUtils.insert("", "text", 0), "text")

    def test_insert_empty_substring(self):
        """Test insert empty substring."""
        self.assertEqual(StrUtils.insert("hello", "", " "), "hello")

    # -------------------------------------------------------------------------
    # strip_ansi Tests
    # -------------------------------------------------------------------------

    def test_strip_ansi_removes_sgr_color(self):
        """SGR color codes are removed, the visible text is kept verbatim."""
        self.assertEqual(StrUtils.strip_ansi("\x1b[35mFile\x1b[0m"), "File")
        self.assertEqual(StrUtils.strip_ansi("\x1b[1;31mbold red\x1b[0m"), "bold red")

    def test_strip_ansi_python_colored_traceback(self):
        """CPython emits this shape whenever stderr is a TTY; a tee inherits it."""
        line = (
            '  File \x1b[35m"space_view3d.py"\x1b[0m, line \x1b[35m5560\x1b[0m, '
            "in \x1b[35mdraw\x1b[0m"
        )
        self.assertEqual(
            StrUtils.strip_ansi(line), '  File "space_view3d.py", line 5560, in draw'
        )
        self.assertNotIn("\x1b", StrUtils.strip_ansi(line))

    def test_strip_ansi_preserves_indentation_and_newlines(self):
        """Block formatting downstream keys off leading whitespace — it must survive."""
        self.assertEqual(
            StrUtils.strip_ansi("    arm = \x1b[1;31mx.data\x1b[0m\n"),
            "    arm = x.data\n",
        )

    def test_strip_ansi_non_sgr_sequences(self):
        """Cursor/erase sequences and two-character escapes go too, not just color."""
        self.assertEqual(StrUtils.strip_ansi("a\x1b[2J\x1b[Hb"), "ab")  # erase + home
        self.assertEqual(StrUtils.strip_ansi("a\x1bDb"), "ab")  # ESC Fe (index)

    def test_strip_ansi_leaves_a_bare_escape_alone(self):
        """Scope is CSI + the ESC Fe range (0x40-0x5F) — the sequences a color-capable
        stream actually emits. An ESC that opens nothing recognizable is left as-is
        rather than swallowing the character after it."""
        self.assertEqual(StrUtils.strip_ansi("a\x1bb"), "a\x1bb")

    def test_strip_ansi_passes_clean_text_through(self):
        """Maya's reporter carries no escapes — its text must be untouched."""
        text = "// Error: line 1 //\n"
        self.assertEqual(StrUtils.strip_ansi(text), text)

    def test_strip_ansi_is_idempotent(self):
        """Stripped at ingest AND at the widget; the second pass must be a no-op."""
        once = StrUtils.strip_ansi("\x1b[35mx\x1b[0m")
        self.assertEqual(StrUtils.strip_ansi(once), once)

    def test_strip_ansi_edge_inputs(self):
        """Empty/non-str input returns unchanged rather than raising."""
        self.assertEqual(StrUtils.strip_ansi(""), "")
        self.assertIsNone(StrUtils.strip_ansi(None))
        self.assertEqual(StrUtils.strip_ansi(42), 42)

    # -------------------------------------------------------------------------
    # rreplace Tests
    # -------------------------------------------------------------------------

    def test_rreplace_all_occurrences(self):
        """Test rreplace replaces all from right side."""
        self.assertEqual(StrUtils.rreplace("aabbccbb", "bb", 22), "aa22cc22")

    def test_rreplace_limited(self):
        """Test rreplace with count limit."""
        self.assertEqual(StrUtils.rreplace("aabbccbb", "bb", 22, 1), "aabbcc22")
        self.assertEqual(StrUtils.rreplace("aabbccbb", "bb", 22, 3), "aa22cc22")

    def test_rreplace_zero_count(self):
        """Test rreplace with zero count."""
        self.assertEqual(StrUtils.rreplace("aabbccbb", "bb", 22, 0), "aabbccbb")

    def test_rreplace_not_found(self):
        """Test rreplace when pattern not found."""
        self.assertEqual(StrUtils.rreplace("hello", "xyz", "abc"), "hello")

    def test_rreplace_empty_string(self):
        """Test rreplace on empty string."""
        self.assertEqual(StrUtils.rreplace("", "a", "b"), "")

    # -------------------------------------------------------------------------
    # collapse_delimiter_runs Tests
    # -------------------------------------------------------------------------

    def test_collapse_delimiter_runs_basic(self):
        """Test runs of the delimiter collapse to a single one."""
        self.assertEqual(
            StrUtils.collapse_delimiter_runs("vdat____Shape702"), "vdat_Shape702"
        )

    def test_collapse_delimiter_runs_trailing(self):
        """Test trailing runs are stripped by default."""
        self.assertEqual(StrUtils.collapse_delimiter_runs("Crate__"), "Crate")
        self.assertEqual(
            StrUtils.collapse_delimiter_runs("Crate__", strip_trailing=False), "Crate_"
        )

    def test_collapse_delimiter_runs_leading_preserved(self):
        """Test a leading delimiter (legality prefix) survives."""
        self.assertEqual(
            StrUtils.collapse_delimiter_runs("_Lead__x"), "_Lead_x"
        )

    def test_collapse_delimiter_runs_noop_and_edge(self):
        """Test clean names and non-string input pass through."""
        self.assertEqual(StrUtils.collapse_delimiter_runs("a_b_c"), "a_b_c")
        self.assertEqual(StrUtils.collapse_delimiter_runs(""), "")
        self.assertIsNone(StrUtils.collapse_delimiter_runs(None))

    # -------------------------------------------------------------------------
    # truncate Tests
    # -------------------------------------------------------------------------

    def test_truncate_start(self):
        """Test truncate from start (default)."""
        self.assertEqual(StrUtils.truncate("12345678", 4), "..5678")

    def test_truncate_end(self):
        """Test truncate from end."""
        self.assertEqual(StrUtils.truncate("12345678", 4, "end"), "1234..")

    def test_truncate_custom_indicator(self):
        """Test truncate with custom indicator."""
        self.assertEqual(StrUtils.truncate("12345678", 4, "end", "--"), "1234--")

    def test_truncate_middle(self):
        """Test truncate from middle."""
        self.assertEqual(StrUtils.truncate("12345678", 6, "middle"), "12..78")

    def test_truncate_path_keeps_root_dir_and_filename(self):
        """'path' mode opens on drive + first dir and closes on the filename."""
        p = "O:/Cloud/Projects/jets/c130j/sourceimages/textures/c130j_body_DIFF.png"
        result = StrUtils.truncate(p, 48, "path", "…")
        self.assertEqual(result, "O:/Cloud/Projects/…/textures/c130j_body_DIFF.png")
        self.assertLessEqual(len(result), 48)

    def test_truncate_path_matches_the_documented_example(self):
        """Pin the docstring example — a wrong example is a wrong contract."""
        self.assertEqual(
            StrUtils.truncate(
                "O:/Cloud/jets/c130j/sourceimages/tex/x_DIFF.png", 36, "path"
            ),
            "O:/Cloud/jets/../tex/x_DIFF.png",
        )

    def test_truncate_path_relative_keeps_leading_dirs(self):
        """The common DCC case: paths stored relative to a project root."""
        p = "sourceimages/textures/props/crate/deep/crate_DIFF.png"
        self.assertEqual(
            StrUtils.truncate(p, 40, "path", "…"),
            "sourceimages/textures/…/crate_DIFF.png",
        )

    def test_truncate_path_head_cap_widens_the_tail(self):
        """``head`` pins the front so the budget lands on the filename end."""
        p = "O:/Cloud/Projects/jets/c130j/sourceimages/textures/c130j_body_DIFF.png"
        capped = StrUtils.truncate(p, 48, "path", "…", head=1)
        self.assertEqual(capped, "O:/…/sourceimages/textures/c130j_body_DIFF.png")
        self.assertLessEqual(len(capped), 48)
        # Same budget, uncapped: the head takes what the tail could not use.
        uncapped = StrUtils.truncate(p, 48, "path", "…")
        self.assertEqual(uncapped, "O:/Cloud/Projects/…/textures/c130j_body_DIFF.png")

    def test_truncate_path_head_cap_never_narrows_the_tail(self):
        """The tail is grown first, so a lower cap can only widen it."""
        p = "O:/Cloud/Projects/jets/c130j/sourceimages/textures/c130j_body_DIFF.png"

        def tail_len(result):
            return len(result.split("…", 1)[1])

        for length in (40, 48, 56, 67, 80):
            wide = StrUtils.truncate(p, length, "path", "…")
            narrow = StrUtils.truncate(p, length, "path", "…", head=1)
            if "…" in wide and "…" in narrow:
                self.assertGreaterEqual(tail_len(narrow), tail_len(wide), length)

    def test_truncate_path_head_none_is_the_previous_behavior(self):
        p = "O:/Cloud/Projects/jets/c130j/sourceimages/textures/c130j_body_DIFF.png"
        self.assertEqual(
            StrUtils.truncate(p, 48, "path", "…", head=None),
            StrUtils.truncate(p, 48, "path", "…"),
        )

    def test_truncate_path_head_below_one_is_clamped(self):
        """A head of 0 would leave no anchor at all; 1 is the floor."""
        p = "O:/Cloud/Projects/jets/c130j/sourceimages/textures/c130j_body_DIFF.png"
        self.assertEqual(
            StrUtils.truncate(p, 48, "path", "…", head=0),
            StrUtils.truncate(p, 48, "path", "…", head=1),
        )

    def test_truncate_path_head_cap_ignored_by_other_modes(self):
        p = "O:/Cloud/Projects/jets/sourceimages/c130j_body_DIFF.png"
        for mode in ("start", "end", "middle"):
            self.assertEqual(
                StrUtils.truncate(p, 30, mode, "…", head=1),
                StrUtils.truncate(p, 30, mode, "…"),
            )

    def test_truncate_path_head_cap_still_marks_a_real_elision(self):
        """Capping the head must not emit an insert between adjacent parts."""
        p = "aaa/bbb/ccc.png"
        result = StrUtils.truncate(p, 13, "path", "…", head=1)
        kept = [c for c in result.split("/") if c != "…"]
        if "…" in result:
            self.assertLess(len(kept), len(p.split("/")))

    def test_truncate_path_cuts_only_at_separators(self):
        """The point of 'path' over 'middle': no half-component at the seam."""
        p = "O:/Cloud/Projects/jets/c130j/sourceimages/textures/c130j_body_DIFF.png"
        path_mode = StrUtils.truncate(p, 48, "path", "…")
        middle_mode = StrUtils.truncate(p, 48, "middle", "…")
        for segment in path_mode.split("/"):
            self.assertIn(segment, p.split("/") + ["…"])
        self.assertNotEqual(path_mode, middle_mode)

    def test_truncate_path_preserves_separator_style(self):
        p = "D:\\Art\\Projects\\jets\\c130j\\sourceimages\\tex\\body_DIFF.png"
        result = StrUtils.truncate(p, 48, "path")
        self.assertNotIn("/", result)
        self.assertTrue(result.startswith("D:\\Art\\"))
        self.assertTrue(result.endswith("body_DIFF.png"))

    def test_truncate_path_preserves_leading_separators(self):
        """A UNC share / posix root must not lose its leading separators."""
        unc = "//server/share/proj/sourceimages/textures/tile_1001.png"
        self.assertTrue(StrUtils.truncate(unc, 44, "path").startswith("//server/"))
        posix = "/usr/local/share/textures/deep/deeper/deepest/tex_DIFF.png"
        self.assertTrue(StrUtils.truncate(posix, 40, "path").startswith("/usr/"))

    def test_truncate_path_falls_back_when_nothing_to_drop(self):
        """Two components, or a filename that alone overruns, degrade to 'middle'."""
        two = "textures/an_extremely_long_texture_filename_here.png"
        self.assertEqual(
            StrUtils.truncate(two, 30, "path"), StrUtils.truncate(two, 30, "middle")
        )
        tiny = "O:/Cloud/Projects/jets/sourceimages/tex/body_DIFF.png"
        self.assertEqual(
            StrUtils.truncate(tiny, 12, "path"), StrUtils.truncate(tiny, 12, "middle")
        )

    def test_truncate_path_no_separator_falls_back(self):
        no_sep = "nofileseparatoratallbutlongenoughtotriggertruncation.png"
        self.assertEqual(
            StrUtils.truncate(no_sep, 30, "path"),
            StrUtils.truncate(no_sep, 30, "middle"),
        )

    def test_truncate_path_marker_always_marks_a_real_elision(self):
        """Never emit an insert between components that were adjacent."""
        p = "aaa/" + "b" * 20 + "/ccc.png"
        result = StrUtils.truncate(p, 20, "path", "…")
        self.assertEqual(result, "aaa/…/ccc.png")
        # Round-tripping the kept components must skip at least one original.
        kept = [c for c in result.split("/") if c != "…"]
        self.assertLess(len(kept), len(p.split("/")))

    def test_truncate_path_short_enough_is_untouched(self):
        p = "sourceimages/textures/props/crate/crate_DIFF.png"
        self.assertEqual(StrUtils.truncate(p, 48, "path"), p)

    def test_truncate_path_never_exceeds_length(self):
        p = "O:/Cloud/Projects/jets/c130j/sourceimages/textures/c130j_body_DIFF.png"
        for n in range(8, 80):
            self.assertLessEqual(len(StrUtils.truncate(p, n, "path", "…")), n)

    def test_truncate_none_input(self):
        """Test truncate with None input."""
        self.assertIsNone(StrUtils.truncate(None, 4))

    def test_truncate_no_truncation_needed(self):
        """Test truncate when string is shorter than limit."""
        self.assertEqual(StrUtils.truncate("hi", 10), "hi")

    def test_truncate_empty_string(self):
        """Test truncate with empty string."""
        self.assertEqual(StrUtils.truncate("", 10), "")

    def test_truncate_exact_length(self):
        """Test truncate when string equals limit."""
        self.assertEqual(StrUtils.truncate("1234", 4), "1234")

    def test_truncate_unicode(self):
        """Test truncate with unicode characters."""
        result = StrUtils.truncate("αβγδεζηθ", 4)
        self.assertEqual(len(result), 6)  # 4 chars + 2 for ..

    def test_truncate_very_long_string(self):
        """Test truncate with very long string."""
        long_str = "a" * 1000
        result = StrUtils.truncate(long_str, 10)
        self.assertEqual(len(result), 12)  # 10 chars + 2 for ..

    # -------------------------------------------------------------------------
    # get_trailing_integers Tests
    # -------------------------------------------------------------------------

    def test_get_trailing_integers_basic(self):
        """Test get_trailing_integers extracts numbers from end of string."""
        self.assertEqual(StrUtils.get_trailing_integers("p001Cube1"), 1)

    def test_get_trailing_integers_as_string(self):
        """Test get_trailing_integers returning string."""
        self.assertEqual(StrUtils.get_trailing_integers("p001Cube1", 0, True), "1")

    def test_get_trailing_integers_increment(self):
        """Test get_trailing_integers with increment."""
        self.assertEqual(StrUtils.get_trailing_integers("p001Cube1", 1), 2)

    def test_get_trailing_integers_none_input(self):
        """Test get_trailing_integers with None input."""
        self.assertIsNone(StrUtils.get_trailing_integers(None))

    def test_get_trailing_integers_no_numbers(self):
        """Test get_trailing_integers with no trailing numbers."""
        result = StrUtils.get_trailing_integers("Cube")
        self.assertIsNone(result)

    def test_get_trailing_integers_as_string_no_numbers(self):
        """as_string=True must return None (not the string 'None') on no match."""
        self.assertIsNone(StrUtils.get_trailing_integers("Cube", as_string=True))

    def test_get_trailing_integers_multi_digit(self):
        """Test get_trailing_integers with multi-digit number."""
        self.assertEqual(StrUtils.get_trailing_integers("object123"), 123)

    # -------------------------------------------------------------------------
    # time_stamp Tests
    # -------------------------------------------------------------------------

    def test_time_stamp_round_trip(self):
        """Attaching then detaching a timestamp must return the original path,
        including paths that contain spaces."""
        import tempfile
        import os

        tmpdir = tempfile.mkdtemp(prefix="time stamp ")  # space in dir name
        path = os.path.join(tmpdir, "auto save.0001.mb")
        try:
            with open(path, "w") as f:
                f.write("x")
            normalized = path.replace("\\", "/")

            stamped = StrUtils.time_stamp(path)
            self.assertNotEqual(stamped, normalized)
            self.assertTrue(stamped.endswith(normalized))

            detached = StrUtils.time_stamp(stamped)
            self.assertEqual(detached, normalized)
        finally:
            os.remove(path)
            os.rmdir(tmpdir)

    def test_get_trailing_integers_zero(self):
        """Test get_trailing_integers with trailing zero."""
        self.assertEqual(StrUtils.get_trailing_integers("item0"), 0)

    def test_get_trailing_integers_leading_zeros(self):
        """Test get_trailing_integers with leading zeros - zeros are NOT preserved."""
        # Note: The implementation uses int() internally, so leading zeros are lost
        self.assertEqual(StrUtils.get_trailing_integers("item007", 0, True), "7")

    def test_get_trailing_integers_empty_string(self):
        """Test get_trailing_integers with empty string returns empty string."""
        result = StrUtils.get_trailing_integers("")
        self.assertEqual(result, "")

    # -------------------------------------------------------------------------
    # find_str Tests
    # -------------------------------------------------------------------------

    def test_find_str_wildcard(self):
        """Test find_str with wildcard patterns."""
        lst = [
            "invertVertexWeights",
            "keepCreaseEdgeWeight",
            "keepBorder",
            "keepBorderWeight",
            "keepColorBorder",
            "keepColorBorderWeight",
        ]
        expected = [
            "invertVertexWeights",
            "keepCreaseEdgeWeight",
            "keepBorderWeight",
            "keepColorBorderWeight",
        ]
        self.assertEqual(StrUtils.find_str("*Weight*", lst), expected)

    def test_find_str_regex(self):
        """Test find_str with regex patterns."""
        lst = [
            "invertVertexWeights",
            "keepCreaseEdgeWeight",
            "keepBorder",
            "keepBorderWeight",
        ]
        expected = [
            "invertVertexWeights",
            "keepCreaseEdgeWeight",
            "keepBorderWeight",
        ]
        self.assertEqual(
            StrUtils.find_str("Weight$|Weights$", lst, regex=True), expected
        )

    def test_find_str_case_insensitive(self):
        """Test find_str with case insensitive matching."""
        lst = ["Weight", "WEIGHT", "weight"]
        result = StrUtils.find_str("*weight*", lst, False, True)
        self.assertEqual(len(result), 3)

    def test_find_str_multiple_patterns(self):
        """Test find_str with multiple patterns."""
        lst = ["invertVertexWeights", "keepBorderWeight"]
        self.assertEqual(
            StrUtils.find_str("*Weights|*Weight", lst),
            ["invertVertexWeights", "keepBorderWeight"],
        )

    def test_find_str_empty_list(self):
        """Test find_str with empty list."""
        self.assertEqual(StrUtils.find_str("*test*", []), [])

    def test_find_str_no_matches(self):
        """Test find_str when nothing matches."""
        self.assertEqual(StrUtils.find_str("xyz", ["abc", "def"]), [])

    def test_find_str_all_match(self):
        """Test find_str when all match."""
        self.assertEqual(StrUtils.find_str("*", ["a", "b", "c"]), ["a", "b", "c"])

    # -------------------------------------------------------------------------
    # find_str_and_format Tests
    # -------------------------------------------------------------------------

    def test_find_str_and_format_remove_pattern(self):
        """Test find_str_and_format to remove matched pattern."""
        lst = ["invertVertexWeights"]
        self.assertEqual(
            StrUtils.find_str_and_format(lst, "", "*Weights"), ["invertVertex"]
        )

    def test_find_str_and_format_replace(self):
        """Test find_str_and_format with replacement."""
        lst = ["invertVertexWeights"]
        self.assertEqual(
            StrUtils.find_str_and_format(lst, "new name", "*Weights"), ["new name"]
        )

    def test_find_str_and_format_insert(self):
        """Test find_str_and_format with insert pattern."""
        lst = ["invertVertexWeights"]
        self.assertEqual(
            StrUtils.find_str_and_format(lst, "*insert*", "*Weights"),
            ["invertVertexinsert"],
        )

    def test_find_str_and_format_suffix(self):
        """Test find_str_and_format adding suffix."""
        lst = ["invertVertexWeights"]
        self.assertEqual(
            StrUtils.find_str_and_format(lst, "*_suffix", "*Weights"),
            ["invertVertex_suffix"],
        )
        self.assertEqual(
            StrUtils.find_str_and_format(lst, "**_suffix", "*Weights"),
            ["invertVertexWeights_suffix"],
        )

    def test_find_str_and_format_prefix(self):
        """Test find_str_and_format adding prefix."""
        lst = ["invertVertexWeights"]
        self.assertEqual(
            StrUtils.find_str_and_format(lst, "prefix_*", "*Weights"),
            ["prefix_"],
        )
        self.assertEqual(
            StrUtils.find_str_and_format(lst, "prefix_**", "*Weights"),
            ["prefix_invertVertexWeights"],
        )

    def test_find_str_and_format_with_original(self):
        """Test find_str_and_format with return_originals."""
        lst = ["invertVertexWeights"]
        self.assertEqual(
            StrUtils.find_str_and_format(
                lst, "new name", "*weights", False, True, True
            ),
            [("invertVertexWeights", "new name")],
        )

    def test_find_str_and_format_empty_list(self):
        """Test find_str_and_format with empty list."""
        self.assertEqual(StrUtils.find_str_and_format([], "new", "*old*"), [])

    # -------------------------------------------------------------------------
    # find_str_and_format - multi-term filters (per-term 'from')
    # -------------------------------------------------------------------------

    def test_find_str_and_format_multi_term_filter(self):
        """A '|' filter must format each string against the term that matched it.

        Regression: 'frm_ = fltr.strip("*")' collapsed the whole filter into one
        literal ('pCube*|nurbs'), which is never a substring of any name -- so
        matched strings silently came back unformatted.
        """
        lst = ["pCube1", "nurbsSphere1"]
        self.assertEqual(
            StrUtils.find_str_and_format(lst, "*box*", "pCube*|nurbs*"),
            ["box1", "boxSphere1"],
        )

    def test_find_str_and_format_multi_term_filter_strip(self):
        """Strip mode must also resolve 'from' per matching filter term."""
        lst = ["pCube1", "nurbsSphere1"]
        self.assertEqual(
            StrUtils.find_str_and_format(lst, "", "*Cube*|*Sphere*"),
            ["p1", "nurbs1"],
        )

    def test_find_str_and_format_preserves_duplicate_inputs(self):
        """A filtered format returns one entry per input, duplicates included.

        Regression: filtering delegated to find_str, which dedupes -- so two
        objects sharing a short name collapsed to a single rename pair and only
        the first of them was ever renamed.
        """
        self.assertEqual(
            StrUtils.find_str_and_format(["pCube1", "pCube1"], "*box*", "*Cube*"),
            ["pbox1", "pbox1"],
        )

    def test_find_str_and_format_preserves_input_order(self):
        """Results stay parallel to the surviving inputs, duplicates included."""
        self.assertEqual(
            StrUtils.find_str_and_format(
                ["arm_R", "arm_L", "arm_R"], "*_lt|*_rt", "*_L|*_R"
            ),
            ["arm_rt", "arm_lt", "arm_rt"],
        )

    def test_find_str_and_format_term_pairing(self):
        """'|' terms in 'to' pair positionally with '|' terms in the filter."""
        lst = ["arm_L", "arm_R"]
        self.assertEqual(
            StrUtils.find_str_and_format(lst, "*_lt|*_rt", "*_L|*_R"),
            ["arm_lt", "arm_rt"],
        )

    def test_find_str_and_format_term_pairing_order_independent(self):
        """Pairing follows the matching term, not the input ordering."""
        lst = ["arm_R", "arm_L"]
        self.assertEqual(
            StrUtils.find_str_and_format(lst, "*_lt|*_rt", "*_L|*_R"),
            ["arm_rt", "arm_lt"],
        )

    def test_find_str_and_format_single_to_term_applies_to_all(self):
        """One 'to' term against many filter terms applies to every match."""
        lst = ["pCube1", "nurbsSphere1"]
        self.assertEqual(
            StrUtils.find_str_and_format(lst, "**_GEO", "pCube*|nurbs*"),
            ["pCube1_GEO", "nurbsSphere1_GEO"],
        )

    def test_find_str_and_format_pipe_literal_without_filter_terms(self):
        """'|' in 'to' stays literal when the filter has no terms to pair with."""
        self.assertEqual(
            StrUtils.find_str_and_format(["pCube1"], "a|b", "*Cube*"), ["a|b"]
        )

    # -------------------------------------------------------------------------
    # find_str_and_format - regex mode
    # -------------------------------------------------------------------------

    def test_find_str_and_format_regex_replace_chars(self):
        """Regex mode must substitute via the compiled pattern, not a literal.

        Regression: the regex source was used as a plain substring, so the
        filter selected the right strings and then formatting was a no-op.
        """
        self.assertEqual(
            StrUtils.find_str_and_format(
                ["pCube1", "pCube2"], "*box*", r"pCube\d+", regex=True
            ),
            ["box", "box"],
        )

    def test_find_str_and_format_regex_replace_suffix(self):
        """Regex replace-suffix must cut at the match, not blindly append."""
        self.assertEqual(
            StrUtils.find_str_and_format(["pCube1"], "*_GEO", r"Cube.*", regex=True),
            ["p_GEO"],
        )

    def test_find_str_and_format_regex_replace_prefix(self):
        """Regex replace-prefix must cut at the end of the match."""
        self.assertEqual(
            StrUtils.find_str_and_format(["pCube1"], "L_*", r"^pCube", regex=True),
            ["L_1"],
        )

    def test_find_str_and_format_regex_strip(self):
        """Regex strip must remove the matched span."""
        self.assertEqual(
            StrUtils.find_str_and_format(["pCube1"], "", r"Cube", regex=True), ["p1"]
        )

    def test_find_str_and_format_regex_preserves_asterisks(self):
        """A regex filter must not be mangled by asterisk stripping.

        Regression: 'fltr.strip("*")' turned '.*Cube.*' into '.*Cube.'.
        """
        self.assertEqual(
            StrUtils.find_str_and_format(["pCube1"], "*box*", r".*Cube.*", regex=True),
            ["box"],
        )

    def test_find_str_and_format_regex_ignore_case(self):
        """ignore_case must reach the substitution in regex mode too."""
        self.assertEqual(
            StrUtils.find_str_and_format(
                ["PCUBE1"], "*box*", "pcube", regex=True, ignore_case=True
            ),
            ["box1"],
        )

    def test_find_str_and_format_regex_alternation_is_not_a_term_split(self):
        """In regex mode '|' stays alternation; it is not a term separator."""
        self.assertEqual(
            StrUtils.find_str_and_format(
                ["arm_L", "arm_R"], "*_x", r"_L|_R", regex=True
            ),
            ["arm_x", "arm_x"],
        )

    # -------------------------------------------------------------------------
    # find_str_and_format - capture-group backrefs (regex mode only)
    # -------------------------------------------------------------------------

    def test_find_str_and_format_backrefs_replace_chars(self):
        """'\\1'/'\\2' in 'to' expand to filter capture groups."""
        self.assertEqual(
            StrUtils.find_str_and_format(
                ["pCube1", "pCube2"], r"*\1_box_\2*", r"(p)Cube(\d+)", regex=True
            ),
            ["p_box_1", "p_box_2"],
        )

    def test_find_str_and_format_backrefs_append_suffix(self):
        """Backrefs expand in append/replace modes, not just substitution."""
        self.assertEqual(
            StrUtils.find_str_and_format(
                ["pCube1"], r"**_\1", r"Cube(\d+)", regex=True
            ),
            ["pCube1_1"],
        )

    def test_find_str_and_format_backrefs_named_group(self):
        """Named groups expand via the standard '\\g<name>' form."""
        self.assertEqual(
            StrUtils.find_str_and_format(
                ["pCube1"], r"*\g<idx>_GEO", r"Cube(?P<idx>\d+)", regex=True
            ),
            ["p1_GEO"],
        )

    def test_find_str_and_format_backrefs_not_expanded_without_regex(self):
        """In wildcard mode a backslash sequence stays literal."""
        self.assertEqual(
            StrUtils.find_str_and_format(["pCube1"], r"*a\1b*", "*Cube*"), [r"pa\1b1"]
        )

    def test_find_str_and_format_invalid_backref_falls_back_to_literal(self):
        """An unusable escape in 'to' must not raise; it is used verbatim."""
        self.assertEqual(
            StrUtils.find_str_and_format(["pCube1"], r"*bo\dx*", r"Cube", regex=True),
            [r"pbo\dx1"],
        )

    def test_find_str_and_format_unknown_group_name_falls_back_to_literal(self):
        """A named backref with no matching group must not raise either.

        re.sub raises IndexError (not re.error) for an unknown group *name*, so
        catching only re.error let it escape.
        """
        self.assertEqual(
            StrUtils.find_str_and_format(
                ["pCube1"], r"*\g<nope>*", r"Cube", regex=True
            ),
            [r"p\g<nope>1"],
        )

    def test_find_str_and_format_unknown_group_number_falls_back_to_literal(self):
        """Same for a numbered backref past the pattern's group count."""
        self.assertEqual(
            StrUtils.find_str_and_format(["pCube1"], r"**_\3", r"Cube(\d)", regex=True),
            [r"pCube1_\3"],
        )

    # -------------------------------------------------------------------------
    # find_str_and_format - degenerate patterns
    # -------------------------------------------------------------------------

    def test_find_str_and_format_double_asterisk_is_noop(self):
        """'**' appends an empty suffix -- it must not strip the match.

        Regression: the replace_chars branch caught all-asterisk patterns first,
        so '**' silently deleted the matched token.
        """
        self.assertEqual(
            StrUtils.find_str_and_format(["pCube1"], "**", "*Cube*"), ["pCube1"]
        )
        self.assertEqual(
            StrUtils.find_str_and_format(["pCube1"], "***", "*Cube*"), ["pCube1"]
        )

    def test_find_str_and_format_single_asterisk_truncates_at_match(self):
        """'*' is replace-suffix with an empty payload."""
        self.assertEqual(
            StrUtils.find_str_and_format(["pCube1"], "*", "*Cube*"), ["p"]
        )

    # -------------------------------------------------------------------------
    # strip_suffix Tests
    # -------------------------------------------------------------------------

    def test_strip_suffix_removes_only_a_listed_suffix(self):
        exts = [".fbx", ".usd", ".usda"]
        self.assertEqual(StrUtils.strip_suffix("asset.fbx", exts), "asset")
        self.assertEqual(StrUtils.strip_suffix("asset.FBX", exts), "asset")
        self.assertEqual(StrUtils.strip_suffix("asset.usda", exts), "asset")
        # A dotted token that is not a listed suffix is part of the name.
        self.assertEqual(StrUtils.strip_suffix("asset.v2", exts), "asset.v2")
        self.assertEqual(StrUtils.strip_suffix("asset", exts), "asset")
        self.assertEqual(StrUtils.strip_suffix("asset.fbx", []), "asset.fbx")

    def test_strip_suffix_strips_once_longest_first(self):
        self.assertEqual(StrUtils.strip_suffix("a.fbx.fbx", [".fbx"]), "a.fbx")
        self.assertEqual(StrUtils.strip_suffix("a.usda", [".usd", ".usda"]), "a")

    # -------------------------------------------------------------------------
    # retain_suffix Tests
    # -------------------------------------------------------------------------

    def test_retain_suffix_replaces_recognized_new_suffix(self):
        valid = ["_GRP", "_LOC", "_GEO"]
        self.assertEqual(
            StrUtils.retain_suffix("S00B6_TAG_GRP", "S00B8_TAG_LOC", valid),
            "S00B8_TAG_GRP",
        )
        self.assertEqual(
            StrUtils.retain_suffix("S00B6_TAG_LOC", "S00B8_TAG_LOC", valid),
            "S00B8_TAG_LOC",
        )

    def test_retain_suffix_strips_trailing_digits(self):
        valid = ["_GRP", "_LOC"]
        self.assertEqual(
            StrUtils.retain_suffix("Asset_GRP2", "NewAsset_LOC", valid), "NewAsset_GRP"
        )

    def test_retain_suffix_unknown_new_suffix_kept(self):
        self.assertEqual(
            StrUtils.retain_suffix("Part_GEO", "Detail_HIGH", ["_GEO"]),
            "Detail_HIGH_GEO",
        )

    def test_retain_suffix_any_suffix_when_unrestricted(self):
        self.assertEqual(StrUtils.retain_suffix("Foo_GRP", "Bar_GEO"), "Bar_GRP")
        self.assertEqual(StrUtils.retain_suffix("Foo_GRP", "Bar"), "Bar_GRP")

    def test_retain_suffix_numeric_token_is_not_a_suffix(self):
        self.assertEqual(StrUtils.retain_suffix("Screw_01", "Bolt"), "Bolt")

    def test_retain_suffix_no_suffix_or_not_valid(self):
        self.assertEqual(StrUtils.retain_suffix("Screw", "Bolt"), "Bolt")
        self.assertEqual(StrUtils.retain_suffix("Screw_ABC", "Bolt", ["_GRP"]), "Bolt")

    # -------------------------------------------------------------------------
    # format_suffix Tests
    # -------------------------------------------------------------------------

    def test_format_suffix_basic(self):
        """Test format_suffix adds suffixes correctly."""
        self.assertEqual(
            StrUtils.format_suffix("p001Cube1", "_suffix", "Cube1"),
            "p001_suffix",
        )

    def test_format_suffix_list_strip(self):
        """Test format_suffix with list of strings to strip."""
        self.assertEqual(
            StrUtils.format_suffix("p001Cube1", "_suffix", ["Cu", "be1"]),
            "p001_suffix",
        )

    def test_format_suffix_strip_trailing(self):
        """Test format_suffix with strip_trailing option."""
        self.assertEqual(
            StrUtils.format_suffix("p001Cube1", "_suffix", "", True),
            "p001Cube_suffix",
        )

    def test_format_suffix_strip_chars(self):
        """Test format_suffix with strip_chars option."""
        self.assertEqual(
            StrUtils.format_suffix("pCube_GEO1", "_suffix", "", True, True),
            "pCube_suffix",
        )

    def test_format_suffix_empty_suffix(self):
        """Test format_suffix with empty suffix."""
        self.assertEqual(
            StrUtils.format_suffix("test", "", ""),
            "test",
        )

    def test_format_suffix_no_strip(self):
        """Test format_suffix with nothing to strip."""
        self.assertEqual(
            StrUtils.format_suffix("hello", "_end", "xyz"),
            "hello_end",
        )

    def test_format_suffix_strip_trailing_ints_preserves_underscore_separated(self):
        """Verify strip_trailing_ints only strips digits directly appended to the
        name (e.g. CHECKLIST01 -> CHECKLIST) and preserves intentional underscore-
        separated numbering (e.g. CHECKLIST_01 stays CHECKLIST_01).

        Bug: regex r'\\d+$' stripped digits regardless of preceding underscore.
        Fixed: 2026-03-02
        """
        # Digits directly appended — should be stripped
        self.assertEqual(
            StrUtils.format_suffix("CHECKLIST01", "", "", strip_trailing_ints=True),
            "CHECKLIST",
        )
        # Underscore-separated digits — should be preserved
        self.assertEqual(
            StrUtils.format_suffix("CHECKLIST_01", "", "", strip_trailing_ints=True),
            "CHECKLIST_01",
        )
        # Single digit directly appended
        self.assertEqual(
            StrUtils.format_suffix("pCube1", "", "", strip_trailing_ints=True),
            "pCube",
        )
        # Single digit after underscore — preserved
        self.assertEqual(
            StrUtils.format_suffix("pCube_1", "", "", strip_trailing_ints=True),
            "pCube_1",
        )

    # -------------------------------------------------------------------------
    # alpha_sequence Tests
    # -------------------------------------------------------------------------

    def test_alpha_sequence_single_letter(self):
        """Indices 0-25 produce A-Z."""
        self.assertEqual(StrUtils.alpha_sequence(0), "A")
        self.assertEqual(StrUtils.alpha_sequence(1), "B")
        self.assertEqual(StrUtils.alpha_sequence(25), "Z")

    def test_alpha_sequence_double_letter(self):
        """Indices 26+ wrap to AA, AB, ..., AZ, BA."""
        self.assertEqual(StrUtils.alpha_sequence(26), "AA")
        self.assertEqual(StrUtils.alpha_sequence(27), "AB")
        self.assertEqual(StrUtils.alpha_sequence(51), "AZ")
        self.assertEqual(StrUtils.alpha_sequence(52), "BA")

    def test_alpha_sequence_triple_letter(self):
        """Index 702 wraps to AAA (after ZZ at 701)."""
        self.assertEqual(StrUtils.alpha_sequence(701), "ZZ")
        self.assertEqual(StrUtils.alpha_sequence(702), "AAA")

    def test_alpha_sequence_negative_raises(self):
        """Negative indices raise ValueError."""
        with self.assertRaises(ValueError):
            StrUtils.alpha_sequence(-1)

    # -------------------------------------------------------------------------
    # sequential_suffixes Tests
    # -------------------------------------------------------------------------

    def test_sequential_suffixes_letters_under_threshold(self):
        """Counts at or below the switch threshold return uppercase letters."""
        self.assertEqual(StrUtils.sequential_suffixes(0), [])
        self.assertEqual(StrUtils.sequential_suffixes(3), ["A", "B", "C"])
        self.assertEqual(
            StrUtils.sequential_suffixes(26),
            [chr(ord("A") + i) for i in range(26)],
        )

    def test_sequential_suffixes_numeric_above_threshold(self):
        """Counts above the threshold fall back to zero-padded numerics."""
        out = StrUtils.sequential_suffixes(27)
        self.assertEqual(len(out), 27)
        self.assertEqual(out[0], "01")
        self.assertEqual(out[26], "27")
        # Padding widens to match the count's digit length.
        out_120 = StrUtils.sequential_suffixes(120)
        self.assertEqual(out_120[0], "001")
        self.assertEqual(out_120[-1], "120")

    def test_sequential_suffixes_lowercase(self):
        """``lowercase=True`` returns lowercase letters in the letter scheme."""
        self.assertEqual(StrUtils.sequential_suffixes(3, lowercase=True), ["a", "b", "c"])
        # Lowercase is irrelevant once we're in numeric mode.
        out = StrUtils.sequential_suffixes(40, lowercase=True)
        self.assertTrue(all(s.isdigit() for s in out))

    def test_sequential_suffixes_custom_switch_at(self):
        """``switch_at`` lets callers force the numeric branch earlier."""
        self.assertEqual(StrUtils.sequential_suffixes(5, switch_at=3)[:3], ["01", "02", "03"])

    # -------------------------------------------------------------------------
    # resolve_name_collisions Tests
    # -------------------------------------------------------------------------

    def test_resolve_collisions_alpha_basic(self):
        """Three colliding mats get _A, _B, _C; lone wood3 just strips to wood."""
        result = StrUtils.resolve_name_collisions(
            ["mat", "mat1", "mat2", "wood3"],
            strip_trailing_ints=True,
            collision_suffix="alpha",
        )
        self.assertEqual(
            result,
            {"mat": "mat_A", "mat1": "mat_B", "mat2": "mat_C", "wood3": "wood"},
        )

    def test_resolve_collisions_single_member_strips_to_base(self):
        """A non-colliding name still strips to base regardless of suffix scheme."""
        result = StrUtils.resolve_name_collisions(
            ["mat3"], strip_trailing_ints=True, collision_suffix="alpha"
        )
        self.assertEqual(result, {"mat3": "mat"})

    def test_resolve_collisions_no_change_omitted(self):
        """A name already at its target base does not appear in the result."""
        result = StrUtils.resolve_name_collisions(
            ["mat"], strip_trailing_ints=True, collision_suffix="alpha"
        )
        self.assertEqual(result, {})

    def test_resolve_collisions_none_keeps_originals(self):
        """collision_suffix=None leaves multi-member groups unchanged."""
        result = StrUtils.resolve_name_collisions(
            ["mat", "mat1"], strip_trailing_ints=True, collision_suffix=None
        )
        self.assertEqual(result, {})

    def test_resolve_collisions_none_still_strips_singletons(self):
        """Even with collision_suffix=None, single-member groups strip to base."""
        result = StrUtils.resolve_name_collisions(
            ["wood3"], strip_trailing_ints=True, collision_suffix=None
        )
        self.assertEqual(result, {"wood3": "wood"})

    def test_resolve_collisions_numeric(self):
        """Numeric scheme zero-pads to width = max(2, len(str(count)))."""
        result = StrUtils.resolve_name_collisions(
            ["mat", "mat1", "mat2"],
            strip_trailing_ints=True,
            collision_suffix="numeric",
        )
        self.assertEqual(
            result, {"mat": "mat_01", "mat1": "mat_02", "mat2": "mat_03"}
        )

    def test_resolve_collisions_numeric_pads_for_large_groups(self):
        """100+ members -> 3-digit padding."""
        names = [f"x{i}" if i else "x" for i in range(100)]
        result = StrUtils.resolve_name_collisions(
            names, strip_trailing_ints=True, collision_suffix="numeric"
        )
        self.assertEqual(result["x"], "x_001")
        self.assertEqual(result["x99"], "x_100")

    def test_resolve_collisions_callable_scheme(self):
        """Custom callable suffix(i, count) is honored."""
        result = StrUtils.resolve_name_collisions(
            ["mat", "mat1"],
            strip_trailing_ints=True,
            collision_suffix=lambda i, count: f"v{i}",
        )
        self.assertEqual(result, {"mat": "mat_v0", "mat1": "mat_v1"})

    def test_resolve_collisions_preserves_input_order(self):
        """Within a group, suffixes are assigned in input order."""
        result = StrUtils.resolve_name_collisions(
            ["mat2", "mat", "mat1"],
            strip_trailing_ints=True,
            collision_suffix="alpha",
        )
        self.assertEqual(
            result, {"mat2": "mat_A", "mat": "mat_B", "mat1": "mat_C"}
        )

    def test_resolve_collisions_empty_base_skipped(self):
        """Names that strip to empty are omitted from grouping."""
        result = StrUtils.resolve_name_collisions(
            ["123", "mat", "mat1"],
            strip_trailing_ints=True,
            collision_suffix="alpha",
        )
        self.assertNotIn("123", result)
        self.assertEqual(result, {"mat": "mat_A", "mat1": "mat_B"})

    def test_resolve_collisions_custom_separator(self):
        """suffix_separator is used between base and suffix."""
        result = StrUtils.resolve_name_collisions(
            ["mat", "mat1"],
            strip_trailing_ints=True,
            collision_suffix="alpha",
            suffix_separator="-",
        )
        self.assertEqual(result, {"mat": "mat-A", "mat1": "mat-B"})

    def test_resolve_collisions_alpha_27_members(self):
        """Group of 27 wraps from Z to AA."""
        names = [f"x{i}" if i else "x" for i in range(27)]
        result = StrUtils.resolve_name_collisions(
            names, strip_trailing_ints=True, collision_suffix="alpha"
        )
        self.assertEqual(result["x"], "x_A")
        self.assertEqual(result["x25"], "x_Z")
        self.assertEqual(result["x26"], "x_AA")

    # -------------------------------------------------------------------------
    # replace_placeholders Tests
    # -------------------------------------------------------------------------

    def test_replace_placeholders_basic(self):
        self.assertEqual(
            StrUtils.replace_placeholders("{a}_{b}", a="x", b="y"), "x_y"
        )

    def test_replace_placeholders_format_spec(self):
        self.assertEqual(
            StrUtils.replace_placeholders("v{n:03d}", n=5), "v005"
        )

    def test_replace_placeholders_missing_preserves_placeholder(self):
        self.assertEqual(
            StrUtils.replace_placeholders("{a}_{b}", a="x"), "x_{b}"
        )

    def test_replace_placeholders_missing_preserves_format_spec(self):
        # The bug fixed in SafeFormatter.format_field: unresolved {n:03d}
        # used to collapse to {n}, losing padding for a second pass.
        self.assertEqual(
            StrUtils.replace_placeholders("{stem}_v{n:03d}", stem="shot"),
            "shot_v{n:03d}",
        )
        self.assertEqual(
            StrUtils.replace_placeholders("{user}_{stem}_v{n:03d}", user="maya"),
            "maya_{stem}_v{n:03d}",
        )

    def test_replace_placeholders_two_stage_substitution(self):
        # Stage 1 leaves {n:03d} intact; stage 2 applies the spec.
        stage1 = StrUtils.replace_placeholders("{user}_{stem}_v{n:03d}", user="m")
        self.assertEqual(stage1.format(stem="shot", n=7), "m_shot_v007")

    def test_replace_placeholders_other_format_specs_preserved(self):
        self.assertEqual(
            StrUtils.replace_placeholders("{key:>10}"), "{key:>10}"
        )
        self.assertEqual(
            StrUtils.replace_placeholders("{key:.4f}"), "{key:.4f}"
        )

    # -------------------------------------------------------------------------
    # resolve_placeholders Tests
    # -------------------------------------------------------------------------

    def test_resolve_placeholders_reports_resolved_and_unresolved(self):
        info = StrUtils.resolve_placeholders(
            "{root}/{name}_{ver:03d}", root="C:/p", name="shot"
        )
        # Unresolved tokens keep their format spec verbatim (see replace_placeholders).
        self.assertEqual(info["result"], "C:/p/shot_{ver:03d}")
        self.assertEqual(info["fields"], ["root", "name", "ver"])
        self.assertEqual(info["resolved"], {"root": "C:/p", "name": "shot"})
        self.assertEqual(info["unresolved"], ["ver"])

    def test_resolve_placeholders_result_matches_replace_placeholders(self):
        """The 'result' key must be byte-identical to replace_placeholders."""
        tmpl = "{scenes}/{name}/{missing}"
        info = StrUtils.resolve_placeholders(tmpl, scenes="scenes", name="shot")
        self.assertEqual(
            info["result"],
            StrUtils.replace_placeholders(tmpl, scenes="scenes", name="shot"),
        )

    def test_resolve_placeholders_first_seen_order_and_dedup(self):
        """Fields are first-seen order; repeats collapse to one entry."""
        info = StrUtils.resolve_placeholders("{b}{a}{b}{c}", a="1")
        self.assertEqual(info["fields"], ["b", "a", "c"])
        self.assertEqual(info["unresolved"], ["b", "c"])
        self.assertEqual(info["resolved"], {"a": "1"})

    def test_resolve_placeholders_index_access_reports_base_name(self):
        """Index access reduces to the base name in the reported fields."""
        info = StrUtils.resolve_placeholders("{parts[0]}", parts=["a", "b"])
        self.assertEqual(info["fields"], ["parts"])
        self.assertIn("parts", info["resolved"])

    def test_resolve_placeholders_stringifies_values(self):
        info = StrUtils.resolve_placeholders("{n}", n=5)
        self.assertEqual(info["resolved"], {"n": "5"})

    def test_resolve_placeholders_empty_template(self):
        info = StrUtils.resolve_placeholders("no placeholders here")
        self.assertEqual(info["fields"], [])
        self.assertEqual(info["unresolved"], [])
        self.assertEqual(info["result"], "no placeholders here")

    # -------------------------------------------------------------------------
    # Regression tests (audit fixes)
    # -------------------------------------------------------------------------

    def test_insert_negative_occurrence_inserts_at_last(self):
        """insert with a negative occurrence must locate the Nth match from the
        right and insert there. Previously it replaced the delimiter with spaces
        (a no-op only when the delimiter itself was a space) so any non-space
        delimiter silently no-oped."""
        self.assertEqual(StrUtils.insert("a.b.c.d", "X", ".", occurrence=-1), "a.b.c.Xd")
        # -2 -> second-from-last dot (position 3), insert after it.
        self.assertEqual(StrUtils.insert("a.b.c.d", "X", ".", occurrence=-2), "a.b.Xc.d")
        # before=True places the insert ahead of the matched delimiter.
        self.assertEqual(
            StrUtils.insert("a.b.c.d", "X", ".", occurrence=-1, before=True),
            "a.b.cX.d",
        )

    def test_insert_positive_occurrence_by_position(self):
        """Positive occurrence is 1-based and selects the Nth match from the left."""
        self.assertEqual(StrUtils.insert("a.b.c.d", "X", ".", occurrence=1), "a.Xb.c.d")
        self.assertEqual(StrUtils.insert("a.b.c.d", "X", ".", occurrence=2), "a.b.Xc.d")
        # Out-of-range occurrence returns the source unchanged.
        self.assertEqual(StrUtils.insert("a.b.c.d", "X", ".", occurrence=9), "a.b.c.d")

    def test_set_case_none_returns_string_unchanged(self):
        """set_case(case=None) is the documented 'no transform': the string comes
        back unchanged instead of raising TypeError from getattr(string, None)."""
        self.assertEqual(StrUtils.set_case("fooBar", case=None), "fooBar")
        self.assertEqual(StrUtils.set_case(["fooBar", "baz"], case=None), ["fooBar", "baz"])

    def test_find_str_and_format_replace_prefix_drops_matched_prefix(self):
        """Case-sensitive replace_prefix must drop the matched prefix, matching the
        ignore_case branch's behavior (previously it re-embedded the old prefix)."""
        self.assertEqual(
            StrUtils.find_str_and_format(["oldSuffix"], to="new*", fltr="old*"),
            ["newSuffix"],
        )
        # Parity with the ignore_case path on the same input.
        self.assertEqual(
            StrUtils.find_str_and_format(
                ["oldSuffix"], to="new*", fltr="old*", ignore_case=True
            ),
            ["newSuffix"],
        )

    def test_get_matching_hierarchy_items_multichar_string_delimiter(self):
        """A multi-char delimiter string (each char a delimiter) must split the
        target the same way it splits the items; previously the target was left
        unsplit so nothing matched."""
        items = ["a.b.c", "a.b", "a"]
        self.assertEqual(
            StrUtils.get_matching_hierarchy_items(
                items, "a.b", downstream=True, delimiters=".|"
            ),
            ["a.b.c"],
        )
        # The equivalent list form yields the same result.
        self.assertEqual(
            StrUtils.get_matching_hierarchy_items(
                items, "a.b", downstream=True, delimiters=[".", "|"]
            ),
            ["a.b.c"],
        )


if __name__ == "__main__":
    unittest.main(exit=False)
