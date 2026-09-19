#!/usr/bin/python
# coding=utf-8
"""Unit tests for pythontk Deprecation.

Run with:
    python -m pytest test_deprecation.py -v
    python test_deprecation.py
"""

import importlib
import inspect
import pkgutil
import sys
import types
import unittest
import warnings

import pythontk as ptk
from pythontk.core_utils.deprecation import Deprecation, DeprecationRecord

from conftest import BaseTestCase


class DeprecationRecordTest(BaseTestCase):
    """The record, its message, and the validation that makes a deadline real."""

    def test_message_names_what_when_and_replacement(self):
        record = DeprecationRecord(
            what="FileUtils.set_json",
            replacement="FileUtils.write_json",
            remove_in="0.11.0",
            module="pythontk.file_utils._file_utils",
        )
        message = record.message()
        self.assertIn("FileUtils.set_json", message)
        self.assertIn("FileUtils.write_json", message)
        self.assertIn("pythontk 0.11.0", message)

    def test_reason_is_appended(self):
        record = DeprecationRecord("A", "B", "1.0.0", reason="It never worked.")
        self.assertTrue(record.message().endswith("It never worked."))

    def test_replacement_is_mandatory(self):
        """A deprecation nobody can act on is noise -- so it cannot be built.

        The rule was previously a convention that one test asserted for one
        cluster; here it is the constructor's job, for every shape.
        """
        with self.assertRaises(ValueError) as caught:
            DeprecationRecord("A", "   ", "1.0.0")
        self.assertIn("replacement", str(caught.exception))

    def test_what_is_mandatory(self):
        with self.assertRaises(ValueError):
            DeprecationRecord("", "B", "1.0.0")

    def test_unparseable_version_raises_at_construction(self):
        """The failure this module exists to prevent, prevented at the source.

        ``"next release"`` is what every hand-rolled notice in the ecosystem
        said, and it is exactly what nothing could ever check.
        """
        for bad in ("next release", "", "0", "1.2.3.4", "v1.2.3", "1.2.3a"):
            with self.subTest(remove_in=bad):
                with self.assertRaises(ValueError):
                    DeprecationRecord("A", "B", bad)

    def test_non_ascii_digits_are_not_a_version(self):
        """``\\d`` matches Devanagari digits; a version key built from them
        would compare equal to nothing the release tooling ever writes."""
        with self.assertRaises(ValueError):
            DeprecationRecord("A", "B", "१.२.३")

    def test_two_part_version_is_accepted(self):
        self.assertEqual(Deprecation.version_key("1.4"), (1, 4, 0))

    def test_expiry_boundary_is_inclusive(self):
        record = DeprecationRecord("A", "B", "1.2.0")
        self.assertTrue(record.expired("1.2.0"))
        self.assertTrue(record.expired("1.2.1"))
        self.assertTrue(record.expired("2.0.0"))
        self.assertFalse(record.expired("1.1.9"))

    def test_expiry_compares_numerically_not_lexically(self):
        """``"0.9.40" > "0.10.0"`` as strings; as releases it is the reverse."""
        record = DeprecationRecord("A", "B", "0.10.0")
        self.assertFalse(record.expired("0.9.40"))
        self.assertTrue(record.expired("0.10.0"))

    def test_package_is_read_off_the_module(self):
        record = DeprecationRecord("A", "B", "1.0.0", module="mayatk.uv_utils")
        self.assertEqual(record.package, "mayatk")


class DeprecationWarnTest(BaseTestCase):
    """``Deprecation.warn`` -- the escape hatch for shapes the decorators miss.

    It exists so that a deprecation the four shapes cannot express still has a
    sanctioned route. Without one, the only option left is the hand-rolled
    ``warnings.warn`` this module replaced, which is how the ecosystem ended up
    with seven spellings and two hand-derived stacklevels six call sites
    disagreed over.
    """

    def test_it_warns_and_registers(self):
        with self.assertWarns(DeprecationWarning) as caught:
            record = Deprecation.warn(
                "Owner.legacy_path", "Owner.new_path", remove_in="9.9.9"
            )
        self.assertIn("Owner.new_path", str(caught.warning))
        self.assertIn(record, Deprecation.registered(module=__name__))

    def test_the_module_defaults_to_the_caller(self):
        """The module names the package in the message, so reading it off the
        caller is what makes 'removed in pythontk 0.11.0' say 'pythontk'."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            record = Deprecation.warn("A.b", "A.c", remove_in="9.9.9")
        self.assertEqual(record.module, __name__)

    def test_stacklevel_1_points_at_this_frame(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            Deprecation.warn("A.b", "A.c", remove_in="9.9.9")
        self.assertEqual(caught[0].filename, __file__)

    def test_stacklevel_counts_out_through_a_consumer_helper(self):
        """The case the raised stacklevel exists for: a package wraps this in
        its own one-line helper so a cluster of entry points share a message.
        At 1 the warning blames the helper; the caller needs 3."""

        def helper():
            Deprecation.warn("A.b", "A.c", remove_in="9.9.9", stacklevel=3)

        def entry_point():
            helper()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            entry_point()
        here = inspect.currentframe().f_lineno - 1
        self.assertEqual(caught[0].lineno, here)

    def test_an_unparseable_version_raises_rather_than_warning(self):
        with self.assertRaises(ValueError):
            Deprecation.warn("A.b", "A.c", remove_in="next release")


class DeprecationFrameAccountingTest(BaseTestCase):
    """The machinery's own frames, which every shape's attribution rests on."""

    def test_every_transparent_module_still_exists(self):
        """The set is spelled by NAME, so a module rename would silently stop
        the skipping and start blaming the lazy resolver for the caller's line.
        Nothing else would fail."""
        from pythontk.core_utils import deprecation as module

        for name in module._TRANSPARENT_MODULES:
            with self.subTest(module=name):
                if name == module.__name__:
                    continue
                self.assertIsNotNone(importlib.import_module(name))

    def test_the_resolver_is_transparent(self):
        """A deprecated attribute reached through the package root is served by
        the resolver's ``__getattr__``; blaming the resolver tells the reader
        nothing about which of their own lines to change."""
        from pythontk.core_utils import deprecation as module

        self.assertIn(
            "pythontk.core_utils.module_resolver", module._TRANSPARENT_MODULES
        )


class DeprecationSymbolTest(BaseTestCase):
    """``Deprecation.symbol`` over functions, methods and classes."""

    def test_behaviour_is_unchanged(self):
        """Warning is not breaking -- the whole point of a one-release alias."""

        @Deprecation.symbol("live_fn", remove_in="9.9.9")
        def old_fn(a, b=2):
            return a + b

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            self.assertEqual(old_fn(1), 3)
            self.assertEqual(old_fn(1, b=10), 11)

    def test_warns_with_the_replacement_and_version(self):
        @Deprecation.symbol("live_fn", remove_in="9.9.9")
        def old_fn():
            return 1

        with self.assertWarns(DeprecationWarning) as caught:
            old_fn()
        self.assertIn("live_fn", str(caught.warning))
        self.assertIn("9.9.9", str(caught.warning))

    def test_attributed_to_the_caller_not_the_machinery(self):
        """A warning pointing at the deprecation module names no line the
        caller can change, which is the one thing the notice is for."""

        @Deprecation.symbol("live_fn", remove_in="9.9.9")
        def old_fn():
            return 1

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            old_fn()
        self.assertEqual(caught[0].filename, __file__)

    def test_decorator_order_is_irrelevant(self):
        """Written above ``@classmethod`` the decorator sees the descriptor,
        not the function. Both orders must warn, or one of them is a silent
        no-op that reads exactly like the working one."""

        class Below:
            @classmethod
            @Deprecation.symbol("live", remove_in="9.9.9")
            def old(cls):
                return "below"

        class Above:
            @Deprecation.symbol("live", remove_in="9.9.9")
            @classmethod
            def old(cls):
                return "above"

        for owner, expected in ((Below, "below"), (Above, "above")):
            with self.subTest(owner=owner.__name__):
                with self.assertWarns(DeprecationWarning):
                    self.assertEqual(owner.old(), expected)

    def test_staticmethod_is_preserved(self):
        class Owner:
            @staticmethod
            @Deprecation.symbol("live", remove_in="9.9.9")
            def old(x):
                return x * 2

        with self.assertWarns(DeprecationWarning):
            self.assertEqual(Owner.old(4), 8)

    def test_property_is_supported_in_either_order(self):
        """Above ``@property`` the decorator sees a non-callable descriptor;
        naively wrapping it builds a member that raises on first access."""

        class Over:
            @Deprecation.symbol("Over.live", remove_in="9.9.9")
            @property
            def value(self):
                return "over"

        class Under:
            @property
            @Deprecation.symbol("Under.live", remove_in="9.9.9")
            def value(self):
                return "under"

        for owner, expected in ((Over, "over"), (Under, "under")):
            with self.subTest(owner=owner.__name__):
                with self.assertWarns(DeprecationWarning) as caught:
                    self.assertEqual(owner().value, expected)
                # The message must NAME the member. A property descriptor
                # carries neither __qualname__ nor __module__, so describing it
                # directly yielded "<property object at 0x...>" -- a notice
                # nobody can act on, and a roster key that moved every run.
                message = str(caught.warning)
                self.assertIn(f"{owner.__name__}.value", message)
                self.assertNotIn("property object at", message)
                self.assertIn("9.9.9", message)

    def test_a_setter_attached_afterwards_needs_its_own_decorator(self):
        """``@value.setter`` builds a FRESH property from the getter, so a
        decorator written above ``@property`` cannot reach a setter added
        later. Decorating each accessor directly does reach it -- pinned here
        because the alternative is a write path that silently never warns."""

        class Owner:
            @property
            @Deprecation.symbol("Owner.live", remove_in="9.9.9")
            def value(self):
                return self._v

            @value.setter
            @Deprecation.symbol("Owner.live", remove_in="9.9.9")
            def value(self, v):
                self._v = v

        owner = Owner()
        with self.assertWarns(DeprecationWarning):
            owner.value = 3
        with self.assertWarns(DeprecationWarning):
            self.assertEqual(owner.value, 3)

    def test_a_property_with_both_accessors_is_wrapped_whole(self):
        """Applied to a COMPLETE property, every accessor it carries is
        wrapped -- the read-only case above is just the common one."""

        def get(self):
            return getattr(self, "_v", None)

        def put(self, v):
            self._v = v

        class Owner:
            value = Deprecation.symbol("Owner.live", remove_in="9.9.9")(
                property(get, put)
            )

        owner = Owner()
        with self.assertWarns(DeprecationWarning):
            owner.value = 7
        with self.assertWarns(DeprecationWarning):
            self.assertEqual(owner.value, 7)

    def test_class_construction_warns_and_still_builds(self):
        @Deprecation.symbol("NewThing", remove_in="9.9.9")
        class OldThing:
            def __init__(self, a, b=2):
                self.total = a + b

        with self.assertWarns(DeprecationWarning):
            instance = OldThing(1, b=5)
        self.assertEqual(instance.total, 6)

    def test_subclassing_a_retired_class_warns(self):
        """A subclass is a live dependency on the retired class, and the
        hand-written ``__init__`` warn it replaces could not see one."""

        @Deprecation.symbol("NewThing", remove_in="9.9.9")
        class OldThing:
            def __init__(self):
                self.built = True

        class Child(OldThing):
            pass

        with self.assertWarns(DeprecationWarning):
            self.assertTrue(Child().built)

    def test_a_retired_class_keeps_a_staticmethod_new(self):
        """The wrapper is stored the way CPython stores a ``__new__`` defined in
        a class body. Pinned because the call works without it, so the next
        reader would otherwise have every reason to strip it as a no-op."""
        import inspect as _inspect

        @Deprecation.symbol("NewThing", remove_in="9.9.9")
        class OldThing:
            def __init__(self):
                pass

        self.assertIsInstance(
            _inspect.getattr_static(OldThing, "__new__"), staticmethod
        )

    def test_marker_is_the_pep_702_message_string(self):
        @Deprecation.symbol("live_fn", remove_in="9.9.9")
        def old_fn():
            return 1

        self.assertEqual(old_fn.__deprecated__, old_fn.__deprecated_record__.message())

    def test_marker_survives_unwrapping(self):
        """``functools.wraps`` sets ``__wrapped__``, and ``HelpMixin`` unwraps
        to recover a real signature. Marking only the wrapper would make every
        deprecated method read as live in ``help()`` and the registry."""
        import inspect

        @Deprecation.symbol("live_fn", remove_in="9.9.9")
        def old_fn():
            return 1

        self.assertTrue(getattr(inspect.unwrap(old_fn), "__deprecated__", False))

    def test_help_mixin_reads_the_marker_and_the_version(self):
        class Owner(ptk.HelpMixin):
            @classmethod
            @Deprecation.symbol("Owner.live", remove_in="9.9.9")
            def old(cls):
                """Retired."""

            @classmethod
            def live(cls):
                """Current."""

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            members = {r["name"]: r for r in Owner.help(as_dict=True)["members"]}
        self.assertTrue(members["old"]["deprecated"])
        self.assertEqual(members["old"]["remove_in"], "9.9.9")
        self.assertFalse(members["live"]["deprecated"])
        self.assertEqual(members["live"]["remove_in"], "")

    def test_the_wrapper_does_not_hide_the_definition_site(self):
        """``help()``/``where()`` must still point at the retired member's own
        source, not at the deprecation module the wrapper is compiled in --
        that source line is the whole reason someone looks a deprecated member
        up. ``inspect.unwrap`` carries it, but only because both halves are
        marked and ``functools.wraps`` set ``__wrapped__``."""

        @Deprecation.symbol("live_fn", remove_in="9.9.9")
        def old_fn():
            return 1

        self.assertEqual(
            inspect.getsourcefile(inspect.unwrap(old_fn)),
            inspect.getsourcefile(type(self)),
        )

    def test_help_mixin_sees_a_deprecated_property(self):
        """A ``property`` has no ``__dict__``, so the marker lives on an
        accessor. Reading only the descriptor reported the member as LIVE while
        the static registry marked it retired, and the runtime-vs-static drift
        gate compares (qualname, kind) -- so nothing caught the disagreement."""

        class Owner(ptk.HelpMixin):
            @Deprecation.symbol("Owner.live", remove_in="9.9.9")
            @property
            def value(self):
                """Retired."""
                return 1

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            members = {r["name"]: r for r in Owner.help(as_dict=True)["members"]}
        self.assertEqual(members["value"]["kind"], "property")
        self.assertTrue(members["value"]["deprecated"])
        self.assertEqual(members["value"]["remove_in"], "9.9.9")

    def test_registry_row_carries_the_deadline(self):
        record = ptk.SymbolRecord(
            name="old",
            qualname="Owner.old",
            kind="classmethod",
            signature="()",
            summary="",
            line=1,
            deprecated=True,
            remove_in="9.9.9",
        )
        self.assertIn("**DEPRECATED (remove in 9.9.9)**", record.to_registry_row())


class DeprecationParameterTest(BaseTestCase):
    """``Deprecation.parameter`` -- the renamed keyword, warned instead of
    documented."""

    @staticmethod
    def _mirror():
        @Deprecation.parameter(
            "use_object_axes",
            new="axis_frame",
            transform=lambda v: "world" if v is False else "auto",
            remove_in="9.9.9",
        )
        def mirror(obj, axis_frame="auto"):
            return (obj, axis_frame)

        return mirror

    def test_old_spelling_is_remapped(self):
        mirror = self._mirror()
        with self.assertWarns(DeprecationWarning):
            self.assertEqual(mirror("cube", use_object_axes=False), ("cube", "world"))

    def test_live_spelling_is_silent(self):
        mirror = self._mirror()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = mirror("cube", axis_frame="local")
        self.assertEqual(result, ("cube", "local"))
        self.assertEqual(caught, [])

    def test_the_live_spelling_wins_a_conflict(self):
        """Both spellings in one call. Letting the retired one override is how
        a deprecated bool silently beat the enum callers were told to use."""
        mirror = self._mirror()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = mirror("cube", axis_frame="local", use_object_axes=False)
        self.assertEqual(result, ("cube", "local"))
        self.assertEqual(len(caught), 2)
        self.assertIn("wins", str(caught[1].message))

    def test_positional_calls_are_not_warned(self):
        """A positional argument carries no name, so nothing says the caller
        meant the retired spelling -- and for a pure rename the position did
        not move, so such a call needs no migration at all."""

        @Deprecation.parameter("old_name", new="new_name", remove_in="9.9.9")
        def fn(a, new_name=None):
            return (a, new_name)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = fn(1, "value")
        self.assertEqual(result, (1, "value"))
        self.assertEqual(caught, [])

    def test_drop_discards_a_no_effect_parameter(self):
        @Deprecation.parameter("as_strings", drop=True, remove_in="9.9.9")
        def names(x):
            return [x]

        with self.assertWarns(DeprecationWarning):
            self.assertEqual(names("a", as_strings=True), ["a"])

    def test_the_message_matches_what_actually_happens(self):
        """Three cases, three different things a caller must do. A forwarded
        parameter described as having "no effect" is a false instruction."""

        @Deprecation.parameter("a", new="live", remove_in="9.9.9")
        def renamed(live=None):
            return live

        @Deprecation.parameter("b", drop=True, remove_in="9.9.9")
        def dropped(**kwargs):
            return None

        @Deprecation.parameter("c", remove_in="9.9.9")
        def forwarded(c=None):
            return c

        with self.assertWarns(DeprecationWarning) as caught:
            renamed(a=1)
        self.assertIn("use live instead", str(caught.warning))
        with self.assertWarns(DeprecationWarning) as caught:
            dropped(b=1)
        self.assertIn("already has no effect", str(caught.warning))
        with self.assertWarns(DeprecationWarning) as caught:
            forwarded(c=1)
        self.assertIn("stop passing it", str(caught.warning))

    def test_without_new_the_value_is_still_forwarded(self):
        """Warning is not breaking: with no replacement keyword and no
        ``drop``, this release behaves exactly as it did."""

        @Deprecation.parameter("legacy", remove_in="9.9.9")
        def fn(legacy=None):
            return legacy

        with self.assertWarns(DeprecationWarning):
            self.assertEqual(fn(legacy="kept"), "kept")

    def test_the_owner_is_not_marked_deprecated(self):
        """The method stays; only the keyword goes. Marking the owner would
        make the registry announce the removal of a live method."""

        @Deprecation.parameter("old_kw", new="new_kw", remove_in="9.9.9")
        def fn(new_kw=None):
            return new_kw

        self.assertFalse(getattr(fn, "__deprecated__", False))

    def test_it_is_registered_as_a_parameter(self):
        @Deprecation.parameter("rostered_kw", new="live_kw", remove_in="9.9.9")
        def fn(live_kw=None):
            return live_kw

        found = [
            r
            for r in Deprecation.registered(kind="parameter")
            if "rostered_kw" in r.what
        ]
        self.assertEqual(len(found), 1)


class DeprecationAttributesTest(BaseTestCase):
    """``Deprecation.attributes`` -- module attributes that moved."""

    def _module(self, name="_dep_test_module"):
        module = types.ModuleType(name)
        module.live_name = "live"
        sys.modules[name] = module
        self.addCleanup(sys.modules.pop, name, None)
        return module

    def test_moved_attribute_resolves_and_warns(self):
        module = self._module()
        Deprecation.attributes(
            module.__dict__,
            {"SymbolRecord": "pythontk.core_utils.symbol_record.SymbolRecord"},
            remove_in="9.9.9",
        )
        with self.assertWarns(DeprecationWarning) as caught:
            resolved = module.SymbolRecord
        self.assertIs(resolved, ptk.SymbolRecord)
        self.assertIn("9.9.9", str(caught.warning))

    def test_a_live_name_is_untouched(self):
        module = self._module()
        Deprecation.attributes(
            module.__dict__,
            {"Gone": "pythontk.core_utils.symbol_record.SymbolRecord"},
            remove_in="9.9.9",
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertEqual(module.live_name, "live")
        self.assertEqual(caught, [])

    def test_an_unknown_name_still_raises_attribute_error(self):
        module = self._module()
        Deprecation.attributes(
            module.__dict__,
            {"Gone": "pythontk.core_utils.symbol_record.SymbolRecord"},
            remove_in="9.9.9",
        )
        with self.assertRaises(AttributeError):
            module.no_such_name

    def test_an_existing_getattr_is_chained_not_clobbered(self):
        """Several packages front a lazy loader with a module ``__getattr__``.
        Replacing it outright would make every lazily-loaded name on the
        module vanish -- a far bigger break than the alias it was serving."""
        module = self._module()

        def lazy(name):
            if name == "LazyName":
                return "lazily-built"
            raise AttributeError(name)

        module.__dict__["__getattr__"] = lazy
        Deprecation.attributes(
            module.__dict__,
            {"Gone": "pythontk.core_utils.symbol_record.SymbolRecord"},
            remove_in="9.9.9",
        )
        self.assertEqual(module.LazyName, "lazily-built")
        with self.assertWarns(DeprecationWarning):
            self.assertIs(module.Gone, ptk.SymbolRecord)
        with self.assertRaises(AttributeError):
            module.neither

    def test_dir_lists_both_halves(self):
        module = self._module()
        Deprecation.attributes(
            module.__dict__,
            {"Gone": "pythontk.core_utils.symbol_record.SymbolRecord"},
            remove_in="9.9.9",
        )
        listed = dir(module)
        self.assertIn("Gone", listed)
        self.assertIn("live_name", listed)

    def test_registered_at_import_not_at_first_use(self):
        """An alias nobody has tripped is the only kind that survives a release
        unnoticed, so the roster has to see it before anyone touches it."""
        self._module("_dep_test_unused")
        Deprecation.attributes(
            sys.modules["_dep_test_unused"].__dict__,
            {"NeverTouched": "pythontk.core_utils.symbol_record.SymbolRecord"},
            remove_in="9.9.9",
        )
        found = [
            r
            for r in Deprecation.registered(kind="attribute")
            if r.what.endswith("NeverTouched")
        ]
        self.assertEqual(len(found), 1)


class DeprecationValuesTest(BaseTestCase):
    """``Deprecation.values`` -- retired members of a value vocabulary."""

    def test_alias_resolves_to_the_live_value(self):
        resolve = Deprecation.values(
            {"stretch": "orbit"}, what="ShadowRig mode", remove_in="9.9.9"
        )
        with self.assertWarns(DeprecationWarning):
            self.assertEqual(resolve("stretch"), "orbit")

    def test_a_live_value_passes_through_silently(self):
        resolve = Deprecation.values(
            {"stretch": "orbit"}, what="ShadowRig mode", remove_in="9.9.9"
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertEqual(resolve("orbit"), "orbit")
            self.assertIsNone(resolve(None))
        self.assertEqual(caught, [])

    def test_an_unhashable_value_cannot_be_an_alias(self):
        resolve = Deprecation.values(
            {"stretch": "orbit"}, what="ShadowRig mode", remove_in="9.9.9"
        )
        self.assertEqual(resolve(["a"]), ["a"])

    def test_registered_at_construction(self):
        Deprecation.values(
            {"never_passed": "live"}, what="Rostered vocabulary", remove_in="9.9.9"
        )
        found = [
            r for r in Deprecation.registered(kind="value") if "never_passed" in r.what
        ]
        self.assertEqual(len(found), 1)


class DeprecationSinkTest(BaseTestCase):
    """The extra channel a DCC needs, because ``DeprecationWarning`` is hidden."""

    def setUp(self):
        super().setUp()
        self.seen = []
        self.addCleanup(setattr, Deprecation, "sink", Deprecation.sink)
        Deprecation.sink = self.seen.append

    def _fresh(self, what):
        record = DeprecationRecord(what, "live", "9.9.9", module="pythontk.test")
        Deprecation._announced.discard(record.key)
        self.addCleanup(Deprecation._announced.discard, record.key)
        return record

    def test_the_sink_receives_the_message(self):
        record = self._fresh("SinkTest.one")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            Deprecation._emit(record)
        self.assertEqual(self.seen, [record.message()])

    def test_the_sink_speaks_once_per_record(self):
        """A deprecated call inside a per-frame loop would otherwise paper over
        the viewport for the length of a session."""
        record = self._fresh("SinkTest.two")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            for _ in range(5):
                Deprecation._emit(record)
        self.assertEqual(len(self.seen), 1)

    def test_warnings_warn_still_fires_every_time(self):
        """The sink is additive. Gating ``warnings.warn`` on it would break
        ``assertWarns`` and ``-W error`` for everything downstream."""
        record = self._fresh("SinkTest.three")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for _ in range(3):
                Deprecation._emit(record)
        self.assertEqual(len(caught), 3)

    def test_a_raising_sink_cannot_break_the_call(self):
        """``cmds.warning`` raises off the main thread and after teardown; a
        deprecation notice must not be able to take the call down with it."""
        Deprecation.sink = lambda message: 1 / 0
        record = self._fresh("SinkTest.four")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            Deprecation._emit(record)  # must not raise


class DeprecationRosterTest(BaseTestCase):
    """The roster, and the gate it exists to feed."""

    def test_registration_is_idempotent(self):
        """A DCC reload re-executes a module and forks class state; the roster
        must not grow a duplicate row each time."""
        before = len(Deprecation.registered())
        for _ in range(3):
            Deprecation._register(DeprecationRecord("Idempotent.one", "live", "9.9.9"))
        self.assertEqual(len(Deprecation.registered()), before + 1)

    def test_module_filter_matches_the_package_prefix(self):
        Deprecation._register(
            DeprecationRecord(
                "Filtered.one", "live", "9.9.9", module="madeuppkg.sub.mod"
            )
        )
        self.assertTrue(Deprecation.registered(module="madeuppkg"))
        self.assertFalse(Deprecation.registered(module="madeuppkgx"))

    def test_sorted_by_deadline(self):
        versions = [r.remove_in for r in Deprecation.registered()]
        keys = [Deprecation.version_key(v) for v in versions]
        self.assertEqual(keys, sorted(keys))

    def test_report_marks_the_overdue(self):
        Deprecation._register(
            DeprecationRecord("Overdue.one", "live", "0.0.1", module="madeuppkg")
        )
        self.assertIn("EXPIRED", Deprecation.report("1.0.0", module="madeuppkg"))
        self.assertNotIn("EXPIRED", Deprecation.report("0.0.0", module="madeuppkg"))

    def test_report_is_explicit_when_empty(self):
        self.assertEqual(
            Deprecation.report(module="nothing.here"), "No deprecations registered."
        )


class PythontkRetirementDebtTest(BaseTestCase):
    """The live gate over pythontk's own surface.

    ``UvUtils.flip_uvs`` was deprecated on 2025-12-17 and shipped in 50
    releases afterwards, because "removed in the next release" was a sentence
    rather than a comparison. This is the comparison.
    """

    @staticmethod
    def _import_every_module():
        """Import all of pythontk so the roster is complete.

        ``export_all()`` is not enough: it resolves the names in
        ``DEFAULT_INCLUDE``, and a pure alias module (a moved import path)
        registers its retirements at import and is named by nothing. Measured at 0.3s for the whole package, which also
        exercises the no-side-effects-on-import rule.
        """
        for found in pkgutil.walk_packages(ptk.__path__, "pythontk."):
            try:
                importlib.import_module(found.name)
            except Exception:  # an unimportable module is other suites' problem
                pass

    def test_nothing_in_pythontk_has_outlived_its_window(self):
        self._import_every_module()
        overdue = Deprecation.expired(ptk.__version__, module="pythontk")
        self.assertEqual(
            [r.what for r in overdue],
            [],
            "Deprecation(s) past their removal release. Delete the alias and "
            "its tests, or raise remove_in deliberately and say why in "
            f"CHANGELOG.md:\n{Deprecation.report(ptk.__version__, module='pythontk')}",
        )

    def test_every_pythontk_deprecation_names_a_removal_version(self):
        self._import_every_module()
        for record in Deprecation.registered(module="pythontk"):
            with self.subTest(what=record.what):
                self.assertTrue(record.remove_in)


if __name__ == "__main__":
    unittest.main(exit=False)
