# !/usr/bin/python
# coding=utf-8
"""Tests for pythontk.core_utils.operation_catalog: Operation, OpContext,
Outcome, OpRequest, PureOperationHost, BatchOperationHost (a fresh Python
process stands in for the application) and OperationCatalog (build, discover,
validate, run, CLI)."""

import contextlib
import enum
import importlib
import io
import json
import os
import sys
import tempfile
import textwrap
import threading
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, List, Optional, Union
from unittest.mock import patch

from pythontk import (
    ApiSurface,
    BatchOperationHost,
    Change,
    CancelScope,
    OpCall,
    OpContext,
    Operation,
    OperationCatalog,
    OperationCatalogDocument,
    OperationEntry,
    OperationHost,
    OperationProvider,
    OperationServer,
    OpRequest,
    Outcome,
    PureOperationHost,
    PureSceneFiles,
    SceneFiles,
    SchemaSpec,
)

_ENTRY_POINTS = (
    "pythontk.core_utils.operation_catalog._operation_catalog.PackageManager.entry_points"
)
_LOG: List[str] = []
_SEEN: list = []


# -- target functions the fixture operations call -----------------------------
def add(a: int, b: int = 2) -> int:
    """Add two numbers.

    Parameters:
        a: The first number.
        b: The second number,
            on two lines.

    Returns:
        The sum.
    """
    return a + b


def rename(names: List[str], suffix: str = "_new", ctx=None) -> dict:
    """Rename every name.

    Parameters:
        names: The objects to rename.
        suffix: What to append.
    """
    done = []
    for name in names:
        if ctx.should_change(name, f"rename to {name}{suffix}"):
            _LOG.append(name + suffix)
            done.append(name)
    return {"renamed": done}


def wrapped(x: int = 0) -> int:
    """Return *x* as :class:`int`, the ``count``
    it was given. More detail follows.

    Parameters:
        x: A number.
    """
    return x


def rebuild(name: str, ctx=None) -> None:
    """Delete one object and make its replacement."""
    if ctx.should_delete(name, "remove the old one"):
        _LOG.append(f"deleted {name}")
    if ctx.should_create(name + "_v2", "its replacement"):
        _LOG.append(f"created {name}_v2")


def wipe(path: str) -> None:
    """Delete a file; it cannot plan, so it has no dry run."""
    _LOG.append(f"wiped {path}")


def wipe_all(paths: List[str]) -> None:
    """Delete files, failing after the first; it keeps no journal."""
    _LOG.append(f"wiped {paths[0]}")
    raise RuntimeError("disk full")


def explode(names: List[str], ctx=None) -> None:
    """Change the first name, then fail."""
    if ctx.should_change(names[0], "touch"):
        _LOG.append(f"touched {names[0]}")
    raise RuntimeError("boom")


def stop(ctx=None) -> None:
    """Cancel the run it is in, then reach a checkpoint."""
    ctx.scope.cancel("user")
    ctx.check()


def lint(ctx=None) -> None:
    """Report one issue with its fix."""
    ctx.issue(
        "demo.bad_name",
        "pCube1 is badly named",
        severity="warning",
        location="pCube1",
        fix={"op": "demo.rename", "args": {"names": ["pCube1"]}},
    )


def tag(name: str = None, count: Optional[int] = None, pick: Union[int, str] = None) -> None:
    """Tag something; every argument may be left out."""
    _SEEN.append(count)


def bail() -> None:
    """Exit the process, as a script's main() does."""
    sys.exit(3)


def chatty() -> int:
    """Print, then return."""
    print("not JSON")
    return 1


def loop() -> list:
    """Return a list that holds itself."""
    cycle: list = []
    cycle.append(cycle)
    return cycle


def ghost(x: "NoSuchType") -> None:  # noqa: F821 - the annotation must not resolve
    """Take a type that does not exist."""


def odd(x: float = float("nan")) -> None:
    """Default to NaN."""


class Axis(enum.Enum):
    X = "x"
    Y = "y"


@dataclass
class Box(SchemaSpec):
    """A box."""

    TYPED = True

    size: float = SchemaSpec.spec_field(help="Edge length.", required=True)


def pack(box: Box, where: Path = None, axis: Axis = Axis.X, spares: Optional[List[Box]] = None) -> float:
    """Pack a box; its arguments arrive as the types declared.

    Parameters:
        box: The box.
        where: Where to put it.
        axis: Which way up.
        spares: More boxes.
    """
    _SEEN.append((box, where, axis, spares))
    return box.size


def scan(names: List[str], repair: bool = False, on_step: Optional[Callable[[str], None]] = None) -> dict:
    """Report each name, repairing it when asked.

    Parameters:
        names: What to scan.
        repair: Fix what it finds.
        on_step: Called with each name.
    """
    if repair:
        _LOG.append("repaired")
    return {name: "repaired" if repair else "scanned" for name in names}


def each(fn: Callable[[str], None]) -> None:
    """Call *fn*: it cannot run without code no caller can send."""


class Job:
    """Another library's context object."""


def ship(path: str, ctx: Optional[Job] = None) -> None:
    """Ship *path* under an exporter's own context, which cannot plan."""
    _SEEN.append(ctx)


def plan_typed(name: str, ctx: Optional[OpContext] = None) -> None:
    """Touch *name*, its context annotated."""
    if ctx is None or ctx.should_change(name, "touch"):
        _LOG.append(name)


class Shelf:
    def put(self, item: str) -> None:
        """An instance method: not an operation."""


class Named:
    """An application object that carries its name."""

    def __init__(self, name: str):
        self.name = name


def _op(op_id, name, **kwargs):
    return Operation(op_id, f"{__name__}:{name}", **kwargs)


FIXTURE_OPS = (
    _op("demo.add", "add"),
    _op("demo.rename", "rename", effect="state"),
    _op("demo.wipe", "wipe", effect="files"),
    _op("demo.wipe_all", "wipe_all", effect="files"),
    _op("demo.explode", "explode", effect="state"),
    _op("demo.stop", "stop"),
    _op("demo.lint", "lint"),
    _op("demo.tag", "tag"),
    _op("demo.bail", "bail"),
    _op("demo.chatty", "chatty"),
    _op("demo.loop", "loop"),
    _op("demo.pack", "pack"),
)


def _catalog(ops=FIXTURE_OPS, hosts=None) -> OperationCatalog:
    return OperationCatalog([op.describe() for op in ops], hosts=hosts)


class TestOperation(unittest.TestCase):
    def test_a_malformed_id_effect_or_target_is_refused(self):
        for kwargs in (
            dict(id="Mesh.Mirror", target="m:f"),
            dict(id="mirror", target="m:f"),
            dict(id="mesh.mirror", target="m.f"),
            dict(id="mesh.mirror", target="m:f", effect="write"),
            dict(id="mesh.mirror", target="m:f", host="maya"),
        ):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                Operation(**kwargs)

    def test_describe_reads_the_schema_off_the_signature_and_docstring(self):
        entry = _op("demo.add", "add").describe()
        self.assertEqual(entry["id"], "demo.add")
        self.assertEqual(entry["summary"], "Add two numbers.")
        self.assertEqual(entry["title"], "Add two numbers")
        self.assertEqual(entry["effect"], "read")
        self.assertNotIn("host", entry)
        self.assertTrue(entry["dry_run"])
        self.assertEqual(
            entry["args"],
            {
                "type": "object",
                "properties": {
                    "a": {"type": "integer", "description": "The first number."},
                    "b": {
                        "type": "integer",
                        "description": "The second number, on two lines.",
                        "default": 2,
                    },
                },
                "additionalProperties": False,
                "required": ["a"],
            },
        )
        self.assertEqual(entry["result"], {"type": "integer"})
        self.assertEqual(OperationEntry.validate(entry).errors, [])

    def test_the_summary_is_the_first_sentence_spelled_as_markdown(self):
        entry = _op("demo.wrapped", "wrapped").describe()
        self.assertEqual(entry["summary"], "Return *x* as `int`, the `count` it was given.")
        for doc, summary in (
            ("Rename e.g. a material. Then more.", "Rename e.g. a material."),
            ("Read the file, i.e. the header. Then more.", "Read the file, i.e. the header."),
            ("Bake for e.g. Maya. Then more.", "Bake for e.g. Maya."),
            ("Two steps.\n`ctx` plans them.", "Two steps."),
        ):
            self.assertEqual(ApiSurface.summary(doc), summary)  # the catalog's rule is the engine's

    def test_ctx_is_not_an_argument_and_gives_a_change_its_dry_run(self):
        entry = _op("maya.rename", "rename", effect="state", host="maya").describe()
        self.assertEqual(set(entry["args"]["properties"]), {"names", "suffix"})
        self.assertEqual(entry["args"]["properties"]["names"]["type"], "array")
        self.assertEqual(entry["host"], "maya")
        self.assertTrue(entry["dry_run"])
        self.assertFalse(_op("demo.wipe", "wipe", effect="files").describe()["dry_run"])

    def test_a_none_default_admits_null_and_reads_alike_under_every_interpreter(self):
        # Python before 3.11 reads `name: str = None` as Optional[str] (and
        # Union[int, str] = None as one flat Union), later ones as written.
        properties = _op("demo.tag", "tag").describe()["args"]["properties"]
        null_or = lambda *kinds: {
            "anyOf": [{"type": k} for k in kinds] + [{"type": "null"}],
            "default": None,
        }
        self.assertEqual(properties["name"], null_or("string"))
        self.assertEqual(properties["count"], null_or("integer"))
        self.assertEqual(properties["pick"], null_or("integer", "string"))

    def test_a_schema_typed_argument_renders_as_json(self):
        document = OperationProvider.document((_op("demo.pack", "pack"),), "test")
        text = OperationProvider.render(document)  # raised on the spec's class claim
        args = json.loads(text)["operations"][0]["args"]
        self.assertEqual(args["properties"]["box"]["$ref"], "#/$defs/Box")
        self.assertIn("Box", args["$defs"])
        self.assertEqual(args["properties"]["where"]["anyOf"][0], {"type": "string"})
        self.assertEqual(args["properties"]["axis"]["enum"], ["x", "y"])

    def test_an_annotation_that_does_not_resolve_is_refused_not_dropped(self):
        with self.assertRaises(ValueError) as caught:
            _op("demo.ghost", "ghost").describe()
        self.assertIn("does not resolve at runtime", str(caught.exception))

    def test_a_nan_default_is_left_out_of_the_catalog(self):
        x = _op("demo.odd", "odd").describe()["args"]["properties"]["x"]
        self.assertNotIn("default", x)

    def test_an_instance_method_is_not_an_operation(self):
        with self.assertRaises(ValueError) as caught:
            _op("demo.put", "Shelf.put").describe()
        self.assertIn("is an instance method", str(caught.exception))

    def test_only_an_operation_context_gives_a_dry_run(self):
        self.assertTrue(Operation.takes_context(plan_typed))
        self.assertFalse(Operation.takes_context(ship))  # its ctx is another kind
        entry = _op("demo.ship", "ship", effect="files").describe()
        self.assertEqual((entry["dry_run"], list(entry["args"]["properties"])), (False, ["path"]))
        _SEEN.clear()
        done = _catalog((_op("demo.ship", "ship", effect="files"),)).run("demo.ship", {"path": "a"}, apply=True)
        self.assertEqual((done.status, _SEEN), ("done", [None]))  # it keeps its own default

    def test_a_code_hook_is_left_out_and_a_required_one_refused(self):
        # A callable never arrives as JSON: the call keeps its default.
        self.assertEqual(set(_op("demo.scan", "scan").describe()["args"]["properties"]), {"names", "repair"})
        with self.assertRaises(ValueError) as caught:
            _op("demo.each", "each").describe()
        self.assertIn("takes code", str(caught.exception))

class TestOperationCatalogRun(unittest.TestCase):
    def setUp(self):
        _LOG.clear()
        _SEEN.clear()

    def test_a_read_runs_and_returns_its_result(self):
        outcome = _catalog().run("demo.add", {"a": 1, "b": 5})
        self.assertEqual((outcome.status, outcome.result, outcome.dry_run), ("done", 6, False))
        self.assertTrue(outcome.ok)

    def test_an_unknown_missing_or_mistyped_argument_is_refused_before_the_call(self):
        catalog = _catalog()
        for op_id, args, expected in (
            ("demo.add", {"a": 1, "c": 2}, "unknown argument 'c' (expected: a, b)"),
            ("demo.add", {}, "missing argument 'a'"),
            ("demo.add", {"a": "one"}, "a: expected integer, got string"),
            ("demo.add", {"a": True}, "a: expected integer, got boolean"),
            ("demo.tag", {"count": "x"}, "count: expected integer, got string"),
            ("demo.pack", {"box": {"size": "big"}}, "box.size: expected number, got string"),
            # An Optional's miss names its one real shape.
            ("demo.pack", {"box": {"size": 1}, "where": 5}, "where: expected string, got integer"),
        ):
            with self.subTest(op=op_id, args=args):
                outcome = catalog.run(op_id, args)
                self.assertEqual(outcome.status, "failed")
                self.assertIn(expected, outcome.error)
        outcome = catalog.run("demo.rename", {"names": ["a", 2]}, apply=True)
        self.assertIn("names[1]: expected string, got integer", outcome.error)
        self.assertEqual(_LOG, [])

    def test_arguments_arrive_as_the_types_declared(self):
        catalog = _catalog()
        outcome = catalog.run(
            "demo.pack",
            {"box": {"size": 2.0}, "where": "a/b.txt", "axis": "y", "spares": [{"size": 1.0}]},
        )
        self.assertEqual(outcome.result, 2.0)
        ((box, where, axis, spares),) = _SEEN
        self.assertIsInstance(box, Box)
        self.assertEqual(where, Path("a/b.txt"))
        self.assertIs(axis, Axis.Y)
        self.assertIsInstance(spares[0], Box)
        # JSON Schema counts 2.0 an integer; the function gets an int.
        self.assertTrue(catalog.run("demo.tag", {"count": 2.0}).ok)
        self.assertEqual(_SEEN[-1], 2)
        self.assertIsInstance(_SEEN[-1], int)

    def test_a_change_is_a_dry_run_unless_applied(self):
        catalog = _catalog()
        plan = catalog.run("demo.rename", {"names": ["a", "b"]})
        self.assertTrue(plan.dry_run)
        self.assertEqual(plan.status, "done")
        self.assertEqual([c.target for c in plan.changes], ["a", "b"])
        self.assertEqual(plan.changes[0].kind, "change")
        self.assertEqual(plan.result, {"renamed": []})
        self.assertEqual(_LOG, [], "a dry run changed something")

        done = catalog.run("demo.rename", {"names": ["a", "b"]}, apply=True)
        self.assertFalse(done.dry_run)
        self.assertEqual(done.result, {"renamed": ["a", "b"]})
        self.assertEqual(_LOG, ["a_new", "b_new"])

    def test_a_plan_tells_a_creation_and_a_deletion_from_an_edit(self):
        catalog = _catalog(FIXTURE_OPS + (_op("demo.rebuild", "rebuild", effect="state"),))
        plan = catalog.run("demo.rebuild", {"name": "a"})
        self.assertEqual(
            [(c.kind, c.target) for c in plan.changes], [("delete", "a"), ("create", "a_v2")]
        )
        self.assertEqual(_LOG, [])
        self.assertEqual(Change.KINDS, ("create", "change", "delete", "write"))

    def test_a_write_with_no_dry_run_runs_only_when_applied(self):
        catalog = _catalog()
        refused = catalog.run("demo.wipe", {"path": "x.txt"})
        self.assertEqual(refused.status, "failed")
        self.assertIn("has no dry run: apply it", refused.error)
        self.assertEqual(_LOG, [])
        self.assertTrue(catalog.run("demo.wipe", {"path": "x.txt"}, apply=True).ok)
        self.assertEqual(_LOG, ["wiped x.txt"])

    def test_a_host_handed_a_dry_run_of_an_op_with_no_journal_refuses_it(self):
        # The catalog never asks this; a host asked directly must not run it live.
        entry = _op("demo.wipe", "wipe", effect="files").describe()
        outcome = PureOperationHost().run(entry, OpRequest(op="demo.wipe", args={"path": "x"}, dry_run=True))
        self.assertEqual(outcome.status, "failed")
        self.assertIn("has no dry run", outcome.error)
        self.assertEqual(_LOG, [])

    def test_a_raise_after_a_live_change_is_partial_and_before_it_failed(self):
        catalog = _catalog()
        live = catalog.run("demo.explode", {"names": ["a"]}, apply=True)
        self.assertEqual(live.status, "partial")
        self.assertEqual(live.error, "RuntimeError: boom")
        self.assertEqual([c.target for c in live.changes], ["a"])
        planned = catalog.run("demo.explode", {"names": ["a"]})
        self.assertEqual(planned.status, "failed")

    def test_a_live_run_with_no_journal_that_raises_is_partial_not_failed(self):
        outcome = _catalog().run("demo.wipe_all", {"paths": ["a", "b"]}, apply=True)
        self.assertEqual(_LOG, ["wiped a"])
        self.assertEqual(outcome.status, "partial")
        self.assertIn("keeps no journal", outcome.error)

    def test_a_cancel_at_a_checkpoint_is_cancelled(self):
        outcome = _catalog().run("demo.stop", scope=CancelScope("test"))
        self.assertEqual(outcome.status, "cancelled")

    def test_issues_ride_on_the_outcome_with_their_fix(self):
        outcome = _catalog().run("demo.lint")
        self.assertTrue(outcome.ok)
        (issue,) = outcome.issues
        self.assertEqual(
            (issue.code, issue.severity, issue.location), ("demo.bad_name", "warning", "pCube1")
        )
        self.assertEqual(issue.fix.op, "demo.rename")
        self.assertEqual(issue.fix.args, {"names": ["pCube1"]})

    def test_an_issue_fix_must_name_its_op(self):
        ctx = OpContext()
        with self.assertRaises(ValueError):
            ctx.issue("demo.bad", "bad", fix={"args": {}})
        issue = ctx.issue("demo.bad", "bad", fix=OpCall(op="demo.add", args={"a": 1}))
        self.assertEqual(issue.fix.op, "demo.add")

    def test_an_operation_that_exits_fails_and_leaves_the_process_running(self):
        outcome = _catalog().run("demo.bail")
        self.assertEqual((outcome.status, outcome.error), ("failed", "SystemExit: 3"))

    def test_what_an_operation_prints_never_reaches_stdout(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            outcome = _catalog().run("demo.chatty")
        self.assertEqual((outcome.result, out.getvalue()), (1, ""))

    def test_an_unreadable_result_never_costs_the_outcome(self):
        outcome = _catalog().run("demo.loop")
        self.assertTrue(outcome.ok)
        self.assertIn("<unreadable result: RecursionError", outcome.result)
        json.loads(outcome.to_json())

    def test_a_named_object_reads_as_its_name_as_a_value_and_a_key(self):
        cube = Named("Cube")  # a Blender datablock, say: a caller passes the name back
        self.assertEqual(PureOperationHost._jsonable({cube: [cube], "n": 1}), {"Cube": ["Cube"], "n": 1})

    def test_a_host_with_no_runner_is_refused_and_a_registered_one_runs_it(self):
        ops = (_op("maya.add", "add", host="maya"),)
        with patch(_ENTRY_POINTS, return_value=[]):
            refused = _catalog(ops).run("maya.add", {"a": 1})
        self.assertEqual(refused.status, "failed")
        self.assertIn("nothing runs 'maya' operations here", refused.error)

        calls = []

        class MayaHost:
            def run(self, entry, request, scope=None, on_progress=None):
                calls.append((entry["id"], dict(request.args), request.dry_run, request.scenes))
                return Outcome(op=entry["id"], status="done", result="ran in maya")

        self.assertIsInstance(MayaHost(), OperationHost)
        outcome = _catalog(ops, hosts={"maya": MayaHost()}).run("maya.add", {"a": 1})
        self.assertEqual(outcome.result, "ran in maya")
        self.assertEqual(calls, [("maya.add", {"a": 1}, False, [])])

    def test_a_host_operation_id_starts_with_its_host(self):
        self.assertIsNone(Operation.id_error("maya.mesh.mirror", "maya"))
        self.assertIn("must start with 'maya.'", Operation.id_error("mesh.mirror", "maya"))
        entry = dict(_op("demo.add", "add").describe(), host="maya")
        with self.assertRaisesRegex(ValueError, "must start with 'maya.'"):
            OperationCatalog([entry])

    def test_an_id_fits_an_mcp_tool_name(self):
        longest = "a." + "b" * (Operation.ID_MAX - 2)
        self.assertIsNone(Operation.id_error(longest))
        self.assertIn("longer than", Operation.id_error(longest + "c"))
        # Served by the m3trik plugin, a client names the tool
        # mcp__plugin_<plugin>_<server>__<tool> (measured), and Claude caps a
        # tool name at 64 characters.
        plugin_tool = "mcp__plugin_m3trik_ops-apply__" + OperationServer.tool_name(longest)
        self.assertLessEqual(len(plugin_tool), 64)

    def test_an_id_led_by_a_host_name_runs_in_that_host(self):
        ops = [
            _op("maya.add", "add", host="maya").describe(),
            _op("maya.tag", "tag").describe(),
            _op("maya_tools.tag", "tag").describe(),
        ]
        with self.assertRaisesRegex(ValueError, "'maya.tag' runs in any Python"):
            OperationCatalog(ops)
        self.assertEqual(OperationCatalog([ops[0], ops[2]]).ids(), ["maya.add", "maya_tools.tag"])

    def test_scenes_reach_the_host_resolved_and_are_refused_where_none_open(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene = Path(tmp, "shot.ma")
            scene.write_text("scene", encoding="utf-8")
            seen = []

            class Host:
                def run(self, entry, request, scope=None, on_progress=None):
                    seen.append(request.scenes)
                    return Outcome(op=entry["id"], status="done")

            ops = (_op("maya.add", "add", host="maya"), _op("demo.add", "add"))
            catalog = _catalog(ops, hosts={"maya": Host()})
            self.assertTrue(catalog.run("maya.add", {"a": 1}, scenes=[scene]).ok)
            self.assertEqual(seen, [[str(scene.resolve())]])
            missing = catalog.run("maya.add", {"a": 1}, scenes=[Path(tmp, "gone.ma")])
            self.assertIn("no such scene file", missing.error)
            here = catalog.run("demo.add", {"a": 1}, scenes=[scene])
            self.assertEqual(here.status, "failed")
            self.assertIn("opens no scenes", here.error)

    def test_a_registered_host_is_loaded_from_its_entry_point(self):
        class Runner(PureOperationHost):
            pass

        ep = SimpleNamespace(name="maya", value="x:Runner", load=lambda: Runner)
        ops = (_op("maya.add", "add", host="maya"),)
        with patch(_ENTRY_POINTS, return_value=[ep]):
            catalog = _catalog(ops)
            self.assertIsInstance(catalog.host("maya"), Runner)
            self.assertEqual(catalog.run("maya.add", {"a": 2}).result, 4)

    def test_a_host_runner_that_fails_to_load_is_refused_not_raised(self):
        def broken():
            raise ImportError("no maya here")

        ep = SimpleNamespace(name="maya", value="x:Runner", load=broken)
        with patch(_ENTRY_POINTS, return_value=[ep]):
            outcome = _catalog((_op("maya.add", "add", host="maya"),)).run("maya.add", {"a": 1})
        self.assertEqual(outcome.status, "failed")
        self.assertIn("the runner for host 'maya' failed to load: ImportError", outcome.error)

    def test_an_unknown_id_is_refused_naming_the_closest(self):
        outcome = _catalog().run("demo.ad")
        self.assertEqual(outcome.status, "failed")
        self.assertIn("did you mean demo.add", outcome.error)

    def test_a_schema_ref_the_catalog_does_not_define_is_an_error(self):
        entry = _op("demo.add", "add").describe()
        entry["args"]["properties"]["a"] = {"$ref": "#/$defs/Missing"}
        outcome = OperationCatalog([entry]).run("demo.add", {"a": 1})
        self.assertIn("names '#/$defs/Missing', which the catalog does not define", outcome.error)

    def test_an_invalid_entry_is_refused_when_the_catalog_is_built(self):
        with self.assertRaises(ValueError) as caught:
            OperationCatalog([{"id": "demo.x", "title": "X", "effect": "write", "args": {}}])
        self.assertIn("effect", str(caught.exception))

    def test_the_pure_host_satisfies_the_seam(self):
        self.assertIsInstance(PureOperationHost(), OperationHost)

    def test_the_outcome_round_trips_and_validates_as_its_schema(self):
        outcome = _catalog().run("demo.lint")
        data = json.loads(outcome.to_json())
        self.assertEqual(Outcome.validate(data).errors, [])
        self.assertEqual(Outcome.from_dict(data).to_dict(), data)
        schema = Outcome.json_schema()
        self.assertEqual(schema["title"], "Outcome")
        self.assertIn("Issue", schema["$defs"])
        self.assertIn("OperationEntry", OperationCatalogDocument.json_schema()["$defs"])


class _ProviderFixture(unittest.TestCase):
    """A temporary package on sys.path holding a declaration module."""

    PACKAGE = "opsfix_pkg"

    def setUp(self):
        _LOG.clear()
        self._tmp = tempfile.TemporaryDirectory()
        os.makedirs(os.path.join(self._tmp.name, self.PACKAGE))
        self._write("__init__.py", "")
        self._write_declarations('_op("demo.add", "add")')
        sys.path.insert(0, self._tmp.name)
        importlib.invalidate_caches()
        self.module = f"{self.PACKAGE}.operations"

    def tearDown(self):
        sys.path.remove(self._tmp.name)
        self._forget()
        self._tmp.cleanup()

    def _forget(self):
        for name in [m for m in sys.modules if m.split(".")[0] == self.PACKAGE]:
            del sys.modules[name]

    def _write(self, name: str, text: str):
        with open(os.path.join(self._tmp.name, self.PACKAGE, name), "w") as f:
            f.write(text)
        importlib.invalidate_caches()

    def _write_declarations(self, body: str, name: str = "operations"):
        self._write(
            f"{name}.py",
            textwrap.dedent(
                f"""\
                from pythontk import Operation

                def _op(op_id, name, **kwargs):
                    return Operation(op_id, "{__name__}:" + name, **kwargs)

                OPERATIONS = ({body},)
                """
            ),
        )
        sys.modules.pop(f"{self.PACKAGE}.{name}", None)


class TestOperationCatalogBuild(_ProviderFixture):
    def test_write_then_check_is_current_and_a_changed_declaration_is_stale(self):
        path = Path(OperationProvider.write_catalog(self.module))
        self.assertEqual((path.name, path.parent.name), ("operations.json", self.PACKAGE))
        document = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual((document["format"], document["provider"]), (1, self.PACKAGE))
        self.assertEqual([e["id"] for e in document["operations"]], ["demo.add"])
        self.assertEqual(OperationCatalogDocument.validate(document).errors, [])
        self.assertEqual(OperationProvider.problems(self.module), [])

        self._write_declarations('_op("demo.add", "add", effect="read", title="Add")')
        (problem,) = OperationProvider.problems(self.module)
        self.assertIn("is stale (demo.add)", problem)
        self.assertIn(f"operations.catalog module={self.module}", problem)

    def test_discover_reads_a_committed_catalog_without_importing_the_provider(self):
        OperationProvider.write_catalog(self.module)
        self._forget()
        ep = SimpleNamespace(name="opsfix", value=f"{self.module}:OPERATIONS")
        with patch(_ENTRY_POINTS, return_value=[ep]):
            catalog = OperationCatalog.discover()
        self.assertEqual(catalog.ids(), ["demo.add"])
        self.assertEqual(catalog.get("demo.add")["provider"], self.PACKAGE)
        self.assertNotIn(self.PACKAGE, sys.modules, "discover imported the provider")
        self.assertEqual(catalog.run("demo.add", {"a": 3}).result, 5)

    def test_an_unwritten_provider_and_an_explicit_module_are_built_live(self):
        ep = SimpleNamespace(name="opsfix", value=self.module)
        with patch(_ENTRY_POINTS, return_value=[ep]):
            self.assertEqual(OperationCatalog.discover().ids(), ["demo.add"])
        with patch(_ENTRY_POINTS, return_value=[]):
            self.assertEqual(OperationCatalog.discover([self.module]).ids(), ["demo.add"])

    def test_one_id_declared_twice_keeps_the_first_and_names_the_second(self):
        self._write_declarations('_op("demo.add", "add", effect="state")', name="other")
        eps = [
            SimpleNamespace(name="first", value=self.module),
            SimpleNamespace(name="second", value=f"{self.PACKAGE}.other"),
        ]
        with patch(_ENTRY_POINTS, return_value=eps):
            catalog = OperationCatalog.discover()
        self.assertEqual(catalog.get("demo.add")["effect"], "read")
        (problem,) = catalog.problems
        self.assertIn("declared by both 'first' and 'second'", problem)
        with self.assertRaises(ValueError):
            OperationCatalog([_op("demo.add", "add").describe()] * 2)

    def test_one_module_registered_twice_is_read_once(self):
        eps = [SimpleNamespace(name="opsfix", value=self.module)] * 2
        with patch(_ENTRY_POINTS, return_value=eps):
            catalog = OperationCatalog.discover()
        self.assertEqual((catalog.ids(), catalog.problems), (["demo.add"], []))

    def test_a_provider_that_fails_to_load_or_is_malformed_is_a_problem_not_a_crash(self):
        path = Path(OperationProvider.write_catalog(self.module))
        self._write("broken.py", "OPERATIONS = ()\n")
        (path.parent / "broken.json").write_text(
            json.dumps({"format": 1, "provider": "broken", "operations": [{"title": "no id"}]}),
            encoding="utf-8",
        )
        eps = [
            SimpleNamespace(name="missing", value="nosuch_ops_pkg.operations"),
            SimpleNamespace(name="broken", value=f"{self.PACKAGE}.broken"),
            SimpleNamespace(name="opsfix", value=self.module),
        ]
        with patch(_ENTRY_POINTS, return_value=eps):
            catalog = OperationCatalog.discover()
        self.assertEqual(catalog.ids(), ["demo.add"])
        missing, broken = catalog.problems
        self.assertIn("missing (nosuch_ops_pkg.operations): ModuleNotFoundError", missing)
        self.assertIn("catalog of 'broken' is invalid", broken)
        self.assertIn("missing required key 'id'", broken)

    def test_an_explicit_module_replaces_its_registered_provider(self):
        OperationProvider.write_catalog(self.module)
        self._write_declarations('_op("demo.add", "add", effect="state")')
        ep = SimpleNamespace(name="opsfix", value=self.module)
        with patch(_ENTRY_POINTS, return_value=[ep]):
            committed = OperationCatalog.discover()
            live = OperationCatalog.discover([self.module])
        self.assertEqual(committed.get("demo.add")["effect"], "read")
        self.assertEqual(live.get("demo.add")["effect"], "state")
        self.assertEqual(live.problems, [])

    def test_a_catalog_file_is_a_provider(self):
        # A catalog another writer leaves outside any package (a Unity
        # project's), named by its path; a C# writer may lead with a BOM.
        path = Path(self._tmp.name, "Library", "operations.json")
        path.parent.mkdir()
        document = OperationProvider.document(OperationProvider.declarations(self.module), "project")
        path.write_text("﻿" + json.dumps(document), encoding="utf-8")
        with patch(_ENTRY_POINTS, return_value=[]):
            catalog = OperationCatalog.discover([path])
            by_text = OperationCatalog.discover([str(path)])
            missing = OperationCatalog.discover([str(path.with_name("none.json"))])
        self.assertEqual((catalog.ids(), catalog.problems), (["demo.add"], []))
        self.assertEqual(catalog.get("demo.add")["provider"], "project")
        self.assertEqual(by_text.ids(), ["demo.add"])
        self.assertEqual(missing.ids(), [])
        (problem,) = missing.problems
        self.assertIn("none.json", problem)

    def test_a_declaration_module_lists_its_operations_in_one_attribute(self):
        with self.assertRaises(ValueError) as caught:
            OperationProvider.catalog_path(f"{self.module}:MORE")
        self.assertIn("'OPERATIONS', not 'MORE'", str(caught.exception))

    def test_a_newer_catalog_format_is_refused(self):
        path = Path(OperationProvider.write_catalog(self.module))
        document = json.loads(path.read_text(encoding="utf-8"))
        document["format"] = OperationCatalogDocument.FORMAT + 1
        path.write_text(json.dumps(document), encoding="utf-8")
        with self.assertRaises(ValueError) as caught:
            OperationCatalog.load(path)
        self.assertIn("upgrade pythontk", str(caught.exception))

    def test_search_matches_every_word(self):
        catalog = _catalog()
        self.assertEqual([e["id"] for e in catalog.search("rename objects")], ["demo.rename"])
        self.assertEqual(catalog.search("rename zebra"), [])


class TestOperationsCli(_ProviderFixture):
    def setUp(self):
        super().setUp()
        self._write_declarations(
            '_op("demo.add", "add"), _op("demo.rename", "rename", effect="state"), _op("demo.tag", "tag")'
        )
        self._patch = patch(_ENTRY_POINTS, return_value=[])
        self._patch.start()

    def tearDown(self):
        self._patch.stop()
        super().tearDown()

    def _cli(self, *argv):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = OperationCatalog.cli(list(argv) + ["--provider", self.module])
        return code, out.getvalue()

    def test_run_prints_the_outcome_and_exits_by_status(self):
        code, out = self._cli("run", "demo.add", "a=4", "b=1")
        self.assertEqual((code, json.loads(out)["result"]), (0, 5))
        code, out = self._cli("run", "demo.add", "--args", '{"a": "x"}')
        self.assertEqual((code, json.loads(out)["status"]), (1, "failed"))
        with self.assertRaises(SystemExit) as caught:
            self._cli("run", "demo.add", "--args", "{a: 1}")
        self.assertIn("--args is not JSON", str(caught.exception))

    def test_an_option_may_sit_between_the_id_and_its_arguments(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = OperationCatalog.cli(["run", "demo.add", "--provider", self.module, "a=4", "b=1"])
        self.assertEqual((code, json.loads(out.getvalue())["result"]), (0, 5))

    def test_a_string_argument_takes_its_text_as_given(self):
        code, out = self._cli("run", "demo.tag", "name=2024", "count=3")
        self.assertEqual(code, 0, out)

    def test_args_read_from_a_file(self):
        path = Path(self._tmp.name) / "args.json"
        path.write_text('{"a": 7}', encoding="utf-8")
        code, out = self._cli("run", "demo.add", "--args", f"@{path}")
        self.assertEqual((code, json.loads(out)["result"]), (0, 9))

    def test_run_plans_and_apply_changes(self):
        code, out = self._cli("run", "demo.rename", 'names=["a"]')
        self.assertEqual((code, json.loads(out)["dry_run"], _LOG), (0, True, []))
        code, out = self._cli("apply", "demo.rename", 'names=["a"]')
        self.assertEqual((code, _LOG), (0, ["a_new"]))

    def test_list_search_and_describe(self):
        code, out = self._cli("list")
        self.assertEqual(code, 0)
        self.assertIn("demo.add", out)
        self.assertIn("demo.rename  state", out)
        code, out = self._cli("search", "rename")
        self.assertNotIn("demo.add", out)
        code, out = self._cli("describe", "demo.add")
        self.assertEqual(json.loads(out)["args"]["required"], ["a"])

    def test_upkeep_runs_as_operations(self):
        code, out = self._cli(
            "apply", "operations.catalog", "module=nosuch_ops_pkg.operations",
            "--provider", "pythontk.operations",
        )
        self.assertEqual(code, 1)
        self.assertIn("nosuch_ops_pkg", json.loads(out)["error"])

    def test_an_error_issue_fails_the_command(self):
        # A run that finishes but reports an error (a check, a lint) exits 1,
        # as a linter does, so CI can gate on it.
        self._write(
            "checks.py",
            'def flag(ctx=None) -> None:\n    """Report one error."""\n    ctx.issue("demo.bad", "it is bad")\n',
        )
        self._write_declarations('_op("demo.lint", "lint"), Operation("demo.flag", "opsfix_pkg.checks:flag")')
        code, out = self._cli("run", "demo.flag")
        self.assertEqual((code, json.loads(out)["status"]), (1, "done"))
        code, _ = self._cli("run", "demo.lint")  # a warning does not
        self.assertEqual(code, 0)

    def test_python_m_pythontk_routes_ops(self):
        main = importlib.import_module("pythontk.__main__")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = main._main(["ops", "list", "--json", "--provider", self.module])
        self.assertEqual(code, 0)
        ids = [e["id"] for e in json.loads(out.getvalue())]
        self.assertEqual(ids, ["demo.add", "demo.rename", "demo.tag"])



# -- the batch protocol: a fresh Python process stands in for the application --
_BATCH_PACKAGE = "opsbatch_pkg"
_BATCH_SCENE = """\
import json
import os
import time

SCENE = {"path": None, "objects": [], "dirty": False}


class JsonSceneFiles:
    \"\"\"A JSON file of object names plays the scene.\"\"\"

    def open(self, paths):
        for path in paths:
            with open(path, encoding="utf-8") as f:
                SCENE.update(path=path, objects=json.load(f)["objects"], dirty=False)

    def modified(self):
        return [SCENE["path"]] if SCENE["path"] and SCENE["dirty"] else []

    def save(self, paths):
        for path in paths:
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"objects": SCENE["objects"]}, f)
        SCENE["dirty"] = False

    def close(self):
        pass


class ScriptedSceneFiles(JsonSceneFiles):
    \"\"\"As in Blender: its dirty flag misses an edit made from code, so a
    live run that changed something tells it (changed).\"\"\"

    def changed(self):
        SCENE["dirty"] = True


def rename_quietly(old: str, new: str, ctx=None) -> str:
    \"\"\"Rename one object without touching the dirty flag, as code edits do.\"\"\"
    index = SCENE["objects"].index(old)
    if ctx.should_change(old, "rename to " + new):
        SCENE["objects"][index] = new
    return new


class SlowSaveSceneFiles(JsonSceneFiles):
    \"\"\"Says it is saving, then takes its time before it writes.\"\"\"

    def save(self, paths):
        print("::progress:: 999/1000 saving", flush=True)
        time.sleep(3)
        super().save(paths)


def count() -> int:
    \"\"\"How many objects the scene holds.\"\"\"
    return len(SCENE["objects"])


def rename(old: str, new: str, ctx=None) -> str:
    \"\"\"Rename one object.\"\"\"
    index = SCENE["objects"].index(old)
    if ctx.should_change(old, "rename to " + new):
        SCENE["objects"][index] = new
        SCENE["dirty"] = True
    return new


def rename_slowly(old: str, new: str, ctx=None) -> str:
    \"\"\"Rename one object, taking a second over it.\"\"\"
    time.sleep(1)
    return rename(old, new, ctx=ctx)


def rename_while_saved_elsewhere(old: str, new: str, ctx=None) -> str:
    \"\"\"Rename one object while someone else saves the scene file.\"\"\"
    with open(SCENE["path"], "w", encoding="utf-8") as f:
        json.dump({"objects": ["theirs"]}, f)
    return rename(old, new, ctx=ctx)


def rename_then_fail(old: str, new: str, ctx=None) -> None:
    \"\"\"Rename one object, then fail.\"\"\"
    rename(old, new, ctx=ctx)
    raise RuntimeError("boom")


def wait_for_cancel(ctx=None) -> None:
    \"\"\"Report half way, then wait at checkpoints until cancelled.\"\"\"
    ctx.progress(0.5, "half")
    SCENE["dirty"] = True
    for _ in range(600):
        ctx.check()
        time.sleep(0.05)


def die() -> None:
    \"\"\"End the process before any outcome is written.\"\"\"
    os._exit(3)


def stubborn(ctx=None) -> None:
    \"\"\"Report half way, then work on without a checkpoint.\"\"\"
    ctx.progress(0.5, "half")
    time.sleep(120)
"""


class TestBatchOperationHost(unittest.TestCase):
    """The batch protocol end to end: request file in, fresh process, outcome out."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        (root / _BATCH_PACKAGE).mkdir()
        (root / _BATCH_PACKAGE / "__init__.py").write_text("", encoding="utf-8")
        (root / _BATCH_PACKAGE / "scene.py").write_text(_BATCH_SCENE, encoding="utf-8")
        sys.path.insert(0, self._tmp.name)
        importlib.invalidate_caches()
        self.scene = root / "shot.json"
        self.scene.write_text(json.dumps({"objects": ["a", "b"]}), encoding="utf-8")

    def tearDown(self):
        sys.path.remove(self._tmp.name)
        for name in [m for m in sys.modules if m.split(".")[0] == _BATCH_PACKAGE]:
            del sys.modules[name]
        self._tmp.cleanup()

    def _catalog(self, scene_files=f"{_BATCH_PACKAGE}.scene:JsonSceneFiles", app=None):
        def op(name, **kwargs):
            return Operation(f"py.{name}", f"{_BATCH_PACKAGE}.scene:{name}", host="py", **kwargs)

        ops = (
            op("count"),
            op("rename", effect="state"),
            op("rename_slowly", effect="state"),
            op("rename_while_saved_elsewhere", effect="state"),
            op("rename_then_fail", effect="state"),
            op("wait_for_cancel", effect="state"),
            op("die"),
            op("stubborn", effect="state"),
            op("rename_quietly", effect="state"),
        )
        host = BatchOperationHost(
            app or sys.executable, scene_files=scene_files, timeout=120, grace=self.grace
        )
        return OperationCatalog([o.describe() for o in ops], hosts={"py": host})

    grace = 60.0

    def _objects(self):
        return json.loads(self.scene.read_text(encoding="utf-8"))["objects"]

    def test_a_read_runs_in_a_fresh_process_on_the_opened_scene(self):
        outcome = self._catalog().run("py.count", scenes=[self.scene])
        self.assertEqual((outcome.status, outcome.result), ("done", 2), outcome.error)
        self.assertEqual(outcome.changes, [])

    def test_a_dry_run_plans_the_change_and_the_save_and_leaves_the_file(self):
        outcome = self._catalog().run("py.rename", {"old": "a", "new": "z"}, scenes=[self.scene])
        self.assertTrue(outcome.ok and outcome.dry_run, outcome.error)
        self.assertEqual(
            [(c.kind, c.target, c.action) for c in outcome.changes],
            [("change", "a", "rename to z"), ("write", str(self.scene), "save the scene")],
        )
        self.assertEqual(self._objects(), ["a", "b"])

    def test_a_live_run_saves_the_scene_it_changed(self):
        outcome = self._catalog().run(
            "py.rename", {"old": "a", "new": "z"}, apply=True, scenes=[self.scene]
        )
        self.assertTrue(outcome.ok, outcome.error)
        self.assertEqual(outcome.changes[-1].target, str(self.scene))
        self.assertEqual(self._objects(), ["z", "b"])

    def test_a_host_whose_flag_misses_code_edits_is_told_so_the_change_is_saved(self):
        # Blender's dirty flag stays clear after a data-API edit; the run tells it.
        outcome = self._catalog(f"{_BATCH_PACKAGE}.scene:ScriptedSceneFiles").run(
            "py.rename_quietly", {"old": "a", "new": "z"}, apply=True, scenes=[self.scene]
        )
        self.assertTrue(outcome.ok, outcome.error)
        self.assertEqual(self._objects(), ["z", "b"])

    def test_a_run_that_does_not_finish_saves_nothing(self):
        outcome = self._catalog().run(
            "py.rename_then_fail", {"old": "a", "new": "z"}, apply=True, scenes=[self.scene]
        )
        self.assertEqual(outcome.status, "partial")
        self.assertIn("RuntimeError: boom", outcome.error)
        self.assertIn("not saved, so as they were", outcome.error)
        self.assertEqual(self._objects(), ["a", "b"])

    def test_cancelling_stops_the_run_at_a_checkpoint_and_saves_nothing(self):
        scope, progress = CancelScope("test"), []

        def on_progress(fraction, message):
            progress.append((fraction, message))
            scope.cancel("user")

        outcome = self._catalog().run(
            "py.wait_for_cancel",
            apply=True,
            scenes=[self.scene],
            scope=scope,
            on_progress=on_progress,
        )
        self.assertEqual(outcome.status, "cancelled", outcome.error)
        self.assertEqual(progress, [(0.5, "half")])
        self.assertEqual(self._objects(), ["a", "b"])

    def test_an_ambient_cancel_stops_the_run_at_a_checkpoint_never_by_a_kill(self):
        scope = CancelScope("panel")

        def on_progress(fraction, message):
            scope.cancel("Esc")

        with scope.activate():  # a panel's @Cancelable: the AMBIENT scope
            outcome = self._catalog().run(
                "py.wait_for_cancel", apply=True, scenes=[self.scene], on_progress=on_progress
            )
        self.assertEqual(outcome.status, "cancelled", outcome.error)
        # The instance wrote this outcome itself; a kill would say so instead.
        self.assertNotIn("instance", outcome.error)
        self.assertEqual(self._objects(), ["a", "b"])

    def test_a_run_that_ignores_the_cancel_is_killed_after_the_grace(self):
        self.grace = 1.0
        for apply, status in ((False, "cancelled"), (True, "partial")):
            scope = CancelScope("test")
            outcome = self._catalog().run(
                "py.stubborn",
                apply=apply,
                scenes=[self.scene],
                scope=scope,
                on_progress=lambda f, m: scope.cancel("user"),
            )
            # A kill is no checkpoint: a live change it cut short may have done anything.
            self.assertEqual(outcome.status, status, outcome.error)
            self.assertIn("did not stop within 1 s of the cancel", outcome.error)

    def test_a_cancel_never_cuts_a_save_short(self):
        self.grace = 1.0
        scope = CancelScope("test")
        outcome = self._catalog(scene_files=f"{_BATCH_PACKAGE}.scene:SlowSaveSceneFiles").run(
            "py.rename",
            {"old": "a", "new": "z"},
            apply=True,
            scenes=[self.scene],
            scope=scope,
            on_progress=lambda f, m: scope.cancel("user"),  # lands while it saves
        )
        self.assertEqual(outcome.status, "done", outcome.error)
        self.assertEqual(self._objects(), ["z", "b"])

    def test_two_live_runs_on_one_scene_take_turns(self):
        catalog, outcomes = self._catalog(), []

        def apply(old, new):
            outcomes.append(
                catalog.run("py.rename_slowly", {"old": old, "new": new}, apply=True, scenes=[self.scene])
            )

        threads = [threading.Thread(target=apply, args=pair) for pair in (("a", "z"), ("b", "y"))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual([o.status for o in outcomes], ["done", "done"], [o.error for o in outcomes])
        self.assertEqual(self._objects(), ["z", "y"])  # neither saved over the other

    def test_a_scene_saved_elsewhere_during_the_run_is_not_saved_over(self):
        outcome = self._catalog().run(
            "py.rename_while_saved_elsewhere", {"old": "a", "new": "z"}, apply=True, scenes=[self.scene]
        )
        self.assertEqual(outcome.status, "failed", outcome.error)
        self.assertIn("changed on disk since this run opened it", outcome.error)
        self.assertEqual(self._objects(), ["theirs"])

    def test_an_outcome_written_before_a_hung_exit_stands(self):
        serve = (
            "import json, pathlib, sys, time; run = pathlib.Path(sys.argv[1]); "
            "(run / 'outcome.json').write_text(json.dumps({'op': 'py.add', 'status': 'done', 'result': 1})); "
            "time.sleep(60)"
        )
        host = BatchOperationHost(sys.executable, command=lambda run: ["-c", serve, str(run)], timeout=3)
        catalog = OperationCatalog([_op("py.add", "add", host="py").describe()], hosts={"py": host})
        outcome = catalog.run("py.add", {"a": 1})
        self.assertEqual((outcome.status, outcome.result), ("done", 1))
        self.assertIn("the instance did not exit: it timed out after 3 s", outcome.error)

    def test_a_process_that_dies_without_an_outcome_is_reported_not_raised(self):
        outcome = self._catalog().run("py.die")
        self.assertEqual(outcome.status, "failed")
        self.assertIn("exited without an outcome (exit code 3)", outcome.error)

    def test_a_host_with_no_scenes_refuses_one(self):
        outcome = self._catalog(scene_files="pythontk:PureSceneFiles").run(
            "py.count", scenes=[self.scene]
        )
        self.assertEqual(outcome.status, "failed")
        self.assertIn("this host opens no scenes", outcome.error)
        self.assertIsInstance(PureSceneFiles(), SceneFiles)

    def test_a_missing_application_is_refused(self):
        outcome = self._catalog(app=lambda: None).run("py.count")
        self.assertEqual(outcome.status, "failed")
        self.assertIn("cannot launch 'py'", outcome.error)

    def test_serve_refuses_a_request_of_a_newer_format(self):
        run = Path(self._tmp.name, "run")
        run.mkdir()
        request = OpRequest(op="py.count").to_dict()
        (run / OpRequest.REQUEST_FILE).write_text(
            json.dumps(dict(request, format=OpRequest.FORMAT + 1)), encoding="utf-8"
        )
        self.assertEqual(BatchOperationHost.serve(run), 1)
        outcome = json.loads((run / OpRequest.OUTCOME_FILE).read_text(encoding="utf-8"))
        self.assertEqual(outcome["status"], "failed")
        self.assertIn("request format 2", outcome["error"])

    def test_an_application_that_serves_the_folder_itself_takes_a_command(self):
        serve = (
            "import json, pathlib, sys; run = pathlib.Path(sys.argv[1]); "
            "req = json.loads((run / 'request.json').read_text()); "
            "print('::progress:: 250/1000 quarter', flush=True); "
            "(run / 'outcome.json').write_text(json.dumps("
            "{'op': req['op'], 'status': 'done', 'result': req['args']['x'] * 2}))"
        )
        progress = []
        host = BatchOperationHost(sys.executable, command=lambda run: ["-c", serve, str(run)])
        # A C# entry's target names no Python module: nothing is imported for it.
        entry = dict(_op("py.add", "add", host="py").describe(), target="M3trik.Unitytk:Add")
        entry["args"] = {"type": "object", "properties": {"x": {"type": "integer"}}}
        catalog = OperationCatalog([entry], hosts={"py": host})
        outcome = catalog.run("py.add", {"x": 4}, on_progress=lambda f, m: progress.append((f, m)))
        self.assertEqual((outcome.status, outcome.result), ("done", 8), outcome.error)
        self.assertIn((0.25, "quarter"), progress)

    def test_diagnose_reads_why_no_outcome_came(self):
        host = BatchOperationHost(
            sys.executable,
            command=lambda run: ["-c", "print('LICENSE NOT FOUND'); raise SystemExit(198)"],
            diagnose=lambda code, tail: "no licence" if code == 198 and "LICENSE" in tail else None,
        )
        catalog = OperationCatalog([_op("py.add", "add", host="py").describe()], hosts={"py": host})
        outcome = catalog.run("py.add", {"a": 1})
        self.assertEqual(outcome.status, "failed")
        self.assertIn("exited without an outcome (exit code 198): no licence", outcome.error)

    def test_the_request_round_trips_and_validates_as_its_schema(self):
        request = OpRequest(op="py.count", args={"x": 1}, dry_run=True, scenes=["s.ma"])
        data = request.to_dict()
        self.assertEqual(OpRequest.validate(data).errors, [])
        self.assertEqual(OpRequest.from_dict(data), request)
        self.assertEqual(OpRequest.json_schema()["required"], ["format", "op"])


class TestOperationProvider(_ProviderFixture):
    """A package's upkeep: its catalog, the rules, and what it could declare."""

    def setUp(self):
        super().setUp()
        self._patch = patch(_ENTRY_POINTS, return_value=[])
        self._patch.start()

    def tearDown(self):
        self._patch.stop()
        super().tearDown()

    def test_writing_a_catalog_plans_first(self):
        path = OperationProvider.catalog_path(self.module)
        ctx = OpContext(dry_run=True)
        self.assertEqual(OperationProvider.write_catalog(self.module, ctx=ctx), str(path))
        self.assertFalse(path.exists())
        self.assertEqual([c.kind for c in ctx.changes], ["write"])
        OperationProvider.write_catalog(self.module)
        again = OpContext()
        OperationProvider.write_catalog(self.module, ctx=again)
        self.assertEqual(again.changes, [])  # current: nothing to write

    def test_a_change_that_cannot_plan_or_an_untyped_argument_is_a_problem(self):
        self._write(
            "loose.py",
            "from typing import Any\n\n\ndef take(thing: Any) -> None:\n    \"\"\"Take anything.\"\"\"\n",
        )
        self._write_declarations(
            '_op("demo.wipe", "wipe", effect="files"), Operation("demo.take", "opsfix_pkg.loose:take")'
        )
        OperationProvider.write_catalog(self.module)
        take, wipe = OperationProvider.problems(self.module)  # in id order
        self.assertIn("demo.take: argument 'thing' has no type", take)
        self.assertIn("demo.wipe changes files but cannot plan", wipe)

    def test_code_runs_with_no_plan_and_a_code_operation_that_plans_is_a_problem(self):
        # Code cannot plan, so `code` is the one change with no dry run; a
        # `files` change with none is still a problem, and so is `code` that
        # takes a context (its plan would be the code echoed back).
        self._write(
            "consoles.py",
            "def run(code: str) -> dict:\n    \"\"\"Run code.\"\"\"\n    return {}\n\n\n"
            "def run_planned(code: str, ctx=None) -> dict:\n    \"\"\"Run code, planned.\"\"\"\n    return {}\n",
        )
        self._write_declarations(
            'Operation("demo.exec", "opsfix_pkg.consoles:run", effect="code"), '
            'Operation("demo.exec_planned", "opsfix_pkg.consoles:run_planned", effect="code"), '
            '_op("demo.wipe", "wipe", effect="files")'
        )
        OperationProvider.write_catalog(self.module)
        problems = OperationProvider.problems(self.module)
        self.assertEqual(len(problems), 2, problems)
        self.assertIn("demo.exec_planned runs code, which has no plan", problems[0])
        self.assertIn("demo.wipe changes files but cannot plan", problems[1])
        entry = Operation("demo.exec", "opsfix_pkg.consoles:run", effect="code").describe()
        (tool,) = OperationServer(OperationCatalog([entry]), apply=True).tools()
        self.assertIn("Runs code", tool["description"])
        self.assertEqual(OperationServer(OperationCatalog([entry])).tools(), [])  # never a plan

    def test_twins_share_their_arguments_and_effect(self):
        self._write(
            "hosts.py",
            textwrap.dedent(
                '''\
                def a_rename(material: str, name: str, ctx=None) -> str:
                    """Rename a material."""
                    return name


                def b_rename(material: str, name: str, scene: str = "", ctx=None) -> str:
                    """Rename a material in one scene."""
                    return name


                def b_retitle(mat: str, name: str, ctx=None) -> str:
                    """Rename a material."""
                    return name
                '''
            ),
        )
        good = (
            'Operation("alpha.mat.rename", "opsfix_pkg.hosts:a_rename", effect="state", host="alpha"), '
            'Operation("beta.mat.rename", "opsfix_pkg.hosts:b_rename", effect="state", host="beta")'
        )
        self._write_declarations(good)
        OperationProvider.write_catalog(self.module)
        self.assertEqual(OperationProvider.check([self.module]), [])  # a host may add its own
        self._write_declarations(
            good.replace('hosts:b_rename", effect="state"', 'hosts:b_retitle", effect="files"')
        )
        OperationProvider.write_catalog(self.module)
        (problem,) = OperationProvider.check([self.module])
        self.assertIn("twins alpha.mat.rename and beta.mat.rename differ", problem)
        self.assertIn("effect state vs files", problem)
        self.assertIn("material", problem)

    def test_a_catalog_another_language_writes_is_held_to_the_rules(self):
        good = {
            "id": "demo.cs_scale",
            "title": "Scale",
            "effect": "state",
            "dry_run": True,
            "args": {"type": "object", "properties": {"factor": {"type": "number"}}},
        }
        path = Path(self._tmp.name, self.PACKAGE, "csharp_ops.json")
        document = {"format": 1, "provider": "unityproj", "operations": [good]}
        path.write_text(json.dumps(document), encoding="utf-8")
        self.assertEqual(OperationProvider.check([f"{self.PACKAGE}.csharp_ops"]), [])  # registered by module path
        # Named by its file: an absolute path with a drive letter (a ':') and
        # dotted folders, as a Unity project's Library/ catalog is.
        project = Path(self._tmp.name, "My.Project", "Library", "unitytk")
        project.mkdir(parents=True)
        named = project / "operations.json"
        named.write_text(json.dumps(document), encoding="utf-8")
        self.assertEqual(OperationProvider.check([str(named.resolve())]), [])
        document["operations"].append(
            dict(good, id="demo.cs_take", dry_run=False, args={"type": "object", "properties": {"thing": {}}})
        )
        path.write_text(json.dumps(document), encoding="utf-8")
        plan, take = OperationProvider.check([str(path)])  # no "stale": nothing here rebuilds it
        self.assertIn("demo.cs_take changes state but cannot plan", plan)
        self.assertNotIn("ctx is None", plan)  # a C# entry: told in no one language's words
        self.assertIn("demo.cs_take: argument 'thing' has no type", take)
        path.write_text('{"format": 1, "provider": "x", "operations": [{"title": "no id"}]}', encoding="utf-8")
        (problem,) = OperationProvider.check([str(path)])
        self.assertIn("missing required key 'id'", problem)

    def test_check_reports_each_problem_as_an_issue_with_its_fix(self):
        ctx = OpContext()  # the catalog was never written
        (problem,) = OperationProvider.check([self.module], ctx=ctx)
        (issue,) = ctx.issues
        self.assertEqual((issue.code, issue.severity, issue.message), ("operations.stale", "error", problem))
        self.assertEqual((issue.fix.op, issue.fix.args), ("operations.catalog", {"module": self.module}))
        ep = SimpleNamespace(name="opsfix", value=self.module)
        with patch(_ENTRY_POINTS, return_value=[ep]):
            self.assertEqual(OperationProvider.check(), [problem])  # every registered provider

    def test_a_package_hidden_by_its_repo_folder_is_named(self):
        # A folder named after the package on sys.path (its repo) makes it
        # import as an empty namespace: every target in it then fails to
        # resolve with a bare AttributeError. Name the cause and the fix.
        with tempfile.TemporaryDirectory() as tmp:
            inner = Path(tmp, "hiddenfix", "hiddenfix")
            inner.mkdir(parents=True)
            (inner / "__init__.py").write_text("", encoding="utf-8")
            sys.path.insert(0, tmp)
            importlib.invalidate_caches()
            try:
                (problem,) = OperationProvider.check(["hiddenfix.operations"])
                self.assertIn("hiddenfix imports as an empty namespace", problem)
                self.assertIn(str(inner.parent), problem)
                with self.assertRaisesRegex(ValueError, "empty namespace"):
                    OperationProvider.init("hiddenfix")
            finally:
                sys.path.remove(tmp)
                sys.modules.pop("hiddenfix", None)

class TestProviderInit(unittest.TestCase):
    """Onboarding a package: its declaration module, its entry point, its catalog."""

    PACKAGE = "newfix_pkg"

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name, "outer", "newfix")
        (self.root / self.PACKAGE).mkdir(parents=True)
        (self.root / self.PACKAGE / "__init__.py").write_text("", encoding="utf-8")
        self.toml = self.root / "pyproject.toml"
        self.toml.write_text('[project]\nname = "newfix"\n', encoding="utf-8")
        sys.path.insert(0, str(self.root))
        importlib.invalidate_caches()

    def tearDown(self):
        sys.path.remove(str(self.root))
        for name in [m for m in sys.modules if m.split(".")[0] == self.PACKAGE]:
            del sys.modules[name]
        self._tmp.cleanup()

    def test_init_plans_then_scaffolds_once(self):
        ctx = OpContext(dry_run=True)
        OperationProvider.init(self.PACKAGE, ctx=ctx)
        self.assertEqual([c.kind for c in ctx.changes], ["create", "change", "write"])
        self.assertFalse((self.root / self.PACKAGE / "operations.py").exists())

        result = OperationProvider.init(self.PACKAGE)
        self.assertIn(
            f'[project.entry-points."pythontk.operations"]\n{self.PACKAGE} = "{self.PACKAGE}.operations"\n',
            self.toml.read_text(encoding="utf-8"),
        )
        document = json.loads((self.root / self.PACKAGE / "operations.json").read_text(encoding="utf-8"))
        self.assertEqual((document["provider"], document["operations"]), (self.PACKAGE, []))
        self.assertIn(f"pip install -e {self.root}", result["reinstall"])
        self.assertEqual(OperationProvider.problems(f"{self.PACKAGE}.operations"), [])

        again = OpContext()
        OperationProvider.init(self.PACKAGE, ctx=again)
        self.assertEqual(again.changes, [])  # nothing left to do

    def test_init_joins_an_existing_entry_point_table(self):
        self.toml.write_text(
            '[project]\nname = "newfix"\n\n[project.entry-points."pythontk.operations"]\n'
            'other = "other.operations"\n\n[tool.x]\ny = 1\n',
            encoding="utf-8",
        )
        OperationProvider.init(self.PACKAGE)
        text = self.toml.read_text(encoding="utf-8")
        self.assertIn(
            f'other = "other.operations"\n{self.PACKAGE} = "{self.PACKAGE}.operations"\n\n[tool.x]', text
        )
        self.assertEqual(text.count("pythontk.operations\"]"), 1)

    def test_init_needs_a_package_and_its_own_pyproject(self):
        with self.assertRaisesRegex(ValueError, "no package"):
            OperationProvider.init("nosuch_pkg_xyz")
        self.toml.replace(Path(self._tmp.name, "pyproject.toml"))  # an unrelated one, further up
        with self.assertRaisesRegex(ValueError, "pyproject.toml"):
            OperationProvider.init(self.PACKAGE)
        self.assertFalse((self.root / self.PACKAGE / "operations.py").exists())


class TestCatalogLocation(unittest.TestCase):
    """A committed catalog is found without importing its provider."""

    def test_a_repo_folder_shadowing_its_package_is_looked_through(self):
        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp, "shadowpkg_repo", "shadowpkg_repo")
            package.mkdir(parents=True)
            (package / "__init__.py").write_text("", encoding="utf-8")
            (package / "operations.py").write_text("OPERATIONS = ()", encoding="utf-8")
            (package / "csharp_ops.json").write_text("{}", encoding="utf-8")
            sys.path.insert(0, tmp)  # the repo folder imports as an empty namespace
            importlib.invalidate_caches()
            try:
                self.assertEqual(
                    OperationProvider.catalog_path("shadowpkg_repo.operations"),
                    package / "operations.json",
                )
                self.assertEqual(
                    OperationProvider.catalog_path("shadowpkg_repo.csharp_ops"),
                    package / "csharp_ops.json",
                )
            finally:
                sys.path.remove(tmp)
                sys.modules.pop("shadowpkg_repo", None)


class TestOperationServer(unittest.TestCase):
    """The catalog as MCP tools: plain data, no SDK needed."""

    def setUp(self):
        _LOG.clear()
        self.calls = []
        test = self

        class Host:
            def run(self, entry, request, scope=None, on_progress=None):
                test.calls.append((entry["id"], dict(request.args), request.dry_run, request.scenes))
                return Outcome(op=entry["id"], status="done")

        ops = FIXTURE_OPS + (_op("maya.rename", "rename", effect="state", host="maya"),)
        self.catalog = _catalog(ops, hosts={"maya": Host()})

    def test_the_plain_server_serves_reads_and_dry_runs_the_apply_server_changes(self):
        plain = {t["name"]: t for t in OperationServer(self.catalog).tools()}
        apply = {t["name"]: t for t in OperationServer(self.catalog, apply=True).tools()}
        self.assertIn("demo_add", plain)
        self.assertIn("demo_rename", plain)  # it can plan
        self.assertNotIn("demo_wipe", plain)  # it cannot plan: apply only
        self.assertEqual(set(apply), {"demo_rename", "demo_wipe", "demo_wipe_all", "demo_explode", "maya_rename"})
        self.assertTrue(plain["demo_rename"]["annotations"]["readOnlyHint"])
        self.assertTrue(apply["demo_rename"]["annotations"]["destructiveHint"])
        self.assertIn("Plans only", plain["demo_rename"]["description"])
        self.assertEqual(plain["demo_add"]["inputSchema"]["required"], ["a"])
        self.assertEqual(plain["demo_add"]["outputSchema"]["required"], ["op", "status"])
        self.assertIn("scenes", plain["maya_rename"]["inputSchema"]["properties"])
        self.assertNotIn("scenes", plain["demo_add"]["inputSchema"]["properties"])
        json.dumps(list(plain.values()))

    def test_a_tool_carries_a_terse_outcome_schema(self):
        (tool,) = OperationServer(_catalog((_op("demo.scan", "scan"),))).tools()
        # The instructions describe the Outcome once; each tool keeps its shape only.
        text = json.dumps(tool["outputSchema"])
        self.assertNotIn('"description"', text)
        self.assertEqual(tool["outputSchema"]["properties"]["status"]["enum"], list(Outcome.STATUSES))
        self.assertLess(len(text), len(json.dumps(Outcome.json_schema())) / 2)

    def test_a_server_that_serves_search_reads_the_packages_while_the_client_starts(self):
        api = _catalog(FIXTURE_OPS + (Operation("api.search", "pythontk:ApiSurface.search"),))
        read = []
        with patch.object(ApiSurface, "registered", return_value=["alpha", "beta"]), patch.object(
            ApiSurface, "read", side_effect=lambda p: read.append(p) or (1 / 0 if p == "alpha" else None)
        ):
            OperationServer(api).warm().join(10)
        self.assertEqual(read, ["alpha", "beta"])  # one that fails costs no other
        self.assertIsNone(OperationServer(api, apply=True).warm())  # it serves no search
        self.assertIsNone(OperationServer(_catalog()).warm())

    def test_a_call_plans_or_applies_by_server_and_hands_the_scenes_to_the_host(self):
        plan = OperationServer(self.catalog).call("demo_rename", {"names": ["a"]})
        self.assertTrue(plan.dry_run)
        self.assertEqual(_LOG, [])
        done = OperationServer(self.catalog, apply=True).call("demo_rename", {"names": ["a"]})
        self.assertFalse(done.dry_run)
        self.assertEqual(_LOG, ["a_new"])
        with tempfile.TemporaryDirectory() as tmp:
            scene = Path(tmp, "a.ma")
            scene.write_text("x", encoding="utf-8")
            OperationServer(self.catalog).call(
                "maya_rename", {"names": ["a"], "scenes": [str(scene)]}
            )
        self.assertEqual(self.calls, [("maya.rename", {"names": ["a"]}, True, [str(scene.resolve())])])

    def test_an_unknown_tool_or_bad_scenes_are_refused(self):
        server = OperationServer(self.catalog)
        self.assertIn("no tool 'demo_nope'", server.call("demo_nope").error)
        self.assertIn("must be a list", server.call("maya_rename", {"names": [], "scenes": "a"}).error)
        self.assertIn("no tool 'demo_wipe'", server.call("demo_wipe", {"path": "x"}).error)

    def test_two_ids_on_one_tool_name_keep_the_first_and_report_the_second(self):
        ops = (_op("demo.a_b", "add"), _op("demo_a.b", "add"), _op("demo.add", "add"))
        server = OperationServer(_catalog(ops))  # one clash never stops the server
        self.assertEqual([t["name"] for t in server.tools()], ["demo_a_b", "demo_add"])
        self.assertEqual(len(server.problems), 1)
        self.assertIn("both be served as tool 'demo_a_b'", server.problems[0])

    def test_the_instructions_name_the_hosts_it_serves(self):
        self.assertIn("(maya_)", OperationServer(self.catalog).instructions())
        plain = OperationServer(_catalog(FIXTURE_OPS)).instructions()
        self.assertNotIn("FRESH instance", plain)
        self.assertIn("Every result is an Outcome", plain)
        self.assertNotIn("api_search", plain)
        api = _catalog(FIXTURE_OPS + (Operation("api.search", "pythontk:ApiSurface.search"),))
        self.assertIn("`api_search` finds any public function", OperationServer(api).instructions())

    def test_a_progress_token_of_zero_is_a_token(self):
        self.assertEqual(OperationServer._progress_token({"progressToken": 0}), 0)
        self.assertIsNone(OperationServer._progress_token({}))


@unittest.skipUnless(importlib.util.find_spec("mcp"), "the MCP SDK is not installed (pythontk[mcp])")
class TestOperationServerStdio(_ProviderFixture):
    """The server end to end over stdio, spoken as a client does (JSON-RPC lines)."""

    def test_list_and_call_over_stdio(self):
        import subprocess

        env = dict(os.environ, PYTHONPATH=os.pathsep.join([self._tmp.name] + sys.path))
        proc = subprocess.Popen(
            [sys.executable, "-m", "pythontk", "ops", "mcp", "--provider", self.module],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            env=env,
        )
        try:

            def send(message):
                proc.stdin.write(json.dumps(dict(message, jsonrpc="2.0")) + "\n")
                proc.stdin.flush()

            def reply(request_id):
                for line in proc.stdout:
                    message = json.loads(line)
                    if message.get("id") == request_id:
                        return message
                self.fail("the server closed without replying")

            send(
                {
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-06-18",
                        "capabilities": {},
                        "clientInfo": {"name": "test", "version": "1"},
                    },
                }
            )
            self.assertIn("instructions", reply(1)["result"])
            send({"method": "notifications/initialized"})
            send({"id": 2, "method": "tools/list"})
            self.assertIn("demo_add", [t["name"] for t in reply(2)["result"]["tools"]])
            send({"id": 3, "method": "tools/call", "params": {"name": "demo_add", "arguments": {"a": 5}}})
            result = reply(3)["result"]
            self.assertFalse(result.get("isError"))
            self.assertEqual(result["structuredContent"]["result"], 7)
        finally:
            proc.stdin.close()
            proc.wait(timeout=30)


if __name__ == "__main__":
    unittest.main()
