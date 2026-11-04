# !/usr/bin/python
# coding=utf-8
"""Command-line entry to HelpMixin introspection.

    python -m pythontk <dotted.path> [member] [options]

Resolve a class or object by dotted path and print its help, source, location,
or signature - the same introspection ``HelpMixin`` exposes in-process, made
reachable from a shell or an agent without writing a REPL snippet. Unlike the
static read (``api.search``) it reads the *live* object, so ``--source``/``--where``
report what is actually loaded. ``--index`` lists what pythontk's root
publishes (``ApiSurface.exports``), one valid target per row.

Everything in this module is private on purpose: it is a CLI shell, not public
API. ``__main__.py`` is also skipped outright by the static read, so the
public surface is unaffected.

``python -m pythontk ops <command>`` is the operations front end instead
(``OperationCatalog.cli``): list, describe, search, run, apply, mcp.

Examples:
    python -m pythontk --index
    python -m pythontk pythontk.CoreUtils
    python -m pythontk pythontk.CoreUtils listify --json
    python -m pythontk pythontk.FileUtils get_file_contents --source
    python -m pythontk pythontk.CoreUtils --members methods --brief
"""

import importlib
import json
import sys

from pythontk.core_utils.api_surface.api_surface import ApiSurface
from pythontk.core_utils.cli import CLI
from pythontk.core_utils.help_mixin import HelpMixin


def _resolve(dotted: str):
    """Resolve a dotted path to a live object.

    Imports the longest importable module prefix, then ``getattr``-walks the
    remainder (so ``pythontk.CoreUtils`` and
    ``pythontk.core_utils._core_utils.CoreUtils`` both resolve). A bare name
    falls back to the pythontk public namespace.
    """
    parts = dotted.split(".")
    for i in range(len(parts), 0, -1):
        candidate = ".".join(parts[:i])
        try:
            obj = importlib.import_module(candidate)
        except ModuleNotFoundError as exc:
            # Only a genuinely absent candidate should fall through to a shorter
            # prefix. A ModuleNotFoundError naming something *else* means the
            # module exists but its own imports are broken - reporting the target
            # as merely "unresolvable" would bury the actual cause.
            if exc.name and not candidate.startswith(exc.name):
                raise
            continue
        try:
            for attr in parts[i:]:
                obj = getattr(obj, attr)
        except AttributeError:
            break
        else:
            return obj
    pkg = importlib.import_module("pythontk")
    if hasattr(pkg, dotted):
        return getattr(pkg, dotted)
    raise SystemExit(f"could not resolve '{dotted}'")


def _print_index(as_json: bool = False) -> int:
    """List what pythontk's root publishes, resolved live.

    Two tiers, resolved from the live package by :meth:`ApiSurface.exports`, so
    the listing exists wherever the wheel is installed and can never drift from
    what actually imports: the advertised ``__all__`` symbols (tier ``root``),
    and the wildcard-exposed bare aliases like ``filter_list`` (tier ``bare``).
    Each row's location is itself a valid ``python -m pythontk`` target.
    """
    rows = ApiSurface.exports("pythontk")
    if as_json:
        print(json.dumps(rows, indent=2))
        return 0
    width = max((len(r["name"]) for r in rows), default=0)
    for r in rows:
        location = (
            f"{r['module']}.{r['qualname']}"
            if r["module"] and r["qualname"]
            else r["module"] or ""
        )
        print(f"{r['name']:<{width}}  {location}  [{r['kind']}]")
    return 0


def _main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ["ops"]:
        # The operations front end (OperationCatalog.cli): list, describe,
        # search, run, apply, mcp. Routed before the introspection parser,
        # whose positional target would read "ops" as a dotted path.
        from pythontk import OperationCatalog

        return OperationCatalog.cli(argv[1:])
    parser = CLI.get_parser("Introspect a pythontk/ecosystem class or object.")
    parser.add_argument(
        "target", nargs="?", help="dotted path, e.g. pythontk.CoreUtils"
    )
    parser.add_argument("member", nargs="?", help="member name on the target class")
    parser.add_argument(
        "--index",
        action="store_true",
        help="list every root-exported symbol with its defining module",
    )
    parser.add_argument("--json", action="store_true", help="structured JSON output")
    parser.add_argument("--source", action="store_true", help="print source code")
    parser.add_argument("--where", action="store_true", help="print file:line location")
    parser.add_argument(
        "--signature", action="store_true", help="print signature detail"
    )
    parser.add_argument("--brief", action="store_true", help="one-line summaries")
    parser.add_argument(
        "--members",
        help="filter: methods|properties|classmethods|staticmethods",
    )
    args = parser.parse_args(argv)

    if args.index:
        if args.target:
            # --index lists the whole public surface; a target asks about one
            # thing. Silently dropping the target would leave a typo'd
            # invocation looking like it ran the (unrelated) command asked
            # for, rather than surfacing the mistake.
            parser.error("--index does not take a target")
        return _print_index(as_json=args.json)
    if not args.target:
        parser.error("target is required unless --index is given")

    obj = _resolve(args.target)

    # A HelpMixin subclass answers for itself - its own methods give the richest
    # output, and each takes the member name as an optional first argument, so
    # the whole-class and single-member cases are one path. ``members``/``brief``
    # are ignored by the member-specific branches inside ``help``.
    if isinstance(obj, type) and issubclass(obj, HelpMixin):
        # A member that does not resolve is a user error, and every sibling
        # path here exits non-zero for one (``_resolve`` raises SystemExit, a
        # bare ``--signature`` raises SystemExit, a bad member on a
        # non-HelpMixin target raises AttributeError). The mixin's own
        # ``returns=True`` accessors instead hand back "'X' has no member 'y'"
        # as an ordinary *value*, which printed to stdout and returned 0 - so
        # ``--source > file || fallback`` never took the fallback and wrote the
        # error text into the file. Probe with the same ``getattr(..., None)``
        # the mixin uses at each of its own miss sites rather than matching its
        # message text, which would couple the CLI to that wording.
        if args.member is not None and getattr(obj, args.member, None) is None:
            if args.json:
                # The {"error": ...} envelope is a machine contract: it stays
                # on stdout and keeps the mixin's exact shape - only the exit
                # code changes.
                print(obj.help(args.member, members=args.members, as_json=True))
            else:
                # Sourced from the mixin so the two branches can never drift.
                print(obj.help(args.member, returns=True), file=sys.stderr)
            return 1
        if args.signature:
            if not args.member:
                # ASCII only: this reaches stderr, which is cp1252 on a stock
                # Windows console and would raise on a fancier character.
                raise SystemExit(
                    "--signature needs a member, e.g. pythontk.CoreUtils listify"
                )
            print(obj.signature(args.member, returns=True))
        elif args.source:
            print(obj.source(args.member, returns=True))
        elif args.where:
            print(obj.where(args.member, returns=True))
        elif args.json:
            print(obj.help(args.member, members=args.members, as_json=True))
        else:
            print(
                obj.help(
                    args.member, members=args.members, brief=args.brief, returns=True
                )
            )
        return 0

    # Any other object (module, function, non-HelpMixin class).
    if args.member:
        obj = getattr(obj, args.member)
    if args.signature:
        print(HelpMixin.about(obj, signature=True, returns=True))
    elif args.source:
        try:
            print(ApiSurface.source_of(obj))
        except (OSError, TypeError):
            print("no source available")
    elif args.where:
        print(ApiSurface.location(obj) or "unknown location")
    elif args.json:
        print(HelpMixin.about(obj, as_json=True))
    else:
        HelpMixin.about(obj, brief=args.brief)
    return 0


if __name__ == "__main__":
    sys.exit(_main())
