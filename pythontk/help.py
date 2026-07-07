# !/usr/bin/python
# coding=utf-8
"""Command-line entry to HelpMixin introspection.

    python -m pythontk.help <dotted.path> [member] [options]

Resolve a class or object by dotted path and print its help, source, location,
or signature - the same introspection ``HelpMixin`` exposes in-process, made
reachable from a shell or an agent without writing a REPL snippet.

Everything in this module is private on purpose: it is a CLI shell, not public
API, so the registry walker (which skips modules with no public surface) leaves
it out.

Examples:
    python -m pythontk.help pythontk.CoreUtils
    python -m pythontk.help pythontk.CoreUtils listify --json
    python -m pythontk.help pythontk.FileUtils get_file_contents --source
    python -m pythontk.help pythontk.CoreUtils --members methods --brief
"""
import importlib
import inspect
import sys

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
        try:
            obj = importlib.import_module(".".join(parts[:i]))
        except ImportError:
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


def _main(argv=None) -> int:
    parser = CLI.get_parser("Introspect a pythontk/ecosystem class or object.")
    parser.add_argument("target", help="dotted path, e.g. pythontk.CoreUtils")
    parser.add_argument("member", nargs="?", help="member name on the target class")
    parser.add_argument("--json", action="store_true", help="structured JSON output")
    parser.add_argument("--source", action="store_true", help="print source code")
    parser.add_argument("--where", action="store_true", help="print file:line location")
    parser.add_argument("--signature", action="store_true", help="print signature detail")
    parser.add_argument("--brief", action="store_true", help="one-line summaries")
    parser.add_argument(
        "--members",
        help="filter: methods|properties|classmethods|staticmethods",
    )
    args = parser.parse_args(argv)

    obj = _resolve(args.target)
    is_help_cls = isinstance(obj, type) and issubclass(obj, HelpMixin)

    # Class + member: use the class's own HelpMixin methods (richest output).
    if is_help_cls and args.member:
        if args.source:
            print(obj.source(args.member, returns=True))
        elif args.where:
            print(obj.where(args.member, returns=True))
        elif args.signature:
            print(obj.signature(args.member, returns=True))
        elif args.json:
            print(obj.help(args.member, as_json=True))
        else:
            print(obj.help(args.member, brief=args.brief, returns=True))
        return 0

    # Class alone.
    if is_help_cls:
        if args.source:
            print(obj.source(returns=True))
        elif args.where:
            print(obj.where(returns=True))
        elif args.json:
            print(obj.help(members=args.members, brief=args.brief, as_json=True))
        else:
            print(obj.help(members=args.members, brief=args.brief, returns=True))
        return 0

    # Any other object (module, function, non-HelpMixin class).
    if args.member:
        obj = getattr(obj, args.member)
    if args.source:
        target = inspect.unwrap(obj) if callable(obj) else obj
        try:
            print(inspect.getsource(target))
        except (OSError, TypeError):
            print("no source available")
    elif args.where:
        try:
            target = inspect.unwrap(obj) if callable(obj) else obj
            src_file = inspect.getsourcefile(target)
            _, line = inspect.getsourcelines(target)
            print(f"{src_file}:{line}" if src_file else "built-in")
        except (OSError, TypeError):
            print("unknown location")
    elif args.json:
        print(HelpMixin.about(obj, as_json=True))
    else:
        HelpMixin.about(obj, brief=args.brief)
    return 0


if __name__ == "__main__":
    sys.exit(_main())
