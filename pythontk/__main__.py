# !/usr/bin/python
# coding=utf-8
"""Command-line entry to HelpMixin introspection.

    python -m pythontk <dotted.path> [member] [options]

Resolve a class or object by dotted path and print its help, source, location,
or signature - the same introspection ``HelpMixin`` exposes in-process, made
reachable from a shell or an agent without writing a REPL snippet. Unlike the
static API registry it reads the *live* object, so ``--source``/``--where``
report what is actually loaded.

Everything in this module is private on purpose: it is a CLI shell, not public
API. ``__main__.py`` is also skipped outright by the registry walker, so the
public surface is unaffected.

Examples:
    python -m pythontk pythontk.CoreUtils
    python -m pythontk pythontk.CoreUtils listify --json
    python -m pythontk pythontk.FileUtils get_file_contents --source
    python -m pythontk pythontk.CoreUtils --members methods --brief
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

    # A HelpMixin subclass answers for itself - its own methods give the richest
    # output, and each takes the member name as an optional first argument, so
    # the whole-class and single-member cases are one path. ``members``/``brief``
    # are ignored by the member-specific branches inside ``help``.
    if isinstance(obj, type) and issubclass(obj, HelpMixin):
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
    name = args.member or getattr(obj, "__name__", str(obj))
    if args.member:
        obj = getattr(obj, args.member)
    target = inspect.unwrap(obj) if callable(obj) else obj
    if args.signature:
        # Reuses HelpMixin's renderer rather than reimplementing it here.
        print(HelpMixin._signature_detail(name, obj))
    elif args.source:
        try:
            print(inspect.getsource(target))
        except (OSError, TypeError):
            print("no source available")
    elif args.where:
        try:
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
