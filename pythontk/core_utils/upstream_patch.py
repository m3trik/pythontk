# !/usr/bin/python
# coding=utf-8
"""UpstreamPatch - correcting a defect in code we do not own, with a way out.

The mirror of :mod:`deprecation` for the other direction. That module retires
*our* public surface and its point is ``remove_in``: the version a deprecation
stops working, recorded so something can FAIL when the date passes. This one
patches *someone else's* code, and its point is :meth:`UpstreamPatch.detects`:
the probe that answers "is the defect still there?", recorded so something can
fail when upstream fixes it.

Without that, a monkey patch is immortal. It keeps overriding a function that
has been correct for three releases, nobody can tell whether removing it is safe,
and the reasoning that justified it lives in a comment next to the call site --
or nowhere. Every such patch this ecosystem has needed was found by reading the
upstream source, and the finding is worth more than the patch: it belongs beside
it, not in a commit message.

So a patch declares four things, and the machinery is the rest:

    * **What it replaces** -- ``"module.path:Class.attribute"``, resolved when the
      patch is applied rather than when it is declared, so a package the host
      does not ship is simply a no-op instead of an import error.
    * **Why** -- the defect, in the terms of the upstream code, with a tracker
      link where one exists.
    * **The replacement** -- through :meth:`replaces`, which hands it the
      original as its first argument. No global holding "the real one".
    * **The probe** -- through :meth:`detects`, which reproduces the defect
      against the STOCK code and returns whether it is still present.

The probe is what makes the patch removable. A test sweeps
:meth:`UpstreamPatch.registry` and asserts every patch is still needed; the day
upstream lands the fix that test fails, names the patch, and the patch is
deleted -- rather than being carried for years because nobody could prove it was
safe to drop.

Scope: one attribute, swapped for the duration of a ``with`` block. Nothing here
installs anything permanently, because a process-wide patch applied at import
time is how a workaround becomes a haunting: it changes behaviour for callers
that never asked, and the traceback it produces names a module whose source does
not contain the frame.
"""

import contextlib
import functools
import importlib
import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple

__all__ = ["UpstreamPatch"]

#: Sentinel for "the attribute was not there before we set it".
_MISSING = object()


class UpstreamPatch:
    """One third-party defect, its replacement, and the probe that retires it.

    Parameters:
        name: A short identifier, unique across the process (``"io_scene_fbx:
            sibling armatures"``). Used in errors and in the sweep test's output.
        target: ``"module.path:Attribute"`` or ``"module.path:Class.attribute"``.
            Resolved on use, never at declaration.
        reason: What upstream does wrong, in terms of upstream's own code. This
            is the finding; write it for whoever has to decide, a year from now,
            whether the patch is still earning its place.
        tracker: The upstream issue/PR, when one exists.

    Example::

        SIBLING_ARMATURES = UpstreamPatch(
            name="io_scene_fbx: sibling armatures skipped",
            target="io_scene_fbx.import_fbx:FbxImportHelperNode.collect_armature_meshes",
            reason="walks self.children while the recursion re-parents out of it",
        )

        @SIBLING_ARMATURES.replaces
        def _collect(original, self):
            if self.is_armature:
                return original(self)
            for child in tuple(self.children):
                child.collect_armature_meshes()

        @SIBLING_ARMATURES.detects
        def _still_broken():
            ...  # build the failing case against the stock code
            return bound != expected

        with SIBLING_ARMATURES.applied():
            ...
    """

    _registry: Dict[str, "UpstreamPatch"] = {}

    def __init__(
        self,
        name: str,
        target: str,
        *,
        reason: str,
        tracker: Optional[str] = None,
    ) -> None:
        if ":" not in target:
            raise ValueError(
                f"{name}: target must be 'module.path:Attribute' (got {target!r})"
            )
        if not reason:
            # Same rule as Deprecation's mandatory replacement: a patch whose
            # justification is not written down cannot be reviewed, and reviewing
            # it is the only way it ever gets removed.
            raise ValueError(f"{name}: a patch must say what it corrects")
        existing = UpstreamPatch._registry.get(name)
        if existing is not None and existing.target != target:
            raise ValueError(
                f"Duplicate UpstreamPatch name {name!r}: already declared against "
                f"{existing.target!r}"
            )
        # Same name AND same target is a module RELOAD re-running its declarations
        # (``ptk.ModuleReloader``, and the DCC test harnesses that purge a package
        # between modules). Replacing the record keeps the newest replacement and
        # probe, where raising would turn a reload into an ImportError.
        self.name = name
        self.target = target
        self.reason = reason
        self.tracker = tracker
        self._replacement: Optional[Callable] = None
        self._probe: Optional[Callable[[], bool]] = None
        UpstreamPatch._registry[name] = self

    # ------------------------------------------------------------------ declaring
    def replaces(self, func: Callable) -> Callable:
        """Register *func* as the replacement; returns it unchanged.

        It is called with the ORIGINAL attribute as its first argument, so a
        replacement that defers to upstream for part of the work says so in its
        signature instead of closing over a module global.
        """
        if self._replacement is not None:
            raise ValueError(f"{self.name}: replacement already declared")
        self._replacement = func
        return func

    def detects(self, func: Callable[[], bool]) -> Callable[[], bool]:
        """Register *func* as the probe; returns it unchanged.

        It must reproduce the defect against the STOCK attribute and return
        whether it is still present. Run it OUTSIDE :meth:`applied` -- proving the
        bug against our own replacement proves nothing.
        """
        if self._probe is not None:
            raise ValueError(f"{self.name}: probe already declared")
        self._probe = func
        return func

    # ------------------------------------------------------------------ resolving
    def _resolve(self) -> Tuple[Any, str]:
        """``(owner, attribute_name)`` for :attr:`target`, or ``(None, "")``.

        ``(None, "")`` means the host does not ship the module (or renamed the
        owner out from under us): a patch for code that is not here is a no-op,
        not an error.
        """
        module_path, _, attribute_path = self.target.partition(":")
        try:
            owner: Any = importlib.import_module(module_path)
        except ImportError:
            return None, ""
        parts = attribute_path.split(".")
        for part in parts[:-1]:
            owner = getattr(owner, part, None)
            if owner is None:
                return None, ""
        return (owner, parts[-1]) if hasattr(owner, parts[-1]) else (None, "")

    @property
    def available(self) -> bool:
        """Whether :attr:`target` resolves in this host."""
        return self._resolve()[0] is not None

    # ------------------------------------------------------------------ applying
    @contextlib.contextmanager
    def applied(self):
        """Swap the replacement in for the duration of the block.

        A no-op when the target does not resolve or no replacement was declared,
        so a caller never has to ask whether the host ships the module. Restores
        on the way out however the block ends, and nests safely: an inner block
        restores what it found, which is the outer block's replacement.

        The attribute is handled as STORED, not as ``getattr`` hands it back: a
        staticmethod or classmethod target is swapped for one of its own kind
        (a classmethod's original arrives bound to the class it was called
        on), and restored as the very object that was there -- or, when the
        owner only inherited it, removed again, so inheritance resumes.
        Restoring a ``getattr`` result wrote the unwrapped function into the
        class, and a staticmethod target stayed a plain function for good.

        Yields:
            True while the replacement is in place; False when the block runs
            against the stock attribute (no target here, or no replacement).
        """
        owner, attribute = self._resolve()
        if owner is None or self._replacement is None:
            yield False
            return
        stored = getattr(owner, "__dict__", {}).get(attribute, _MISSING)
        raw = (
            stored
            if stored is not _MISSING
            else inspect.getattr_static(owner, attribute)
        )
        replacement = self._replacement
        if isinstance(raw, classmethod):

            @functools.wraps(raw.__func__)
            def patched(cls, *args, **kwargs):
                return replacement(raw.__get__(None, cls), *args, **kwargs)

            swapped: Any = classmethod(patched)
        else:
            original = raw.__func__ if isinstance(raw, staticmethod) else raw

            @functools.wraps(original)
            def patched(*args, **kwargs):
                return replacement(original, *args, **kwargs)

            swapped = staticmethod(patched) if isinstance(raw, staticmethod) else patched

        setattr(owner, attribute, swapped)
        try:
            yield True
        finally:
            if stored is _MISSING:
                delattr(owner, attribute)
            else:
                setattr(owner, attribute, stored)

    # ------------------------------------------------------------------ retiring
    def still_needed(self) -> bool:
        """Whether the defect is still present in this host.

        Raises when no probe was declared: a patch nobody can retire is the
        failure mode this class exists to prevent, so it fails loudly at the one
        moment someone is asking.

        Returns:
            What the probe answered: True while the defect reproduces.

        Raises:
            RuntimeError: No probe was declared, or the target does not resolve
                here, so there is nothing to ask.
        """
        if self._probe is None:
            raise RuntimeError(
                f"{self.name}: no probe declared, so nothing can tell whether "
                "upstream has fixed this. Add one with @<patch>.detects."
            )
        if not self.available:
            raise RuntimeError(
                f"{self.name}: target {self.target!r} does not resolve here, so "
                "the probe cannot say anything about it."
            )
        return bool(self._probe())

    # ------------------------------------------------------------------ reporting
    @classmethod
    def registry(cls) -> List["UpstreamPatch"]:
        """Every patch declared in this process, in declaration order.

        What the sweep test walks. Import the modules that declare them first --
        a declaration is a module-level statement, so nothing is registered until
        its module is imported.
        """
        return list(cls._registry.values())

    def __repr__(self) -> str:
        return f"<UpstreamPatch {self.name!r} -> {self.target!r}>"
