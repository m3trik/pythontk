# !/usr/bin/python
# coding=utf-8
"""A pip requirement -- a distribution name plus a version range -- read and
compared without ``packaging``, which is not guaranteed inside a DCC's
interpreter (mayapy, Blender's bundled python)."""

import re
from typing import Optional, Tuple


class _PackageRequirementInternal:
    """Parsing and PEP 440 ordering for :class:`PackageRequirement`."""

    #: PEP 508, minus direct references: a name, ``[extras]``, specifiers
    #: (optionally parenthesized, as older metadata writes them), ``; marker``.
    _REQUIREMENT = re.compile(
        r"""^\s*(?P<name>[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?)\s*
        (?:\[(?P<extras>[^\]]*)\])?\s*
        (?P<specifiers>[^;]*?)\s*
        (?:;\s*(?P<marker>.*?)\s*)?$""",
        re.VERBOSE,
    )
    _SPECIFIER = re.compile(r"^(?P<op>~=|==|!=|<=|>=|<|>)\s*(?P<version>\S+)$")
    #: A PEP 440 public version; a ``+local`` label is accepted and never orders.
    _VERSION = re.compile(
        r"""^\s*v?
        (?:(?P<epoch>\d+)!)?
        (?P<release>\d+(?:\.\d+)*)
        (?:[-_.]?(?P<pre_l>alpha|beta|preview|pre|rc|a|b|c)[-_.]?(?P<pre_n>\d*))?
        (?:-(?P<post_implicit>\d+)|[-_.]?(?:post|rev|r)[-_.]?(?P<post_n>\d*))?
        (?:[-_.]?dev[-_.]?(?P<dev_n>\d*))?
        (?P<local>\+[a-z0-9]+(?:[-_.][a-z0-9]+)*)?
        \s*$""",
        re.VERBOSE | re.IGNORECASE,
    )
    _LEADING_RELEASE = re.compile(r"^\s*v?(?:\d+!)?(\d+(?:\.\d+)*)")
    _PRE_RANK = {"a": 0, "alpha": 0, "b": 1, "beta": 1}  # every other label is rc
    _EXTRA_CLAUSE = re.compile(r"""\bextra\s*==\s*["']([^"']+)["']""")

    @staticmethod
    def _normalize(name: str) -> str:
        """PEP 503 name comparison: case- and separator-insensitive."""
        return re.sub(r"[-_.]+", "-", str(name)).lower()

    @classmethod
    def _parse_version(cls, text) -> tuple:
        """``(epoch, release, pre, post, dev)`` of *text*; ValueError if it is none.

        *pre* is ``(rank, n)`` with a < b < rc, *post* / *dev* an int; each None
        when absent. The local label is dropped.
        """
        match = cls._VERSION.match(str(text)) if text is not None else None
        if not match:
            raise ValueError(f"Not a PEP 440 version: {text!r}")
        pre = None
        if match.group("pre_l") is not None:
            rank = cls._PRE_RANK.get(match.group("pre_l").lower(), 2)
            pre = (rank, int(match.group("pre_n") or 0))
        post = match.group("post_implicit")
        if post is None and match.group("post_n") is not None:
            post = match.group("post_n") or 0
        dev = match.group("dev_n")
        return (
            int(match.group("epoch") or 0),
            tuple(int(p) for p in match.group("release").split(".")),
            pre,
            None if post is None else int(post),
            None if dev is None else int(dev or 0),
        )

    @staticmethod
    def _trimmed(release: tuple) -> tuple:
        """*release* without trailing zeros, so ``1.0`` orders equal to ``1.0.0``."""
        parts = list(release)
        while parts and parts[-1] == 0:
            parts.pop()
        return tuple(parts)

    @classmethod
    def _key(cls, parsed: tuple) -> tuple:
        """PEP 440 sort key: dev < pre (a < b < rc) < release < post."""
        epoch, release, pre, post, dev = parsed
        if pre is None and post is None and dev is not None:
            pre_key = (-1, 0)  # 1.0.dev0 sorts before 1.0a0
        else:
            pre_key = (3, 0) if pre is None else pre
        return (
            epoch,
            cls._trimmed(release),
            pre_key,
            -1 if post is None else post,
            (1, 0) if dev is None else (0, dev),
        )

    @classmethod
    def _matches(cls, op: str, wanted: tuple, have: tuple) -> bool:
        """Whether parsed version *have* satisfies ``<op> <wanted>``.

        The exclusive bounds follow PEP 440 rather than plain ordering: ``<V``
        admits no pre-release of V's own release unless V is one, and ``>V`` no
        post-release of it unless V is one.
        """
        have_key, wanted_key = cls._key(have), cls._key(wanted)
        same_release = (have[0], cls._trimmed(have[1])) == (
            wanted[0],
            cls._trimmed(wanted[1]),
        )
        if op == "==":
            return have_key == wanted_key
        if op == "!=":
            return have_key != wanted_key
        if op == ">=":
            return have_key >= wanted_key
        if op == "<=":
            return have_key <= wanted_key
        if op == "<":
            wanted_pre = wanted[2] is not None or wanted[4] is not None
            have_pre = have[2] is not None or have[4] is not None
            if not wanted_pre and have_pre and same_release:
                return False
            return have_key < wanted_key
        if op == ">":
            if wanted[3] is None and have[3] is not None and same_release:
                return False
            return have_key > wanted_key
        # ~=: at or above it, inside the series its last release segment counts in.
        prefix = wanted[1][:-1]
        padded = tuple(have[1]) + (0,) * len(prefix)
        return have_key >= wanted_key and padded[: len(prefix)] == prefix

    @classmethod
    def _parse(cls, spec: str) -> tuple:
        """``(name, extras, specifiers, marker)`` of *spec*; ValueError if unread."""
        match = cls._REQUIREMENT.match(str(spec or ""))
        if not match:
            raise ValueError(f"Not a requirement this reader supports: {spec!r}")
        text = match.group("specifiers").strip()
        if text.startswith("(") and text.endswith(")"):
            text = text[1:-1].strip()
        specifiers = []
        for clause in filter(None, (c.strip() for c in text.split(","))):
            part = cls._SPECIFIER.match(clause)
            if not part:
                raise ValueError(f"Unsupported specifier {clause!r} in {spec!r}")
            op, version = part.group("op"), part.group("version")
            parsed = cls._parse_version(version)  # raises on a wildcard or ===
            if "+" in version:
                raise ValueError(f"A local version label in {spec!r} is unsupported")
            if op == "~=" and len(parsed[1]) < 2:
                raise ValueError(f"~= needs two release segments: {spec!r}")
            specifiers.append((op, version))
        if text and not specifiers:
            raise ValueError(f"Not a requirement this reader supports: {spec!r}")
        extras = tuple(
            e.strip() for e in (match.group("extras") or "").split(",") if e.strip()
        )
        return (
            match.group("name"),
            extras,
            tuple(specifiers),
            match.group("marker") or "",
        )


class PackageRequirement(_PackageRequirementInternal):
    """A pip requirement -- a distribution name plus a version range.

    ``PackageRequirement("extapps>=0.8.0,<0.9")`` answers whether an installed
    version satisfies the range (:meth:`contains`), the question pip asks of an
    installed distribution before it decides to install anything. ``str()`` is
    the text the caller wrote, which is what pip is handed.

    Reads a PEP 508 / PEP 440 subset:

    * a name, optional ``[extras]``, comma-joined ``>= > <= < == != ~=``
      specifiers (bare or parenthesized, as older metadata writes them), and an
      optional ``; marker``;
    * versions with an epoch, a release, a pre-release (``a`` / ``b`` / ``rc``
      and their aliases), a post-release and a dev release, ordered as PEP 440
      orders them (``0.9.0.dev1 < 0.9.0rc1 < 0.9.0 < 0.9.0.post1``); a ``+local``
      label is accepted on a version and ignored.

    Not supported, raising ValueError at construction: a direct reference
    (``name @ url``, a VCS or file URL), ``===``, the ``==1.0.*`` / ``!=1.0.*``
    wildcards, and a local label inside a specifier. Not done: a marker is kept
    as text and never evaluated, and pip's default exclusion of pre-releases is
    not applied -- a pre-release inside the range satisfies it, as pip treats
    one already installed.

    Attributes:
        name: The distribution name, as written.
        extras: The requested extras, as written.
        specifiers: ``((op, version), ...)`` in the order written; empty for a
            bare name, which every version satisfies.
        marker: The environment marker's text, ``""`` when there is none.
    """

    def __init__(self, spec: str):
        """
        Parameters:
            spec: The requirement, e.g. ``"extapps>=0.8.0,<0.9"``.

        Raises:
            ValueError: *spec* is not in the subset this class reads.
        """
        self.name, self.extras, self.specifiers, self.marker = self._parse(spec)
        self._spec = str(spec).strip()

    def __str__(self) -> str:
        return self._spec

    def __repr__(self) -> str:
        return f"PackageRequirement({self._spec!r})"

    def contains(self, version: Optional[str]) -> bool:
        """Whether *version* satisfies every specifier.

        Parameters:
            version: A version string (an installed ``__version__`` or metadata
                version). None or an unreadable string satisfies no specifier:
                a range the caller asked for has to be provable.

        Returns:
            True for any *version* when there are no specifiers.
        """
        if not self.specifiers:
            return True
        try:
            have = self._parse_version(version)
        except ValueError:
            return False
        return all(
            self._matches(op, self._parse_version(wanted), have)
            for op, wanted in self.specifiers
        )

    @classmethod
    def version_key(cls, version: str) -> tuple:
        """A sort key ordering versions as PEP 440 does.

        Parameters:
            version: e.g. ``"0.9.0rc1"``.

        Returns:
            A tuple: ``version_key("0.9.0rc1") < version_key("0.9.0")``, and
            ``version_key("1.0") == version_key("1.0.0")``.

        Raises:
            ValueError: *version* is not a PEP 440 version.
        """
        return cls._key(cls._parse_version(version))

    @classmethod
    def release(cls, version: Optional[str]) -> Tuple[int, ...]:
        """The leading numeric release segments of *version*; never raises.

        ``"1.2.0rc1"`` -> ``(1, 2, 0)``; ``"1.2.x"`` -> ``(1, 2)``; ``""`` or
        None -> ``()``. A coarse key for a floor check that can tolerate any
        string; :meth:`version_key` is the exact one.
        """
        match = cls._LEADING_RELEASE.match(str(version or ""))
        return tuple(int(p) for p in match.group(1).split(".")) if match else ()

    @classmethod
    def declared_by(
        cls, dist: str, name: str, extra: Optional[str] = None
    ) -> Optional["PackageRequirement"]:
        """The requirement installed distribution *dist* declares on *name*.

        Read from *dist*'s own metadata (``importlib.metadata.requires``), so a
        caller states a range once, in its pyproject, and the code reading it
        cannot drift from it.

        Parameters:
            dist: The declaring distribution, e.g. ``"tentacletk"``.
            name: The required distribution, e.g. ``"extapps"``.
            extra: Read only a row of this extra (``extra == "apps"``). None
                reads only an unconditional row.

        Returns:
            The row without its marker -- the marker selected the row, and pip
            evaluates one in its own environment, where no extra is active, so
            the row as written would install nothing. None when *dist* is not
            installed (a source checkout has no metadata) or declares no such row.
        """
        try:
            from importlib.metadata import requires

            rows = requires(dist) or ()
        except Exception:  # noqa: BLE001 -- not installed, or unreadable metadata
            return None
        wanted = cls._normalize(name)
        wanted_extra = None if extra is None else cls._normalize(extra)
        for row in rows:
            try:
                req = cls(row)
            except ValueError:
                continue
            if cls._normalize(req.name) != wanted:
                continue
            extras = {cls._normalize(e) for e in cls._EXTRA_CLAUSE.findall(req.marker)}
            if (wanted_extra is None and not extras) or wanted_extra in extras:
                return cls(str(row).partition(";")[0])
        return None
