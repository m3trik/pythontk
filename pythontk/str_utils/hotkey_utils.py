# !/usr/bin/python
# coding=utf-8
"""Portable hotkey-token helpers shared by the ecosystem's macro managers.

Both mayatk's and blendertk's ``edit_utils.macros.MacroManager`` speak the same
on-disk hotkey-token convention (``"ctl+sht+i"``, modifier-order-canonical,
single-character keys lowercased) and the same Qt-key-sequence <-> token
round-trip, plus a shared snake_case-name-to-Title-Case humanizer for macro
labels. None of this touches a DCC API (``cmds``/``bpy``) — it is pure
string/dict manipulation — so it belongs here rather than being hand-maintained
twice. Each package's ``MacroManager`` still owns its own DCC-specific pieces
(Maya ``runTimeCommand``/hotkey registration vs. Blender keymap items) and its
own class-hierarchy introspection (macro discovery/category/conflict methods
that walk ``cls.__mro__``); only the DCC-agnostic conversion logic lives here.
"""
from typing import Dict, Optional, Tuple


class HotkeyUtils:
    """Maya-style hotkey-token <-> Qt-key-sequence conversion + label humanizing."""

    #: Canonical modifier order so equivalent chords compare equal regardless
    #: of how they were typed/parsed.
    MOD_ORDER: Tuple[str, str, str] = ("ctl", "alt", "sht")

    #: Qt ``QKeySequence`` modifier name -> this convention's token.
    QT_MOD_MAP: Dict[str, str] = {
        "ctrl": "ctl",
        "control": "ctl",
        "alt": "alt",
        "shift": "sht",
        "meta": "ctl",
        "cmd": "ctl",
    }

    @classmethod
    def parse_key(cls, key: str) -> Tuple[bool, bool, bool, str]:
        """Split a hotkey token into ``(ctl, alt, sht, key)``.

        Modifiers (``ctl``/``alt``/``sht``) are matched case-insensitively; the
        remaining token is the key itself (e.g. ``"i"``, ``"F3"``).
        """
        ctl = alt = sht = False
        k = str(key)
        for char in str(key).split("+"):
            token = char.strip()
            low = token.lower()
            if low == "ctl":
                ctl = True
            elif low == "alt":
                alt = True
            elif low == "sht":
                sht = True
            else:
                k = token
        return ctl, alt, sht, k

    @classmethod
    def qt_sequence_to_key(cls, sequence: str) -> str:
        """Convert a Qt key-sequence string (``"Ctrl+Shift+I"``) to this
        convention's token (``"ctl+sht+i"``).

        Returns ``""`` when *sequence* carries no non-modifier key.
        """
        if not sequence:
            return ""
        present = {"ctl": False, "alt": False, "sht": False}
        key = None
        for part in str(sequence).split("+"):
            token = part.strip()
            mod = cls.QT_MOD_MAP.get(token.lower())
            if mod:
                present[mod] = True
            elif token:
                key = token
        if key is None:
            return ""
        if len(key) == 1:
            key = key.lower()
        mods = [m for m in cls.MOD_ORDER if present[m]]
        return "+".join(mods + [key])

    @classmethod
    def key_to_qt_sequence(cls, key: str) -> str:
        """Convert this convention's token (``"ctl+sht+i"``) to a Qt
        key-sequence string for display (``"Ctrl+Shift+I"``)."""
        if not key:
            return ""
        ctl, alt, sht, k = cls.parse_key(key)
        seq = []
        if ctl:
            seq.append("Ctrl")
        if alt:
            seq.append("Alt")
        if sht:
            seq.append("Shift")
        seq.append(k.upper() if len(k) == 1 else k)
        return "+".join(seq)

    @staticmethod
    def humanize_label(name: str, prefix: str = "", acronyms: Optional[Dict[str, str]] = None) -> str:
        """Humanize a ``snake_case`` name for display, e.g. ``back_face_culling`` ->
        "Back Face Culling". *acronyms* maps a lowercased word to its preferred
        display casing (e.g. ``{"uv": "UV"}``); a word already upper-case in the
        source (len > 1) is preserved as-is regardless of *acronyms*."""
        base = name[len(prefix):] if prefix and name.startswith(prefix) else name
        acronyms = acronyms or {}
        words = []
        for word in base.split("_"):
            if not word:
                continue
            low = word.lower()
            if low in acronyms:
                words.append(acronyms[low])
            elif word.isupper() and len(word) > 1:
                words.append(word)  # already an acronym in the source name
            else:
                words.append(word.capitalize())
        return " ".join(words)


__all__ = ["HotkeyUtils"]
