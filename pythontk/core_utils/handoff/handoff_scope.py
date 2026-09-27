# !/usr/bin/python
# coding=utf-8
"""Which objects a hand-off acts on: the scope vocabulary and its precedence.

Every control that offers a Scope choice -- each bridge panel's ``SCOPE`` row,
the preview push, a baker's Scope combo -- speaks the same three words, and every
host resolves them by the same rules: ``selected`` is the default AND the
fallback, and a widening word the host cannot answer falls back to the selection,
never to something wider. The words and the rules live here, once. What a word
MEANS in a scene (a Maya selection list, a Blender view layer) stays with the
host, which passes its reads in as lookups.

Qt-free and DCC-free.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Union

#: A host read: zero arguments, the objects it names, or ``None`` when this host
#: cannot answer (which the precedence treats as "fall back", not as empty).
Lookup = Callable[[], Optional[List[Any]]]


class HandoffScope:
    """The scope vocabulary (``selected`` / ``all`` / ``visible``) and its resolution.

    The words are data a panel offers (:attr:`WORDS`, keyed by :attr:`PARAM`);
    :meth:`resolve` turns one into objects through lookups the host supplies, so
    each DCC contributes only its scene reads and the precedence cannot drift
    between hosts. Called on the class: it holds no state.
    """

    #: The request / panel parameter that carries the scope.
    PARAM = "SCOPE"

    SELECTED = "selected"
    ALL = "all"
    VISIBLE = "visible"

    #: Every word, in offer order. The FIRST is the default and the fallback.
    #: Append-only: a panel's combo persists by INDEX.
    WORDS = (SELECTED, ALL, VISIBLE)

    #: Other spellings a panel's labels use for a word ("Scene" for the whole scene).
    SYNONYMS: Dict[str, str] = {"scene": ALL}

    @classmethod
    def word(cls, scope: Any) -> str:
        """*scope* as one of :attr:`WORDS`.

        Parameters:
            scope: A word, or one of its :attr:`SYNONYMS`. Matched exactly.

        Returns:
            (str) The word. Anything unrecognised (a typo, ``None``, a retired
            value from an old preset) is :attr:`SELECTED`: an unknown scope must
            never silently WIDEN what a hand-off acts on.
        """
        if not isinstance(scope, str):
            return cls.SELECTED
        scope = cls.SYNONYMS.get(scope, scope)
        return scope if scope in cls.WORDS else cls.SELECTED

    @classmethod
    def resolve(
        cls,
        scope: Any,
        *,
        selected: Lookup,
        **widening: Union[Lookup, Sequence[Lookup]],
    ) -> Optional[List[Any]]:
        """The objects *scope* names, through the host's own lookups.

        Parameters:
            scope: A word from :attr:`WORDS` (or a synonym); see :meth:`word`.
            selected: The host's selection read. Called for ``selected``, for an
                unknown scope, and whenever a widening word gets no answer.
            **widening: One entry per widening word (``all=``, ``visible=``): a
                lookup, or a sequence of lookups tried in order. The first to
                return something other than ``None`` wins.

        Returns:
            What the winning lookup returned (a list, possibly empty; ``None``
            only when *selected* itself returns it -- the hand-off skeleton's
            "the host selection").

        Raises:
            TypeError: A *widening* key that is not a widening word -- a typo
                there would otherwise read as "this host cannot answer" and
                quietly narrow the scope to the selection.
        """
        unknown = set(widening) - (set(cls.WORDS) - {cls.SELECTED})
        if unknown:
            raise TypeError(
                f"HandoffScope.resolve: unknown scope lookup(s) {sorted(unknown)}; "
                f"expected any of {[w for w in cls.WORDS if w != cls.SELECTED]}."
            )
        word = cls.word(scope)
        lookups = widening.get(word) if word != cls.SELECTED else None
        if callable(lookups):
            lookups = (lookups,)
        for lookup in lookups or ():
            found = lookup()
            if found is not None:
                return found
        return selected()
