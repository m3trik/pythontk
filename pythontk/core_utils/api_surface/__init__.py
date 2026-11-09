# !/usr/bin/python
# coding=utf-8
"""API surface -- a package's public API, read one way for every reader.

One engine behind every view of the public API: ``HelpMixin``'s ``.help()``
and ``python -m pythontk``, the ``api.search`` / ``api.describe`` operations
agents call, m3trik's release checks and the runtime-vs-static drift gate.

* :mod:`.api_surface` -- ``ApiSurface``, the facade: the static read, search
  and describe, the live member reading, and the one docstring summary rule.
* :mod:`.package_surface` -- ``PackageSurface``, ``ModuleSurface``,
  ``ClassSurface``: the static shape, kept in the per-user cache.
* :mod:`._source_reader` -- the static walk itself (``ast``, nothing imported).
* :mod:`._csharp_reader` -- the C# a package declares, read the same way (no
  editor, nothing compiled), as the same shape.

The root registers every public name (``from pythontk import ApiSurface``);
this package declares none of its own (CODE_STANDARD sections 3 and 4).
"""
