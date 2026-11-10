# !/usr/bin/python
# coding=utf-8
"""Engineering CAD (STEP, IGES) to meshes and an assembled GLB, with a record of the product's truth.

The root registers every public name here (``ptk.CadImport``, ``ptk.CadRecord``),
so this package declares none of its own and importing it loads nothing:

    _cad_import -- ``CadImport``, the facade: profiles, the conversion, the
                   deviation check, the files it writes
    _occ        -- every OpenCascade call (the optional ``pythontk[cad]`` extra),
                   imported only when a conversion runs
    cad_record  -- ``CadRecord``, the record a conversion writes, declared once
"""
