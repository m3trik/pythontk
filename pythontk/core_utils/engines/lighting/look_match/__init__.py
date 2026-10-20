# !/usr/bin/python
# coding=utf-8
"""Look matching: a mix of light layers held to a reference image, region by region.

Built on the lighting kernel (:mod:`..light_layers`, :mod:`..light_predictor`):

    target    -- ``LookTarget`` (the reference and its regions), ``LookRegion`` and
                 ``RegionShape`` (the shape registry a region draws from)
    measures  -- ``LookMeasures`` (the measure registry) and ``LookScore``
    fit       -- ``LookFit``: layer weights fitted to the measures
    sheet     -- ``LookSheet``: comparison sheets a person judges by
    _look_match -- ``LookMatch``, the facade: one session object and its command line
                 (``python -m pythontk.core_utils.engines.lighting.look_match --help``)
    loop      -- ``LookLoop``: a round from a rig by role -- bake (a DCC worker's
                 queue), capture (an RPC client of ``LookCapture.serve``), fit, judge
    session   -- ``LookSession``: the session file every reader opens, declared once

The root registers every one of these names (``ptk.LookMatch`` and friends are the
surface to call), so this package declares none of its own and importing it loads
nothing (CODE_STANDARD section 4: a docstring alone when the root registers everything).
"""
