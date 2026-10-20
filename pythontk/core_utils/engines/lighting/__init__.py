# !/usr/bin/python
# coding=utf-8
"""DCC-agnostic lighting engine: a scene's baked light as data, and matching it to a reference.

A *layer* is one light's contribution to one viewer frame, captured alone (white,
unit weight) in the real viewer. Lightmaps and the reflection probe are linear in
each light, so any lighting state -- a *mix* of weights and colour temperatures --
renders as the tinted sum of its layers, through the viewer's own tone map. That
makes a lighting change instant to see and cheap to fit.

Parts, kernel last:

- :mod:`.look_match` -- matching a mix to a reference image: :class:`LookTarget` /
  :class:`LookRegion` (the reference and the regions compared), :class:`LookMeasures`
  (the measure registry) / :class:`LookScore`, :class:`LookFit`, :class:`LookSheet`,
  and :class:`LookMatch`, the facade and its command line
  (``python -m pythontk.core_utils.engines.lighting.look_match --help``).
- :mod:`.light_predictor` -- :class:`ViewerTransform`, :class:`LightPredictor`: the frame of a mix.
- :mod:`.light_layers` -- :class:`LightLayers` (the store) and :class:`LightMix`.

Refuses: baking, capturing and building lights -- those are host adapters (mayatk's
Arnold layer baker, pythontk's viewer capture); this engine reads what they produce
and writes plain payloads (mixes, rigs) back.
"""

# Lazy-loaded via parent package - no explicit imports needed
