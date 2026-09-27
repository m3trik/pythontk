# !/usr/bin/python
# coding=utf-8
"""Handoff -- the generic kit for driving another app with a scripted payload.

One family, formerly five flat ``core_utils`` modules:

* :mod:`.app_handoff` -- the Template-Method / Strategy bridge: ``HandoffBridge``,
  ``HandoffRequest``, ``Payload``, the ``Deliverer`` strategies (launch, run,
  round trip) and the carrier vocabulary (``CARRIER_*``, ``RIG_MODES``).
* :mod:`.script_template` -- ``ScriptTemplate``: on-disk templates with ``__KEY__``
  slots and their declared modes (``SEND_TO`` / ``SAVE_AS`` / ``ROUND_TRIP``).
* :mod:`.script_run` -- ``ScriptRunner`` / ``ScriptRunResult`` / ``ProgressRelay``:
  run a script, collect its artifact.
* :mod:`.manifest` -- ``HandoffManifest``: the ``<payload>.manifest.json`` sidecar.
* :mod:`.manifest_plan` -- ``ManifestPlan``: ordered, gated replay of its sections.
* :mod:`.handoff_scope` -- ``HandoffScope``: the Scope words (``selected`` / ``all`` /
  ``visible``) and their precedence, over lookups each host supplies.

Generic by design (pythontk is app-agnostic): the app-specific halves live with
their consumers. The classes and the mode/carrier vocabulary are registered at
the pythontk root -- another package imports ``from pythontk import
HandoffBridge, ScriptTemplate, SEND_TO``, never these module paths.
"""
