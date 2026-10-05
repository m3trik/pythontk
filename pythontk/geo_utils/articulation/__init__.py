# !/usr/bin/python
# coding=utf-8
"""Articulated rigs: rigid links on hinge, swivel, ball and slide joints.

The root registers every name (``from pythontk import ArticulationModel``);
importing this package loads none of its modules.

- :mod:`~pythontk.geo_utils.articulation.model` -- :class:`ArticulationModel`:
  the joint model a rig ships (rest frames, per-joint channels, limits) and the
  grab solver that poses it. The single source of truth the runtimes port:
  unitytk's ``ArticulatedRigController.cs``, the WebXR preview's
  ``articulated_rig.js`` and the mayatk grab tool.
- :mod:`~pythontk.geo_utils.articulation.analysis` --
  :class:`ArticulationAnalysis`: which parts move together, where they pivot
  and how, proposed from the parts' geometry alone.
- :mod:`~pythontk.geo_utils.articulation.record` -- :class:`ArticulationRecord`
  and :class:`ArticulationWeb`: the ``articulation`` record's payload and the
  glTF manifest bound from it, declared once (the ports' types are generated
  from them).
- :mod:`~pythontk.geo_utils.articulation.conformance` --
  :class:`ArticulationConformance`: the golden cases every port of the model
  is held to (``Conformance.cases("articulation")``).
"""
