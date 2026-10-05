# !/usr/bin/python
# coding=utf-8
"""The conformance registry: golden-case documents for every ported model.

A model that runs outside Python -- ported to the WebXR runtime's JavaScript
or unitytk's C# -- is only as good as the cases that pin each port to the
Python reference. Every such model has a *provider* whose
``cases(seed=0, **options)`` returns one JSON-able document,
``{"tolerance": {quantity: bound}, "cases": [...], ...}``, generated from the
reference at test time, so a port's test can never hold it to a stale copy.
A port's test asks for it by name::

    doc = ptk.Conformance.cases("articulation", seed=2, per_rig=3)

The same shape for every model, so one runner per runtime serves them all,
and a model gaining a port lands as one entry in :attr:`Conformance.PROVIDERS`.
"""

import importlib
from typing import Any, Dict, List, Tuple


class Conformance:
    """The registry of golden-case providers, by model name."""

    #: Model name -> ``(module, class)`` of its provider, imported on first
    #: use: listing what is pinned never imports a model.
    PROVIDERS: Dict[str, Tuple[str, str]] = {
        # ArticulationModel: pose, world, read, solve and scale.
        "articulation": (
            "pythontk.geo_utils.articulation.conformance",
            "ArticulationConformance",
        ),
        # ShadowProjection.model, ShadowModel.placement and the far point.
        "shadow_projection": (
            "pythontk.geo_utils.shadow_projection",
            "ShadowConformance",
        ),
    }

    @classmethod
    def names(cls) -> List[str]:
        """The models with golden cases, in declaration order."""
        return list(cls.PROVIDERS)

    @classmethod
    def provider(cls, name: str) -> Any:
        """The provider class registered under *name*.

        Raises:
            KeyError: No model is registered under *name*.
        """
        row = cls.PROVIDERS.get(name)
        if row is None:
            raise KeyError(
                f"No conformance cases for {name!r}; registered: "
                f"{', '.join(cls.PROVIDERS) or 'none'}."
            )
        module, attr = row
        return getattr(importlib.import_module(module), attr)

    @classmethod
    def cases(cls, name: str, seed: int = 0, **options: Any) -> Dict[str, Any]:
        """The conformance document of the model *name*.

        Parameters:
            name: A key of :attr:`PROVIDERS`.
            seed: The provider's random stream seed: the same seed, the same
                cases.
            **options: The provider's own sizing (``per_rig``, ``per_kind``).

        Returns:
            ``{"tolerance", "cases", ...}``, plain JSON-able values.
        """
        return cls.provider(name).cases(seed=seed, **options)
