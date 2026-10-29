# !/usr/bin/python
# coding=utf-8
"""How a group of repeated parts should ship: instanced, combined or left alone.

The auto-instancer's decision tree, with the config that steers it and the
strategy it returns. It is scene-free except for one read, the prototype's
triangle count, which the host injects (:data:`TriangleCounter`, the
strategy's ``count_triangles`` adapter). Callers that already know the count
pass it to :meth:`InstancingStrategy.evaluate` directly and need no adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional

__all__ = ["InstancingStrategy", "StrategyConfig", "StrategyType", "TriangleCounter"]

#: The host adapter :class:`InstancingStrategy` is injected with:
#: ``count(mesh_node) -> int``, the triangle count of the host's prototype.
TriangleCounter = Callable[[Any], int]


class StrategyType(Enum):
    """What to do with a group of repeated parts."""

    BAKE = "BAKE"
    COMBINE = "COMBINE"
    GPU_INSTANCE = "GPU_INSTANCE"
    KEEP_SEPARATE = "KEEP_SEPARATE"


@dataclass
class StrategyConfig:
    """The hard constraints :class:`InstancingStrategy` decides under.

    Parameters:
        is_static: The parts never move (dynamic parts are never combined).
        needs_individual: Each part must stay addressable on its own.
        will_be_lightmapped: The parts get lightmaps (a stricter instancing bar).
        can_gpu_instance: The target renderer can GPU-instance.
    """

    is_static: bool = True
    needs_individual: bool = False
    will_be_lightmapped: bool = False
    can_gpu_instance: bool = True


class InstancingStrategy:
    """Determines the best instancing strategy for a group of objects.

    Threshold class attributes may be overridden per-instance to tune the
    decision tree without editing this module.

    Parameters:
        config: The constraints to decide under.
        count_triangles: The host's read of a prototype's triangle count
            (:data:`TriangleCounter`), asked when :meth:`evaluate` gets a
            *mesh_node* and no count; ``None`` reads no scene (0).
    """

    # Below this triangle count a mesh is "micro" — cheaper combined than
    # instanced (per-draw-call overhead dominates).
    MICRO_TRI_THRESHOLD = 300
    # Minimum repeats before instancing pays for itself.
    MIN_INSTANCE_GROUP_SIZE = 10
    # Standard / lightmapped triangle thresholds for GPU instancing.
    INSTANCE_TRI_THRESHOLD = 800
    INSTANCE_TRI_THRESHOLD_LIGHTMAPPED = 1500
    # Heavy meshes are worth instancing even with few repeats.
    HEAVY_TRI_THRESHOLD = 5000
    HEAVY_MIN_GROUP_SIZE = 3

    def __init__(
        self, config: StrategyConfig, count_triangles: Optional[TriangleCounter] = None
    ):
        self.config = config
        #: The host adapter (:data:`TriangleCounter`), or ``None``.
        self.count_triangles = count_triangles

    def evaluate(
        self,
        group_size: int,
        mesh_node: Optional[object] = None,
        triangle_count: Optional[int] = None,
    ) -> StrategyType:
        """Evaluate the strategy for a given group.

        Parameters:
            group_size: Number of items in the group (including prototype).
            mesh_node: The prototype to analyze through
                :meth:`_get_triangle_count` (optional if triangle_count
                provided).
            triangle_count: Explicit triangle count (overrides mesh_node).

        Returns:
            The strategy for the group.
        """
        # 0) Hard constraints
        if self.config.needs_individual:
            return StrategyType.KEEP_SEPARATE

        if not self.config.is_static:
            # Dynamic objects: prefer GPU_INSTANCE if eligible, else KEEP_SEPARATE
            if self.config.can_gpu_instance:
                return StrategyType.GPU_INSTANCE
            return StrategyType.KEEP_SEPARATE

        # Get triangle count
        if triangle_count is None:
            tri_count = self._get_triangle_count(mesh_node) if mesh_node else 0
        else:
            tri_count = triangle_count

        # 1) Micro-geometry gate
        if tri_count < self.MICRO_TRI_THRESHOLD:
            # Repeated micro meshes combine; a lone unique prop stays separate.
            if group_size > 1:
                return StrategyType.COMBINE
            return StrategyType.KEEP_SEPARATE

        # 2) Instancing eligibility gate
        if not self.config.can_gpu_instance:
            return StrategyType.COMBINE  # Static

        # 3) Worth-instancing gate (repeat + triangle thresholds)
        tri_threshold = (
            self.INSTANCE_TRI_THRESHOLD_LIGHTMAPPED
            if self.config.will_be_lightmapped
            else self.INSTANCE_TRI_THRESHOLD
        )
        if tri_count >= tri_threshold and group_size >= self.MIN_INSTANCE_GROUP_SIZE:
            return StrategyType.GPU_INSTANCE

        # Heavy-mesh exception
        if (
            tri_count >= self.HEAVY_TRI_THRESHOLD
            and group_size >= self.HEAVY_MIN_GROUP_SIZE
        ):
            return StrategyType.GPU_INSTANCE

        # 4) Default fallback: static -> COMBINE
        return StrategyType.COMBINE

    def _get_triangle_count(self, mesh_node: object) -> int:
        """The triangle count of ``mesh_node``, the group's prototype.

        Read through the injected :attr:`count_triangles` adapter; 0 when there
        is none, or when the read fails (a node with no mesh must not abort
        the pass).

        Parameters:
            mesh_node: The host's handle on the prototype.

        Returns:
            The triangle count, or 0.
        """
        if self.count_triangles is None:
            return 0
        try:
            return int(self.count_triangles(mesh_node))
        except Exception:
            return 0
