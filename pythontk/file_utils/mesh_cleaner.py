#!/usr/bin/python
# coding=utf-8
"""Mesh repair / cleanup via PyMeshLab (optional dependency).

PyMeshLab is the Python binding for MeshLab — far more mesh-repair
operators than Metashape exposes. Use as a post-export polish step
when Metashape's built-in ``closeHoles`` / ``removeComponents`` aren't
enough.

Install: ``pip install pymeshlab``.

The pipeline applied by :meth:`MeshCleaner.clean` is the canonical
"prepare for downstream DCC" set — duplicate vertex merge, isolated
piece pruning, non-manifold edge repair, optional hole-fill, optional
quadric edge-collapse decimation. Each step is opt-out via parameter.
"""
import logging
import os
from typing import Optional

try:
    import pymeshlab

    PYMESHLAB_AVAILABLE = True
except ImportError:
    pymeshlab = None
    PYMESHLAB_AVAILABLE = False

logger = logging.getLogger(__name__)


class MeshCleaner:
    """Optional-dep mesh-repair pipeline."""

    def is_available(self) -> bool:
        return PYMESHLAB_AVAILABLE

    def clean(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        merge_distance: float = 1e-5,
        remove_isolated_pieces_diameter_percent: float = 5.0,
        fill_holes_max_edge_count: int = 500,
        decimate_target_faces: int = 0,
    ) -> Optional[str]:
        """Repair / clean a mesh in place; return the output path.

        Parameters:
            input_path: Source mesh file (OBJ, PLY, FBX, …).
            output_path: Destination. Defaults to ``<stem>_clean.<ext>``.
            merge_distance: Merge duplicate vertices within this distance.
            remove_isolated_pieces_diameter_percent: Drop islands smaller
                than this % of bbox diagonal. 0 = skip.
            fill_holes_max_edge_count: Fill holes up to this many edges.
                0 = skip.
            decimate_target_faces: Quadric edge-collapse target. 0 = skip
                decimation (preserve source density).
        """
        if not self.is_available():
            logger.error("pymeshlab not installed; cannot clean mesh.")
            return None
        if not os.path.exists(input_path):
            logger.error(f"Mesh not found: {input_path}")
            return None

        if output_path is None:
            stem, ext = os.path.splitext(input_path)
            output_path = f"{stem}_clean{ext}"

        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(input_path)

        ms.meshing_remove_duplicate_vertices()
        ms.meshing_remove_unreferenced_vertices()
        ms.meshing_merge_close_vertices(threshold=pymeshlab.PercentageValue(merge_distance))

        if remove_isolated_pieces_diameter_percent > 0:
            ms.meshing_remove_connected_component_by_diameter(
                mincomponentdiag=pymeshlab.PercentageValue(
                    remove_isolated_pieces_diameter_percent
                ),
            )
        ms.meshing_repair_non_manifold_edges()

        if fill_holes_max_edge_count > 0:
            try:
                ms.meshing_close_holes(maxholesize=fill_holes_max_edge_count)
            except Exception as e:
                logger.warning(f"meshing_close_holes failed: {e}")

        if decimate_target_faces > 0:
            ms.meshing_decimation_quadric_edge_collapse(
                targetfacenum=decimate_target_faces,
                preserveboundary=True,
                preservenormal=True,
                preservetopology=True,
            )

        ms.save_current_mesh(output_path)
        logger.info(f"Mesh cleaned: {input_path} → {output_path}")
        return output_path
