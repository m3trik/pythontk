# !/usr/bin/python
# coding=utf-8
"""The golden case every host's shot store is held to: what it saves.

The store is one class on every host (:class:`~pythontk.ShotStore` over a
``ShotsHost``), and what it saves -- the ``shot_store`` record a scene
carries -- is a contract older scenes hold, so its spelling and bytes never
move with the code.  This is that contract once, as data: one store holding
one of everything :meth:`ShotStore.to_dict` saves (:meth:`ShotStoreConformance.fill`)
and the exact text it is saved as (:meth:`ShotStoreConformance.payload`).
pythontk's tests pin the reference store to it; mayatk's and blendertk's pin
their host's store, saved through the scene's own persistence.  It answers
``cases()`` in the shape of the ported models' golden cases (``Conformance``),
but is not listed there: a saved record is exact text, so it has no tolerance
and no seeded variety for a port's runner to sweep.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional


class ShotStoreConformance:
    """The ``shot_store`` record's golden case: :meth:`fill` and
    :meth:`payload`."""

    #: The objects the case's shots name -- a host test makes a node of the
    #: first to pin the stable id it saves for it (Maya: the node UUID).
    MEMBERS = ("pin_cube", "pin_sphere")

    #: The record :meth:`fill` saves, byte for byte, every key the store
    #: persists in the order it writes them; ``{member_ids}`` stands for the
    #: ids a host saves for the members (:meth:`payload`).
    _PAYLOAD = (
        '{"shots": [{"shot_id": 0, "name": "Intro", "start": 1.0, "end": 48.0, '
        '"objects": ["pin_cube", "pin_sphere"], "metadata": {"step": "A01"}, '
        '"locked": false, "description": "Opening move"}, {"shot_id": 1, '
        '"name": "Outro", "start": 58.0, "end": 100.0, "objects": ["pin_sphere"], '
        '"metadata": {}, "locked": true, "description": ""}], '
        '"hidden_objects": ["pin_sphere"], "pinned_objects": ["pin_cube"], '
        '"markers": [{"time": 12.0, "note": "beat", "data": {"owner": "manifest"}}], '
        '"gap": 10.0, "detection_threshold": 3.5, "detection_mode": "skip_zero", '
        '"initial_shot_length": 120.0, "fit_mode": "fit_contents", '
        '"snap_whole_frames": true, "select_on_load": true, '
        '"frame_on_shot_change": false, "locked_gaps": [[0, 1]], "scene_fps": 24.0, '
        '"source_csv": "shots/plan.csv", '
        '"manifest_edits": {"": {"version": 1, "rows": {}}}, '
        '"manifest_tags": {"": {"A01": "teal"}}, '
        '"manifest_options": {"": {"template": "default", "values": {"skip": 2}}}, '
        '"auto_publish_export": true, "clip_name_strategy": "name", '
        '"drop_stale_shots": true, "edit_ledger": {"version": 2, '
        '"steps": {"crv_tx": [[48.0, "linear", "step", 1.5]]}, '
        '"keys": {"crv_tx": [[58.0, 1, "start"]]}, '
        '"authored": {"crv_ty": [[20.0, 0, "fade_in", "pin_cube", ""]]}}, '
        '"effect_recipe": {"fade_frames": 12.0, "pulse_period": 2.86, '
        '"pulse_duty": 0.59, "pulse_ramp": 0.25, "pulse_lead_in": 0.72, '
        '"pulse_lead_out": 0.72, "pulse_bright": [0.0, 0.088656, 0.723055], '
        '"pulse_dim": [0.0, 0.0, 0.0], "audio_snap": true}, '
        '"member_ids": {member_ids}}'
    )

    @classmethod
    def fill(cls, store: Any) -> Any:
        """Give *store* -- a ``ShotStore`` on any host -- one of everything
        ``to_dict`` saves, as the case does.

        Returns:
            *store*.
        """
        a, b = cls.MEMBERS
        store.scene_fps = 24.0
        s1 = store.define_shot(
            "Intro",
            1,
            48,
            objects=[a, b],
            metadata={"step": "A01"},
            description="Opening move",
        )
        s2 = store.define_shot("Outro", 58, 100, objects=[b], locked=True)
        store.hidden_objects = {b}
        store.pinned_objects = {a}
        store.locked_objects = {a}
        store.markers = [{"time": 12.0, "note": "beat", "data": {"owner": "manifest"}}]
        store.gap = 10.0
        store.detection_threshold = 3.5
        store.detection_mode = "skip_zero"
        store.initial_shot_length = 120.0
        store.fit_mode = "fit_contents"
        store.snap_whole_frames = True
        store.select_on_load = True
        store.frame_on_shot_change = False
        store.locked_gaps = {(s1.shot_id, s2.shot_id)}
        store.source_csv = "shots/plan.csv"
        store.manifest_edits = {"": {"version": 1, "rows": {}}}
        store.manifest_tags = {"": {"A01": "teal"}}
        store.manifest_options = {"": {"template": "default", "values": {"skip": 2}}}
        store.auto_publish_export = True
        store.clip_name_strategy = "name"
        store.drop_stale_shots = True
        store.edit_ledger.record_step("crv_tx", 48.0, "linear", "step", value=1.5)
        store.edit_ledger.record_key("crv_tx", 58.0, owner=s2.shot_id, edge="start")
        store.edit_ledger.record_authored("crv_ty", 20.0, s1.shot_id, "fade_in", a)
        store.effect_recipe = store.effect_recipe.replace(fade_frames=12.0)
        return store

    @classmethod
    def payload(cls, member_ids: Optional[Dict[str, str]] = None) -> str:
        """The exact ``shot_store`` text a :meth:`fill`-ed store saves.

        Parameters:
            member_ids: The stable ids the host saves for the members (Maya:
                ``{"pin_cube": <node UUID>}``); none on a host without them.
        """
        return cls._PAYLOAD.replace("{member_ids}", json.dumps(member_ids or {}))

    @classmethod
    def cases(cls, seed: int = 0) -> Dict[str, Any]:
        """The case document, shaped as the ported models' are: one case,
        exact -- a saved record is text, so the tolerance is none.

        Parameters:
            seed: Unused; the case is fixed (the registry's shape).
        """
        return {
            "tolerance": {},
            "cases": [{"name": "every_saved_field", "payload": cls.payload()}],
        }
