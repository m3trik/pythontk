# !/usr/bin/python
# coding=utf-8
"""Moved to :mod:`pythontk.core_utils.engines.scene_export`, split by concept into
``scene_records`` (the declarations), ``scene_store`` (the store contract),
``export_snapshot`` (export assembly) and ``record_transfer`` (crossings) (2026-09-26).

A one-release stub for code importing the old path; import the names from the
pythontk root instead (``from pythontk import ...``).
"""

from pythontk.core_utils.deprecation import Deprecation

Deprecation.attributes(
    globals(),
    {
        "Scope": "pythontk.core_utils.engines.scene_export.scene_records.Scope",
        "Kind": "pythontk.core_utils.engines.scene_export.scene_records.Kind",
        "Merge": "pythontk.core_utils.engines.scene_export.scene_records.Merge",
        "Record": "pythontk.core_utils.engines.scene_export.scene_records.Record",
        "RecordSpec": "pythontk.core_utils.engines.scene_export.scene_records.RecordSpec",
        "SceneRecords": "pythontk.core_utils.engines.scene_export.scene_records.SceneRecords",
        "SceneStoreBase": "pythontk.core_utils.engines.scene_export.scene_store.SceneStoreBase",
        "ExportContext": "pythontk.core_utils.engines.scene_export.export_snapshot.ExportContext",
        "ExportSnapshot": "pythontk.core_utils.engines.scene_export.export_snapshot.ExportSnapshot",
        "Producer": "pythontk.core_utils.engines.scene_export.export_snapshot.Producer",
        "TransferContext": "pythontk.core_utils.engines.scene_export.record_transfer.TransferContext",
        "RecordTransfer": "pythontk.core_utils.engines.scene_export.record_transfer.RecordTransfer",
    },
    remove_in="0.13.0",
    since="2026-09-26",
)
