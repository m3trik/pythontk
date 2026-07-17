# pythontk — API Changes

_Diff vs prior baseline. Generated 2026-07-17._

## Signature changed (1)

- `file_utils/temp_artifacts.py::TempArtifacts.path`
  - was: `(self, extension: str = '.fbx', name: Optional[str] = None) -> str`
  - now: `(self, extension: str = '.tmp', name: Optional[str] = None) -> str`
