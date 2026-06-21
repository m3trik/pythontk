# pythontk — API Changes

_Diff vs prior baseline. Generated 2026-06-21._

## Added (9)

- `core_utils/logging_mixin.py::LoggingMixin.clear_log_buffer(cls) -> None`
- `core_utils/logging_mixin.py::LoggingMixin.disable_log_buffer(cls) -> None`
- `core_utils/logging_mixin.py::LoggingMixin.dump_log(cls, target: Union[str, object, None] = None, mode: str = 'w', encoding: str = 'utf-8') -> str`
- `core_utils/logging_mixin.py::LoggingMixin.enable_log_buffer(cls, capacity: int = 2000, level: Union[int, str] = internal_logging.NOTSET) -> None`
- `core_utils/logging_mixin.py::LoggingMixin.set_log_file(cls, filename: Optional[str], level: Union[int, str] = internal_logging.NOTSET) -> None`
- `core_utils/logging_mixin.py::RingBufferHandler(class)`
- `core_utils/logging_mixin.py::RingBufferHandler.clear(self) -> None`
- `core_utils/logging_mixin.py::RingBufferHandler.emit(self, record: internal_logging.LogRecord) -> None`
- `core_utils/logging_mixin.py::RingBufferHandler.format_records(self, formatter: internal_logging.Formatter = None) -> str`
