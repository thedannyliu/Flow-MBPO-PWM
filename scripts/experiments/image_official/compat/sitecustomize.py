"""Small runtime compatibility shims for official image-model jobs."""

try:
    import pyarrow as _pa
except Exception:
    _pa = None

if _pa is not None and not hasattr(_pa, "PyExtensionType") and hasattr(_pa, "ExtensionType"):
    _pa.PyExtensionType = _pa.ExtensionType
