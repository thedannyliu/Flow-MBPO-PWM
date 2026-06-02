"""Small runtime compatibility shims for official image-model jobs."""

from pathlib import Path
import sys

_vendor = Path(__file__).with_name("vendor")
if _vendor.exists() and str(_vendor) not in sys.path:
    sys.path.insert(0, str(_vendor))

try:
    import pyarrow as _pa
except Exception:
    _pa = None

if _pa is not None and not hasattr(_pa, "PyExtensionType") and hasattr(_pa, "ExtensionType"):
    _pa.PyExtensionType = _pa.ExtensionType

try:
    import stable_worldmodel.data as _swm_data
    from stable_worldmodel.data.formats.hdf5 import HDF5Dataset, HDF5Writer
except Exception:
    _swm_data = None
else:
    if not hasattr(_swm_data, "HDF5Dataset"):
        _swm_data.HDF5Dataset = HDF5Dataset
    if not hasattr(_swm_data, "HDF5Writer"):
        _swm_data.HDF5Writer = HDF5Writer
