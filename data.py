from importlib import import_module as _import_module
import sys as _sys

_impl = _import_module("data_layer.backend")
for _name, _value in vars(_impl).items():
    if not _name.startswith("__"):
        globals()[_name] = _value
_sys.modules[__name__] = _impl
