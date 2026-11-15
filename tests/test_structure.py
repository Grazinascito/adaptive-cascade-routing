"""Foundation test: every skeleton module under src/ imports without error."""

import importlib

SKELETON_MODULES = (
    "src",
    "src.data",
    "src.data.loader",
    "src.engine",
    "src.engine.server",
    "src.engine.system1",
    "src.engine.system2",
    "src.engine.router",
    "src.metrics",
    "src.metrics.summary",
    "src.metrics.evaluator",
)


def test_every_skeleton_module_imports_without_error() -> None:
    for module_name in SKELETON_MODULES:
        importlib.import_module(module_name)
