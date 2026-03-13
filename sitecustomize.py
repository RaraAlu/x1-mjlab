"""Startup compatibility shims for legacy Isaac Lab import paths.

Python imports ``sitecustomize`` automatically during startup (unless ``-S`` is
used). We use that hook to keep older scripts working with newer Isaac Lab /
mjlab layouts.
"""

from __future__ import annotations

import importlib
import sys
import types


_LEGACY_PARSE_CFG_ALIAS = (
    "mjlab.third_party.isaaclab.isaaclab_tasks.utils.parse_cfg"
)
_PARSE_CFG_CANDIDATES = (
    "isaaclab_tasks.utils.parse_cfg",
    "omni.isaac.lab_tasks.utils.parse_cfg",
)


def _attach_child(parent_name: str, child_name: str) -> None:
    parent = sys.modules.get(parent_name)
    child = sys.modules.get(child_name)
    if parent is None or child is None:
        return
    setattr(parent, child_name.rsplit(".", 1)[-1], child)


def _ensure_package(name: str, parent_name: str | None = None) -> types.ModuleType:
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        module.__path__ = []  # type: ignore[attr-defined]
        sys.modules[name] = module
    if parent_name is not None:
        _attach_child(parent_name, name)
    return module


def _install_legacy_parse_cfg_alias() -> None:
    try:
        importlib.import_module("mjlab")
    except ModuleNotFoundError:
        return

    target_module = None
    for candidate in _PARSE_CFG_CANDIDATES:
        try:
            target_module = importlib.import_module(candidate)
            break
        except ModuleNotFoundError:
            continue

    if target_module is None:
        return

    _ensure_package("mjlab.third_party", "mjlab")
    _ensure_package("mjlab.third_party.isaaclab", "mjlab.third_party")
    _ensure_package(
        "mjlab.third_party.isaaclab.isaaclab_tasks",
        "mjlab.third_party.isaaclab",
    )
    _ensure_package(
        "mjlab.third_party.isaaclab.isaaclab_tasks.utils",
        "mjlab.third_party.isaaclab.isaaclab_tasks",
    )

    sys.modules[_LEGACY_PARSE_CFG_ALIAS] = target_module
    _attach_child(
        "mjlab.third_party.isaaclab.isaaclab_tasks.utils",
        _LEGACY_PARSE_CFG_ALIAS,
    )


_install_legacy_parse_cfg_alias()

