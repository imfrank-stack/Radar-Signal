from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import h5py
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CLASS_CONFIG_PATH = os.path.join(BASE_DIR, "radchar_classes.json")


@dataclass(frozen=True)
class LabelEntry:
    raw_label: int
    name: str


@dataclass(frozen=True)
class LabelSchema:
    entries: tuple[LabelEntry, ...]

    @property
    def num_classes(self) -> int:
        return len(self.entries)

    @property
    def raw_labels(self) -> tuple[int, ...]:
        return tuple(entry.raw_label for entry in self.entries)

    @property
    def class_names(self) -> tuple[str, ...]:
        return tuple(entry.name for entry in self.entries)

    @property
    def raw_to_model(self) -> dict[int, int]:
        return {entry.raw_label: idx for idx, entry in enumerate(self.entries)}

    @property
    def model_to_raw(self) -> dict[int, int]:
        return {idx: entry.raw_label for idx, entry in enumerate(self.entries)}

    def encode(self, raw_labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raw_labels = np.asarray(raw_labels, dtype=np.int64)
        mapping = self.raw_to_model
        keep_mask = np.isin(raw_labels, list(mapping.keys()))
        encoded = np.full(raw_labels.shape, -1, dtype=np.int64)
        if np.any(keep_mask):
            kept = raw_labels[keep_mask]
            encoded[keep_mask] = np.array([mapping[int(v)] for v in kept], dtype=np.int64)
        return encoded, keep_mask

    def decode(self, model_indices: np.ndarray) -> np.ndarray:
        model_indices = np.asarray(model_indices, dtype=np.int64)
        mapping = self.model_to_raw
        return np.array([mapping[int(v)] for v in model_indices], dtype=np.int64)

    def filter_iq_and_labels(
        self,
        iq: np.ndarray,
        raw_labels: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        encoded, keep_mask = self.encode(raw_labels)
        return iq[keep_mask], encoded[keep_mask], keep_mask

    def name_for_model_index(self, index: int) -> str:
        return self.entries[int(index)].name

    def name_for_raw_label(self, raw_label: int) -> str:
        for entry in self.entries:
            if entry.raw_label == int(raw_label):
                return entry.name
        return f"raw:{int(raw_label)}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "entries": [
                {"raw_label": int(entry.raw_label), "name": entry.name}
                for entry in self.entries
            ]
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LabelSchema":
        entries = []
        for item in payload.get("entries", []):
            if not bool(item.get("enabled", True)):
                continue
            entries.append(LabelEntry(raw_label=int(item["raw_label"]), name=str(item["name"])))
        entries.sort(key=lambda x: x.raw_label)
        if not entries:
            raise ValueError("label schema has no enabled classes")
        return cls(entries=tuple(entries))


def scan_h5_raw_labels(path: str) -> np.ndarray:
    with h5py.File(path, "r") as f:
        labels = f["labels"][:]
    if getattr(labels.dtype, "names", None) and "signal_type" in labels.dtype.names:
        raw = labels["signal_type"]
    else:
        raw = labels
    return np.asarray(raw, dtype=np.int64)


def discover_raw_labels(paths: Iterable[str]) -> list[int]:
    values: set[int] = set()
    for path in paths:
        if path and os.path.isfile(path):
            values.update(int(v) for v in np.unique(scan_h5_raw_labels(path)).tolist())
    return sorted(values)


def build_default_schema(discovered_labels: Iterable[int]) -> LabelSchema:
    entries = [LabelEntry(raw_label=int(v), name=f"类型 {int(v)}") for v in sorted(set(discovered_labels))]
    if not entries:
        raise ValueError("cannot build label schema from empty discovered labels")
    return LabelSchema(entries=tuple(entries))


def ensure_class_config(path: str, discovered_labels: Iterable[int]) -> None:
    if os.path.isfile(path):
        return
    schema = build_default_schema(discovered_labels)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "description": "编辑 enabled/name/raw_label 即可增删类别或修改显示名称",
                "entries": [
                    {"raw_label": entry.raw_label, "name": entry.name, "enabled": True}
                    for entry in schema.entries
                ],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


def load_label_schema(
    config_path: Optional[str] = None,
    *,
    discovered_labels: Optional[Iterable[int]] = None,
    auto_create: bool = True,
    require_all_discovered: bool = False,
) -> LabelSchema:
    path = os.path.abspath(config_path or DEFAULT_CLASS_CONFIG_PATH)
    labels = sorted(set(int(v) for v in (discovered_labels or [])))
    if auto_create and labels:
        ensure_class_config(path, labels)

    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        schema = LabelSchema.from_dict(payload)
    else:
        schema = build_default_schema(labels)

    if labels and require_all_discovered:
        missing = [v for v in labels if v not in schema.raw_to_model]
        if missing:
            raise ValueError(
                f"class config {path} 缺少数据集中的原始标签 {missing}。"
                "请更新配置文件 entries，或重新生成配置。"
            )
    return schema


def schema_from_checkpoint_payload(
    loaded: Any,
    *,
    config_path: Optional[str] = None,
    fallback_num_classes: Optional[int] = None,
) -> LabelSchema:
    if isinstance(loaded, dict):
        for key in ("label_schema", "class_schema"):
            if key in loaded and isinstance(loaded[key], dict):
                return LabelSchema.from_dict(loaded[key])
        if "config" in loaded and isinstance(loaded["config"], dict):
            nested = loaded["config"].get("label_schema")
            if isinstance(nested, dict):
                return LabelSchema.from_dict(nested)

    if config_path and os.path.isfile(config_path):
        return load_label_schema(config_path=config_path, auto_create=False)

    if fallback_num_classes is None:
        raise ValueError("无法从 checkpoint 或配置文件恢复类别定义")
    return build_default_schema(range(fallback_num_classes))


def infer_num_classes_from_state_dict(state: dict[str, Any]) -> int:
    for key in (
        "classifier.weight",
        "classifier.bias",
        "main_head.3.weight",
        "main_head.3.bias",
        "main_head.5.weight",
        "main_head.5.bias",
        "head.5.weight",
        "head.5.bias",
    ):
        if key in state:
            return int(state[key].shape[0])
    candidate_keys = [
        key
        for key, value in state.items()
        if isinstance(value, np.ndarray) is False
        and hasattr(value, "ndim")
        and hasattr(value, "shape")
        and int(value.ndim) == 2
        and (
            str(key).startswith("classifier")
            or str(key).startswith("main_head")
            or str(key).startswith("head")
        )
    ]
    if candidate_keys:
        candidate_keys.sort()
        return int(state[candidate_keys[-1]].shape[0])
    raise ValueError("cannot infer class count from checkpoint state_dict")
