"""Inspect a HuggingFace dataset WITHOUT downloading it.

Uses the free datasets-server API:
  /info        -> configs, splits, column names & feature types
  /size        -> number of rows
  /first-rows  -> a few sample rows
Plus the Hub for the README (dataset card).
"""

from __future__ import annotations

from ..utils import truncate
from . import _http

_DS = "https://datasets-server.huggingface.co"


_canon_cache: dict[str, str] = {}


def _canonical(dataset: str) -> str:
    """Resolve a possibly-renamed/bare dataset name to a valid hub id."""
    if dataset in _canon_cache:
        return _canon_cache[dataset]
    try:
        _http.get(f"{_DS}/is-valid", params={"dataset": dataset}, retries=1)  # raises unless HTTP 200
        _canon_cache[dataset] = dataset
        return dataset
    except Exception:
        pass
    # Fall back to a Hub search for the bare name.
    try:
        from huggingface_hub import HfApi

        for d in HfApi().list_datasets(search=dataset, limit=1):
            _canon_cache[dataset] = d.id
            return d.id
    except Exception:
        pass
    _canon_cache[dataset] = dataset
    return dataset


def _get(path: str, params: dict) -> dict | None:
    try:
        return _http.get(f"{_DS}/{path}", params=params)
    except Exception:
        return None


def dataset_readme(dataset: str) -> dict:
    """Fetch the dataset card (README.md) text from the Hub."""
    try:
        url = f"https://huggingface.co/datasets/{dataset}/raw/main/README.md"
        text = _http.get(url, as_json=False)
        # Strip YAML front-matter for readability.
        if text.startswith("---"):
            parts = text.split("---", 2)
            if len(parts) == 3:
                text = parts[2].strip()
        return {"text": f"README for {dataset}:\n\n{truncate(text, 4000)}"}
    except Exception as exc:
        return {"text": f"Could not fetch README for {dataset}: {exc}"}


def dataset_schema(dataset: str) -> dict:
    """Column names + feature types and available splits/configs."""
    info = _get("info", {"dataset": dataset})
    if not info or "dataset_info" not in info:
        return {"text": f"No schema info available for {dataset} (may be gated/private)."}

    di = info["dataset_info"]
    # dataset_info maps config -> {features, splits, ...}
    out_lines = [f"Schema for {dataset}:"]
    for config, cfg in di.items():
        feats = cfg.get("features", {}) or {}
        cols = ", ".join(f"{k} ({_feat_type(v)})" for k, v in feats.items())
        splits = ", ".join((cfg.get("splits") or {}).keys())
        out_lines.append(f"- config '{config}': splits=[{splits}]")
        out_lines.append(f"    columns: {cols or '(unknown)'}")
    return {"text": "\n".join(out_lines)}


def _feat_type(v) -> str:
    if isinstance(v, dict):
        return v.get("dtype") or v.get("_type") or "feature"
    if isinstance(v, list):
        return "sequence"
    return str(v)


def dataset_size(dataset: str) -> dict:
    """Number of rows (per split) without downloading."""
    size = _get("size", {"dataset": dataset})
    if not size:
        return {"text": f"No size info for {dataset}."}
    s = (size.get("size") or {}).get("dataset") or {}
    nrows = s.get("num_rows")
    splits = (size.get("size") or {}).get("splits") or []
    detail = ", ".join(f"{sp.get('split')}={sp.get('num_rows')}" for sp in splits)
    return {"text": f"{dataset}: ~{nrows} rows total" + (f" ({detail})" if detail else "")}


def dataset_preview(dataset: str, config: str = None, split: str = "train", rows: int = 5) -> dict:
    """A few sample rows from the dataset."""
    params = {"dataset": dataset, "split": split}
    if config:
        params["config"] = config
    else:
        # discover a default config
        info = _get("info", {"dataset": dataset})
        if info and info.get("dataset_info"):
            params["config"] = next(iter(info["dataset_info"]))
    data = _get("first-rows", params)
    if not data or "rows" not in data:
        return {"text": f"No preview rows available for {dataset}."}

    cols = [c["name"] for c in data.get("features", [])]
    sample = data["rows"][:rows]
    lines = [f"Preview of {dataset} (cols: {', '.join(cols)}):"]
    for r in sample:
        row = r.get("row", {})
        lines.append("  " + truncate(str(row), 220))
    return {"text": "\n".join(lines)}


def dataset_inspect(dataset: str) -> dict:
    """One-shot overview: schema + size + a preview + README head."""
    dataset = _canonical(dataset)
    parts = [
        dataset_schema(dataset)["text"],
        dataset_size(dataset)["text"],
        dataset_preview(dataset)["text"],
        truncate(dataset_readme(dataset)["text"], 1200),
    ]
    return {"text": "\n\n".join(parts)}
