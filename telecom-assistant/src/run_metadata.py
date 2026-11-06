"""
Reproducibility helpers: collect environment fingerprints and eval run envelopes.

Evaluation JSON may be either legacy (top-level list) or v2 envelope:

  {"schema_version": 2, "run": {"metadata": ..., "config": ...}, "results": [...]}
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 2


def project_root() -> Path:
    """`telecom-assistant/` (parent of `src/`)."""
    return Path(__file__).resolve().parent.parent


def find_git_root(start: Path) -> Path | None:
    p = start.resolve()
    for _ in range(20):
        if (p / ".git").exists():
            return p
        if p.parent == p:
            return None
        p = p.parent
    return None


def _git_cmd(repo: Path, *args: str) -> str | None:
    try:
        r = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if r.returncode != 0:
            return None
        return (r.stdout or "").strip() or None
    except (OSError, subprocess.TimeoutExpired):
        return None


def collect_git_info(cwd: Path) -> dict[str, Any]:
    root = find_git_root(cwd)
    if root is None:
        return {"present": False}
    commit = _git_cmd(root, "rev-parse", "HEAD")
    branch = _git_cmd(root, "rev-parse", "--abbrev-ref", "HEAD")
    dirty_out = _git_cmd(root, "status", "--porcelain")
    return {
        "present": True,
        "root": str(root),
        "commit": commit,
        "branch": branch,
        "dirty": bool(dirty_out),
    }


def _pkg_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _import_version(module: str, attr: str = "__version__") -> str | None:
    try:
        m = __import__(module, fromlist=["_"])
        v = getattr(m, attr, None)
        return str(v) if v is not None else None
    except Exception:
        return None


def collect_package_versions() -> dict[str, str | None]:
    """Prefer distribution metadata; fall back to `module.__version__` when missing."""
    dist_names = [
        "sympy",
        "chromadb",
        "langchain-core",
        "langchain-openai",
        "langchain-huggingface",
        "langchain-chroma",
        "langchain-community",
        "langchain-text-splitters",
        "pypdf",
        "streamlit",
        "sentence-transformers",
    ]
    out: dict[str, str | None] = {n: _pkg_version(n) for n in dist_names}
    key_map = {
        "sympy": "sympy",
        "chromadb": "chromadb",
        "langchain_core": "langchain-core",
        "langchain_openai": "langchain-openai",
        "langchain_huggingface": "langchain-huggingface",
        "langchain_chroma": "langchain-chroma",
        "langchain_community": "langchain-community",
        "langchain_text_splitters": "langchain-text-splitters",
        "pypdf": "pypdf",
        "streamlit": "streamlit",
        "sentence_transformers": "sentence-transformers",
    }
    for mod, dist_key in key_map.items():
        if out.get(dist_key) is None:
            v = _import_version(mod)
            if v is not None:
                out[dist_key] = v
    return out


def _sha256_file(path: Path, chunk: int = 1024 * 1024) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def fingerprint_pdfs(pdf_dir: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    if not pdf_dir.is_dir():
        return {"pdf_dir": str(pdf_dir), "exists": False, "files": []}
    for name in sorted(pdf_dir.iterdir()):
        if not name.is_file() or name.suffix.lower() != ".pdf":
            continue
        st = name.stat()
        rows.append(
            {
                "name": name.name,
                "bytes": st.st_size,
                "mtime_epoch": int(st.st_mtime),
                "sha256": _sha256_file(name),
            }
        )
    return {"pdf_dir": str(pdf_dir), "exists": True, "files": rows}


def fingerprint_chroma(vectorstore_dir: Path) -> dict[str, Any]:
    sqlite = vectorstore_dir / "chroma.sqlite3"
    out: dict[str, Any] = {
        "vectorstore_dir": str(vectorstore_dir),
        "chroma_sqlite_exists": sqlite.is_file(),
        "chroma_sqlite_sha256": _sha256_file(sqlite) if sqlite.is_file() else None,
    }
    if vectorstore_dir.is_dir():
        extra_dirs = [p for p in vectorstore_dir.iterdir() if p.is_dir()]
        out["uuid_subdirs"] = sorted(p.name for p in extra_dirs)
    return out


def load_eval_run_config(
    root: Path | None = None,
    override_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Load declarative eval config JSON (no secrets)."""
    root = root or project_root()
    path = (
        Path(override_path)
        if override_path
        else root / "data" / "eval_run_config.json"
    )
    if not path.is_file():
        return {"_error": f"missing config file: {path}"}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def build_run_metadata(
    *,
    root: Path | None = None,
    eval_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Runtime fingerprints to store next to every eval JSON write."""
    root = root or project_root()
    cfg = eval_config if eval_config is not None else load_eval_run_config(root)
    pdf_dir = root / str(cfg.get("ingest", {}).get("pdf_dir", "data/pdfs"))
    vs_dir = root / str(cfg.get("ingest", {}).get("vectorstore_dir", "vectorstore"))

    return {
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "cwd": str(Path.cwd()),
        "project_root": str(root),
        "git": collect_git_info(root),
        "packages": collect_package_versions(),
        "fingerprints": {
            "pdfs": fingerprint_pdfs(pdf_dir),
            "vectorstore": fingerprint_chroma(vs_dir),
        },
        "api_keys_present": {
            "CEREBRAS_API_KEY": bool(os.getenv("CEREBRAS_API_KEY")),
            "OPENAI_API_KEY": bool(os.getenv("OPENAI_API_KEY")),
        },
    }


def new_run_block(
    *,
    root: Path | None = None,
    eval_config: dict[str, Any] | None = None,
    started_at_utc: str | None = None,
) -> dict[str, Any]:
    """Initial `run` object (metadata + frozen config snapshot)."""
    root = root or project_root()
    cfg = eval_config if eval_config is not None else load_eval_run_config(root)
    meta = build_run_metadata(root=root, eval_config=cfg)
    if started_at_utc is None:
        started_at_utc = meta["collected_at_utc"]
    meta["started_at_utc"] = started_at_utc
    meta["last_updated_at_utc"] = started_at_utc
    return {"metadata": meta, "config": cfg}


def touch_run_block(run_block: dict[str, Any]) -> None:
    """Cheap update before each incremental JSON write (no re-hashing)."""
    run_block["metadata"]["last_updated_at_utc"] = (
        datetime.now(timezone.utc).isoformat()
    )


def make_eval_envelope(run_block: dict[str, Any], results: list[Any]) -> dict[str, Any]:
    touch_run_block(run_block)
    return {
        "schema_version": SCHEMA_VERSION,
        "run": run_block,
        "results": results,
    }


def unwrap_eval_results(data: Any) -> list[Any]:
    """Accept legacy list or v2 envelope; return the per-query list."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("results"), list):
        return data["results"]
    raise ValueError("Unrecognised eval JSON shape (expected list or envelope with results).")


def load_eval_json_file(path: str | os.PathLike[str]) -> list[Any]:
    with Path(path).open(encoding="utf-8") as f:
        return unwrap_eval_results(json.load(f))
