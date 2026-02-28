from __future__ import annotations

import gzip
import json
import os
import pickle
from pathlib import Path
from typing import Any, Iterable


class DeserializationSecurityError(RuntimeError):
    """Raised when deserialization is blocked by policy."""


_ALLOW_UNSAFE_ENV = "CATPRED_ALLOW_UNSAFE_DESERIALIZATION"
_TRUSTED_ROOTS_ENV = "CATPRED_TRUSTED_DESERIALIZATION_ROOTS"


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    unique: list[Path] = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        key = str(resolved)
        if key not in seen:
            unique.append(resolved)
            seen.add(key)
    return unique


def _default_trusted_roots() -> list[Path]:
    raw = os.environ.get(_TRUSTED_ROOTS_ENV)
    if raw:
        candidates = [Path(item) for item in raw.split(os.pathsep) if item.strip()]
    else:
        cwd = Path.cwd().resolve()
        candidates = [cwd, cwd.parent]
    return _dedupe_paths(candidates)


def trusted_roots(extra_roots: Iterable[str | Path] | None = None) -> list[Path]:
    roots = _default_trusted_roots()
    if extra_roots:
        roots.extend(Path(path) for path in extra_roots)
    return _dedupe_paths(roots)


def is_trusted_path(path: str | Path, roots: Iterable[str | Path] | None = None) -> bool:
    resolved = Path(path).resolve()
    candidate_roots = trusted_roots(roots)
    for root in candidate_roots:
        try:
            resolved.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def ensure_trusted_path(
    path: str | Path,
    *,
    purpose: str,
    roots: Iterable[str | Path] | None = None,
) -> Path:
    resolved = Path(path).resolve()
    candidate_roots = trusted_roots(roots)
    if is_trusted_path(resolved, candidate_roots):
        return resolved
    roots_display = ", ".join(str(item) for item in candidate_roots)
    raise DeserializationSecurityError(
        f'Refusing to load untrusted {purpose} from "{resolved}". '
        f"Allowed roots: {roots_display}. "
        f"Use {_TRUSTED_ROOTS_ENV} to expand trusted roots."
    )


def unsafe_deserialization_enabled(default: bool = True) -> bool:
    return _env_flag(_ALLOW_UNSAFE_ENV, default=default)


def load_pickle_artifact(
    path: str | Path,
    *,
    purpose: str,
    roots: Iterable[str | Path] | None = None,
    allow_unsafe: bool | None = None,
    encoding: str | None = None,
) -> Any:
    resolved = ensure_trusted_path(path, purpose=purpose, roots=roots)
    unsafe = unsafe_deserialization_enabled() if allow_unsafe is None else allow_unsafe
    if not unsafe:
        raise DeserializationSecurityError(
            f"Pickle deserialization is disabled for {purpose}. "
            f"Set {_ALLOW_UNSAFE_ENV}=1 only for trusted artifacts."
        )

    with resolved.open("rb") as handle:
        if encoding is None:
            try:
                return pickle.load(handle)
            except UnicodeDecodeError:
                handle.seek(0)
                return pickle.load(handle, encoding="latin1")
        return pickle.load(handle, encoding=encoding)


def load_index_artifact(
    path: str | Path,
    *,
    purpose: str,
    roots: Iterable[str | Path] | None = None,
    allow_unsafe: bool | None = None,
) -> Any:
    resolved = ensure_trusted_path(path, purpose=purpose, roots=roots)
    suffixes = tuple(s.lower() for s in resolved.suffixes)
    if suffixes and suffixes[-1] == ".json":
        with resolved.open("rt", encoding="utf-8") as handle:
            return json.load(handle)
    if suffixes[-2:] == (".json", ".gz"):
        with gzip.open(resolved, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return load_pickle_artifact(
        resolved,
        purpose=purpose,
        roots=roots,
        allow_unsafe=allow_unsafe,
    )


def load_torch_artifact(
    path: str | Path,
    *,
    purpose: str,
    map_location=None,
    roots: Iterable[str | Path] | None = None,
    allow_unsafe: bool | None = None,
) -> Any:
    resolved = ensure_trusted_path(path, purpose=purpose, roots=roots)
    unsafe = unsafe_deserialization_enabled() if allow_unsafe is None else allow_unsafe

    import torch

    if unsafe:
        try:
            return torch.load(str(resolved), map_location=map_location, weights_only=False)
        except TypeError:
            return torch.load(str(resolved), map_location=map_location)

    try:
        return torch.load(str(resolved), map_location=map_location, weights_only=True)
    except TypeError as exc:
        raise DeserializationSecurityError(
            "Safe torch deserialization requires a torch version that supports weights_only loading. "
            f"Set {_ALLOW_UNSAFE_ENV}=1 for trusted legacy checkpoints."
        ) from exc
    except Exception as exc:
        raise DeserializationSecurityError(
            f"Safe torch deserialization rejected {purpose}. "
            f"If this checkpoint is trusted, set {_ALLOW_UNSAFE_ENV}=1."
        ) from exc
