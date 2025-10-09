"""Internal helpers for coordinating warmup behaviour across strategy callers."""

from __future__ import annotations

from typing import Any, Mapping, MutableMapping

DISABLE_WARMUP_FLAG = "__disable_warmup_fetch__"


def apply_disable_warmup_flag(
    params: Mapping[str, Any] | None, *, disable_warmup: bool
) -> dict[str, Any]:
    """Return a copy of ``params`` with the warmup flag normalised."""

    payload: dict[str, Any] = {}
    if isinstance(params, Mapping):
        payload.update(params)
    elif params is not None:
        payload.update(getattr(params, "__dict__", {}))

    if disable_warmup:
        payload[DISABLE_WARMUP_FLAG] = True
    else:
        payload.pop(DISABLE_WARMUP_FLAG, None)
    return payload


def set_disable_warmup_flag_in_place(
    params: MutableMapping[str, Any], *, disable_warmup: bool
) -> None:
    """Mutate ``params`` to ensure the desired warmup behaviour."""

    if disable_warmup:
        params[DISABLE_WARMUP_FLAG] = True
    else:
        params.pop(DISABLE_WARMUP_FLAG, None)

