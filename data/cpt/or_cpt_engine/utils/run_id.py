from __future__ import annotations

from datetime import datetime


def make_run_id(run_name: str | None = None) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not run_name:
        return timestamp
    safe_name = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in run_name)
    return f"{timestamp}_{safe_name}"
