"""Peak memory and time to stream a file at a given chunk budget.

    uv run python benchmarks/chunk_memory.py FILE [BUDGET_MB ...]

Run on the target server to choose ``chunk_memory_bytes``. Each budget runs
in its own process so peak RSS is not shared between runs.
"""

from __future__ import annotations

import resource
import subprocess
import sys
import time


def run(path: str, budget_mb: int) -> None:
    from metadata_tools import MiB, open_reader

    start = time.perf_counter()
    rows = 0
    biggest = 0
    with open_reader(path, chunk_memory_bytes=budget_mb * MiB) as reader:
        for chunk in reader.iter_chunks():
            rows += len(chunk)
            biggest = max(biggest, int(chunk.memory_usage(deep=True).sum()))
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes = peak if sys.platform == "darwin" else peak * 1024
    print(
        f"budget={budget_mb:>5}MB rows={rows:>12,} biggest_chunk={biggest / 1e6:>7.0f}MB "
        f"peak_rss={peak_bytes / 1e6:>7.0f}MB time={time.perf_counter() - start:>6.1f}s "
        f"warnings={len(reader.warnings)}"
    )


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    if sys.argv[1] == "--one":
        run(sys.argv[2], int(sys.argv[3]))
        return
    path = sys.argv[1]
    budgets = [int(b) for b in sys.argv[2:]] or [64, 256, 1024]
    print(path)
    for budget in budgets:
        subprocess.run([sys.executable, __file__, "--one", path, str(budget)], check=True)


if __name__ == "__main__":
    main()
