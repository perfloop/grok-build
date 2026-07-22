#!/usr/bin/env python3
"""Run the Responses continuation test binary as one proof sample.

The Rust test supplies the repository-native ACP/mock-server workload. This
adapter deliberately owns only proof-JSONL extraction: it waits for the entire
test binary, propagates its real exit status, and emits metrics only after both
the cursor-chain and model-switch assertions have passed.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Sequence

EXPECTED_METRICS = {
    "responses_continuation_request_bytes",
    "responses_continuation_input_items",
}
MARKER = "PERFLOOP_JSON:"


def main(argv: Sequence[str]) -> int:
    if len(argv) != 3:
        print(f"usage: {argv[0]} TEST_BINARY PAGER_BINARY", file=sys.stderr)
        return 2

    test_binary, pager_binary = argv[1:]
    environment = os.environ.copy()
    environment["GROK_BINARY"] = pager_binary
    completed = subprocess.run(
        [
            test_binary,
            "responses_continuation_",
            "--ignored",
            "--nocapture",
        ],
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode:
        sys.stderr.write(completed.stdout)
        sys.stderr.write(completed.stderr)
        return completed.returncode

    samples: list[dict[str, object]] = []
    try:
        for line in completed.stdout.splitlines():
            if MARKER in line:
                samples.append(json.loads(line.split(MARKER, 1)[1]))
    except json.JSONDecodeError as error:
        print(f"invalid proof JSON emitted by test binary: {error}", file=sys.stderr)
        return 1

    metrics = {sample.get("metric") for sample in samples}
    if len(samples) != len(EXPECTED_METRICS) or metrics != EXPECTED_METRICS:
        print(
            "expected exactly one sample for each Responses continuation metric; "
            f"got {samples!r}",
            file=sys.stderr,
        )
        return 1
    if any(not isinstance(sample.get("value"), (int, float)) for sample in samples):
        print(f"non-numeric Responses continuation sample: {samples!r}", file=sys.stderr)
        return 1

    for sample in samples:
        print(json.dumps(sample, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
