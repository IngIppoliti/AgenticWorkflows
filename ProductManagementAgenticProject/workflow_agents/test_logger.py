"""Utility to log test inputs and outputs to text files.

Usage:
    from test_logger import log_test_run

    log_test_run(
        test_file=__file__,
        input_data="the prompt / query",
        output_data="the response / result",
    )

Each call writes two files:
  1. test_logs/test_runs_YYYY-MM-DD.txt  – shared daily log (appended)
  2. test_outputs/<test_name>.txt        – one file per test (overwritten)
"""

import os
from datetime import datetime

_BASE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(_BASE, "test_logs")
OUTPUT_DIR = os.path.join(_BASE, "test_outputs")


def log_test_run(
    test_file: str,
    input_data: str,
    output_data: str,
    extra: str = "",
) -> str:
    """Write a test run record to the shared daily log and to its own file.

    Args:
        test_file: Name (or path) of the test file that ran.
        input_data: The prompt / query passed to the agent.
        output_data: The response / result from the agent.
        extra: Optional extra info (e.g. status, iterations used).

    Returns:
        The path to the per-test output file.
    """
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sep = "=" * 72

    # Build the record text once; reuse for both destinations.
    test_name = os.path.basename(test_file)
    lines = [
        sep,
        f"[{timestamp}]  Test: {test_name}",
        sep,
        f"Command:  python {test_name}",
        "",
        f"INPUT:\n{input_data}",
        "",
        f"OUTPUT:\n{output_data}",
        "",
    ]
    if extra:
        lines.append(f"EXTRA:\n{extra}")
        lines.append("")
    record = "\n".join(lines) + "\n"

    # 1. Append to shared daily log.
    today = datetime.now().strftime("%Y-%m-%d")
    shared_log = os.path.join(LOG_DIR, f"test_runs_{today}.txt")
    with open(shared_log, "a", encoding="utf-8") as f:
        f.write(record)

    # 2. Write (overwrite) per-test file in test_outputs/.
    stem = os.path.splitext(test_name)[0]
    output_path = os.path.join(OUTPUT_DIR, f"{stem}.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(record)

    return output_path
