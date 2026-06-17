"""Utility to log test inputs and outputs to a dated text file.

Usage:
    from test_logger import log_test_run

    log_test_run(
        test_file=__file__,
        input_data="the prompt / query",
        output_data="the response / result",
    )

Each call appends a record to:
    test_logs/test_runs_YYYY-MM-DD.txt
"""

import os
from datetime import datetime

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_logs")


def log_test_run(
    test_file: str,
    input_data: str,
    output_data: str,
    extra: str = "",
) -> str:
    """Append a test run record to today's log file.

    Args:
        test_file: Name (or path) of the test file that ran.
        input_data: The prompt / query passed to the agent.
        output_data: The response / result from the agent.
        extra: Optional extra info (e.g. status, iterations used).

    Returns:
        The path to the log file.
    """
    os.makedirs(LOG_DIR, exist_ok=True)

    today = datetime.now().strftime("%Y-%m-%d")
    log_path = os.path.join(LOG_DIR, f"test_runs_{today}.txt")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sep = "=" * 72

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"{sep}\n")
        f.write(f"[{timestamp}]  Test: {test_file}\n")
        f.write(f"{sep}\n")
        f.write(f"INPUT:\n{input_data}\n\n")
        f.write(f"OUTPUT:\n{output_data}\n\n")
        if extra:
            f.write(f"EXTRA:\n{extra}\n\n")

    return log_path
