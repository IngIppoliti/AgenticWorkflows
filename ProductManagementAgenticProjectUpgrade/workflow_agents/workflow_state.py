"""Workflow state tracker — maintains the step-by-step plan with status,
assigned worker agent, and approved final output.

Uses atomic file writes (write to temp file, then rename) to prevent
dashboard readers from seeing partially-written files on Windows.
Retries on PermissionError (Windows file locking) with backoff.
"""

import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass, field, asdict
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PlanEntry:
    """Single step within a project plan."""

    step_number: int
    task_description: str
    status: str = "pending"  # pending | in_progress | completed | failed
    assigned_worker: Optional[str] = None
    final_output: Optional[str] = None
    error_message: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PlanState:
    """Full state of a workflow plan — entries + metadata."""

    prompt: str = ""
    entries: List[PlanEntry] = field(default_factory=list)
    final_output: str = ""
    artifacts: dict = field(default_factory=lambda: {
        "user_stories": [],
        "features": [],
        "tasks": [],
    })
    _state_file: str = ""

    def __post_init__(self):
        # Default state file path next to this module
        if not self._state_file:
            self._state_file = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "plan_state.json"
            )

    # ── Mutators ──────────────────────────────────────────────────────

    def add_steps(self, steps: List[str]) -> None:
        """Populate plan from a list of task descriptions."""
        for idx, step in enumerate(steps, 1):
            self.entries.append(
                PlanEntry(step_number=idx, task_description=step)
            )
        self.save_state()

    def mark_in_progress(self, step_number: int) -> None:
        entry = self._find(step_number)
        if entry:
            entry.status = "in_progress"
            self.save_state()

    def mark_completed(
        self, step_number: int, worker: str, output: str
    ) -> None:
        entry = self._find(step_number)
        if entry:
            entry.status = "completed"
            entry.assigned_worker = worker
            entry.final_output = output
            self.save_state()

    def mark_failed(self, step_number: int, error: str) -> None:
        entry = self._find(step_number)
        if entry:
            entry.status = "failed"
            entry.error_message = error
            self.save_state()

    # ── Serialization ─────────────────────────────────────────────────

    def set_final_output(self, output: str) -> None:
        """Set the accumulated final output shown at the end of the plan."""
        self.final_output = output
        self.save_state()

    def add_artifact(self, category: str, item: str) -> None:
        """Append a structured artifact (user story, feature, task)."""
        if category in self.artifacts:
            self.artifacts[category].append(item)
            self.save_state()

    # ── Serialization ─────────────────────────────────────────────────

    def to_dict(self) -> dict:
        d = {
            "prompt": self.prompt,
            "final_output": self.final_output,
            "artifacts": self.artifacts,
            "entries": [e.to_dict() for e in self.entries],
        }
        return d

    def save_state(self) -> None:
        """Atomically write state to JSON file.

        Writes to a temporary file first, then renames it to the target
        path. Retries on PermissionError (Windows file locking) with
        exponential backoff (50ms → 100ms → 200ms → 400ms → 800ms).
        """
        data = self.to_dict()

        # Write to a temp file in the same directory (same filesystem)
        dir_name = os.path.dirname(self._state_file)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=dir_name,
            prefix="plan_state_tmp_",
            suffix=".json",
            delete=False,
        ) as tmp:
            tmp_path = tmp.name
            json.dump(data, tmp, indent=2, ensure_ascii=False)

        # Atomic rename — retry on Windows file-locking contention
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                os.replace(tmp_path, self._state_file)
                return
            except PermissionError:
                if attempt < max_attempts - 1:
                    delay = 0.05 * (2 ** attempt)
                    logger.warning(
                        "File lock on %s, retry %d/%d in %.0fms",
                        os.path.basename(self._state_file),
                        attempt + 1,
                        max_attempts,
                        delay * 1000,
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        "Failed to write %s after %d attempts",
                        self._state_file,
                        max_attempts,
                    )
                    raise

    # ── Helpers ───────────────────────────────────────────────────────

    def _find(self, step_number: int) -> Optional[PlanEntry]:
        for e in self.entries:
            if e.step_number == step_number:
                return e
        return None

    # ── Rendering ─────────────────────────────────────────────────────

    def summary(self) -> str:
        """Human-readable plan overview with status icons."""
        icons = {
            "pending": "\u23f3",
            "in_progress": "\U0001f504",
            "completed": "\u2705",
            "failed": "\u274c",
        }
        lines = [
            "\u2550" * 60,
            f"  PLAN: {self.prompt}",
            "\u2550" * 60,
        ]
        for e in self.entries:
            icon = icons.get(e.status, "?")
            worker = e.assigned_worker or "\u2014"
            output_preview = (
                (e.final_output[:60] + "...")
                if e.final_output and len(e.final_output) > 60
                else (e.final_output or "")
            )
            error = f"  ERROR: {e.error_message}" if e.error_message else ""
            lines.append(f"  {icon} Step {e.step_number}: {e.task_description}")
            lines.append(f"     Worker: {worker}")
            if output_preview:
                lines.append(f"     Output: {output_preview}")
            if error:
                lines.append(f"     {error}")
        lines.append("\u2550" * 60)
        completed = sum(1 for e in self.entries if e.status == "completed")
        lines.append(f"  Progress: {completed}/{len(self.entries)} steps completed")
        return "\n".join(lines)

    def to_log_lines(self) -> List[str]:
        """Flat log lines for appending to a test-run log file."""
        lines = [f"Plan: {self.prompt}"]
        for e in self.entries:
            out = e.final_output or "-"
            if len(out) > 80:
                out = out[:80] + "..."
            lines.append(
                f"[{e.status.upper()}] Step {e.step_number}"
                f" | Worker: {e.assigned_worker or '-'}"
                f" | Output: {out}"
            )
        return lines
