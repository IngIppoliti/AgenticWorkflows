"""Workflow state tracker — maintains the step-by-step plan with status,
assigned worker agent, and approved final output."""

import json
import os
from dataclasses import dataclass, field, asdict
from typing import List, Optional


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

    # ── Accumulation ──────────────────────────────────────────────────

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
        return {
            "prompt": self.prompt,
            "final_output": self.final_output,
            "artifacts": self.artifacts,
            "entries": [e.to_dict() for e in self.entries],
        }

    def save_state(self) -> None:
        """Write current state to JSON file for the dashboard."""
        with open(self._state_file, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    # ── Helpers ───────────────────────────────────────────────────────

    def _find(self, step_number: int) -> Optional[PlanEntry]:
        return next((e for e in self.entries if e.step_number == step_number), None)

    # ── Rendering ─────────────────────────────────────────────────────

    def summary(self) -> str:
        """Human-readable plan overview with status icons."""
        icons = {
            "pending": "⏳",
            "in_progress": "🔄",
            "completed": "✅",
            "failed": "❌",
        }
        lines = [
            "═" * 60,
            f"  PLAN: {self.prompt}",
            "═" * 60,
        ]
        for e in self.entries:
            icon = icons.get(e.status, "?")
            worker = e.assigned_worker or "—"
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
        lines.append("═" * 60)
        completed = sum(1 for e in self.entries if e.status == "completed")
        lines.append(f"  Progress: {completed}/{len(self.entries)} steps completed")
        return "\n".join(lines)

    def to_log_lines(self) -> List[str]:
        """Flat log lines for appending to a test-run log file."""
        lines = [f"Plan: {self.prompt}"]
        for e in self.entries:
            lines.append(
                f"[{e.status.upper()}] Step {e.step_number}"
                f" | Worker: {e.assigned_worker or '-'}"
                f" | Output: {(e.final_output[:80] + '...') if e.final_output and len(e.final_output) > 80 else (e.final_output or '-')}"
            )
        return lines
