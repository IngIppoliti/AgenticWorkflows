"""
Deterministic workflow orchestrator for product management.

Replaces the non-deterministic action-planning + embedding-based routing
with a fixed 3-phase pipeline:

  Phase 1 — Features        (Program Manager)
  Phase 2 — User Stories    (Product Manager, seeded with features)
  Phase 3 — Development Tasks (Development Engineer, seeded with stories)

Each phase explicitly receives the output of the previous phase as context,
guaranteeing hierarchical consistency (Feature → User Story → Task).
"""

import logging

from config import AGENT_CONFIG, load_product_specification

from base_agents import (
    EvaluationAgent,
    KnowledgeAugmentedPromptAgent,
)
from workflow_state import PlanState

logger = logging.getLogger(__name__)


# ============================================================================
# DETERMINISTIC WORKFLOW ORCHESTRATOR
# ============================================================================


class ProductWorkflowOrchestrator:
    """
    Runs exactly three deterministic phases in order:

    1. Program Manager  → Product Features
    2. Product Manager  → User Stories (scoped by features from
       phase 1)
    3. Development Engineer → Development Tasks (scoped by stories
       from phase 2)

    Each phase feeds its output into the next, forming a strict
    hierarchy. No embedding-based routing — the agent-per-phase is
    fixed by design.
    """

    PHASE_NAMES = {
        1: ("Program Manager", "features"),
        2: ("Product Manager", "user_stories"),
        3: ("Development Engineer", "tasks"),
    }

    def __init__(self, product_spec: str = ""):
        self.product_spec = product_spec or load_product_specification()
        self.product_spec_loaded_from = (
            "file" if not product_spec else "parameter"
        )

        # ── Phase 1: Program Manager → Features ──────────────────────
        self._agent_fm = self._build_agent("program_manager")
        self._eval_fm = EvaluationAgent(max_interactions=3)
        self._criteria_fm = AGENT_CONFIG["program_manager"]["eval_criteria"]

        # ── Phase 2: Product Manager → User Stories ──────────────────
        self._agent_pm = self._build_agent("product_manager")
        self._eval_pm = EvaluationAgent(max_interactions=3)
        self._criteria_pm = AGENT_CONFIG["product_manager"]["eval_criteria"]

        # ── Phase 3: Development Engineer → Tasks ────────────────────
        self._agent_dev = self._build_agent("development_engineer")
        self._eval_dev = EvaluationAgent(max_interactions=3)
        cfg_dev = AGENT_CONFIG["development_engineer"]
        self._criteria_dev = cfg_dev["eval_criteria"]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        user_prompt: str = "Create a complete development plan",
    ) -> PlanState:
        """
        Execute the full 3-phase pipeline.

        Returns a PlanState with:
        - 3 plan entries (one per phase), each with status & output
        - artifacts dict under ``features``, ``user_stories``, ``tasks``
        - ``final_output`` containing the concatenated result
        """
        context = (
            f"{user_prompt}\n\nProduct Specification:\n{self.product_spec}"
        )
        logger.info(
            "═══ ProductWorkflowOrchestrator started ═══\n"
        )

        plan = PlanState(prompt=user_prompt)
        plan.add_steps([
            "Define product features based on the specification",
            "Define user stories for each feature",
            "Define development tasks for each user story",
        ])

        # ── Phase 1: Features ────────────────────────────────────────
        logger.info("─" * 50)
        logger.info("PHASE 1/3 — Program Manager: Product Features")
        logger.info("─" * 50)

        features_prompt = (
            f"{context}\n\n"
            "Based on the product specification above, identify and "
            "describe the key product features. For each feature "
            "provide:\n"
            "- Feature Name\n"
            "- Description\n"
            "- Key Functionality\n"
            "- User Benefit"
        )
        features_output = self._run_phase(
            plan=plan,
            step_number=1,
            agent=self._agent_fm,
            evaluator=self._eval_fm,
            prompt=features_prompt,
            criteria=self._criteria_fm,
            artifact_key="features",
        )

        # ── Phase 2: User Stories ────────────────────────────────────
        logger.info("─" * 50)
        logger.info("PHASE 2/3 — Product Manager: User Stories")
        logger.info("─" * 50)

        stories_prompt = (
            f"{context}\n\n"
            f"Here are the product features identified:\n\n"
            f"{features_output}\n\n"
            "Based on these features, write user stories for each "
            "feature. Each user story must follow the format:\n"
            "'As a [type of user], I want [action] so that [benefit].'"
            "You can also write more than one user story per feature if needed. "
            "Examples of user stories:\n"
            "- As a [type of user], I want to [action] so that [benefit].\n"
            "- As a [type of user], I want to [action] so that [benefit]."
        )
        stories_output = self._run_phase(
            plan=plan,
            step_number=2,
            agent=self._agent_pm,
            evaluator=self._eval_pm,
            prompt=stories_prompt,
            criteria=self._criteria_pm,
            artifact_key="user_stories",
        )

        # ── Phase 3: Development Tasks ───────────────────────────────
        logger.info("─" * 50)
        logger.info("PHASE 3/3 — Development Engineer: Development Tasks")
        logger.info("─" * 50)

        tasks_prompt = (
            f"{context}\n\n"
            f"Here are the user stories to implement:\n\n"
            f"{stories_output}\n\n"
            "For each user story, define the specific development tasks required."
            "You can also write more than one task per user story if needed."
            " Each task must include:\n"
            "- Task ID\n"
            "- Task Title\n"
            "- Related User Story\n"
            "- Description\n"
            "- Acceptance Criteria\n"
            "- Estimated Effort\n"
            "- Dependencies"
        )
        tasks_output = self._run_phase(
            plan=plan,
            step_number=3,
            agent=self._agent_dev,
            evaluator=self._eval_dev,
            prompt=tasks_prompt,
            criteria=self._criteria_dev,
            artifact_key="tasks",
        )

        # ── Build final aggregated output ────────────────────────────
        full_result = (
            "=== PHASE 1: PRODUCT FEATURES ===\n"
            f"{features_output}\n\n"
            "=== PHASE 2: USER STORIES ===\n"
            f"{stories_output}\n\n"
            "=== PHASE 3: DEVELOPMENT TASKS ===\n"
            f"{tasks_output}"
        )
        plan.set_final_output(full_result)

        logger.info("═══ ProductWorkflowOrchestrator finished ═══\n")
        logger.info("Summary: 3/3 phases completed\n")

        return plan

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_agent(role_key: str) -> KnowledgeAugmentedPromptAgent:
        """Create a KnowledgeAugmentedPromptAgent from the global config."""
        cfg = AGENT_CONFIG[role_key]
        return KnowledgeAugmentedPromptAgent(
            name=cfg["name"],
            persona=cfg["persona"],
            knowledge=cfg["knowledge"],
        )

    def _run_phase(
        self,
        plan: PlanState,
        step_number: int,
        agent: KnowledgeAugmentedPromptAgent,
        evaluator: EvaluationAgent,
        prompt: str,
        criteria: str,
        artifact_key: str,
    ) -> str:
        """Execute one phase: mark in-progress, run evaluator, store result."""
        plan.mark_in_progress(step_number)

        result = evaluator.respond(
            worker_agent=agent,
            user_prompt=prompt,
            criteria=criteria,
        )
        output = result["final_response"]
        iterations = result["iterations"]

        plan.mark_completed(
            step_number, self.PHASE_NAMES[step_number][0], output
        )
        plan.add_artifact(artifact_key, output)

        logger.info(
            f"  → Phase {step_number} completed in {iterations} iteration(s)"
        )
        return output
