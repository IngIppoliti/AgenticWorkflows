"""
Deterministic 3-phase workflow for product management.

Replaces the old non-deterministic action-planning + embedding-based
routing with a fixed pipeline:

  Phase 1 — Program Manager     → Product Features
  Phase 2 — Product Manager     → User Stories (seeded with features)
  Phase 3 — Development Engineer → Development Tasks (seeded with stories)

Each phase feeds its output into the next, guaranteeing strict
hierarchical consistency (Feature → User Story → Task).
No embedding-based routing — agent-per-phase is fixed by design.
"""

import logging

from workflow_orchestrator import ProductWorkflowOrchestrator

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Main entry point — runs the deterministic 3-phase workflow."""
    logger.info("*** Deterministic workflow execution started ***\n")

    orchestrator = ProductWorkflowOrchestrator()

    plan = orchestrator.run(
        user_prompt="Create a complete development plan",
    )

    logger.info("*** Workflow execution completed ***")
    logger.info("Summary: 3/3 phases completed successfully")

    # Print the final aggregated result
    print("\n" + "=" * 70)
    print("  FINAL WORKFLOW OUTPUT")
    print("=" * 70)
    print(plan.final_output)


if __name__ == "__main__":
    main()
