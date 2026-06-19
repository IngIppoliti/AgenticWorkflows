import logging
from typing import Callable, Tuple

from config import AGENT_CONFIG

from base_agents import (
    ActionPlanningAgent,
    EvaluationAgent,
    KnowledgeAugmentedPromptAgent,
    RoutingAgent,
)
from workflow_state import PlanState

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# AGENT FACTORY METHODS
# ============================================================================

def create_role_support_function(role_key: str, max_interactions: int = 3) -> Callable:
    """Build a worker + evaluator pair for a configured role and return its callable."""
    cfg = AGENT_CONFIG[role_key]
    worker = KnowledgeAugmentedPromptAgent(cfg["name"], cfg["persona"], cfg["knowledge"])
    evaluator = EvaluationAgent(
        persona=cfg["eval_persona"],
        evaluation_criteria=cfg["eval_criteria"],
        agent_to_evaluate=worker,
        max_interactions=max_interactions,
    )

    def support_function(query: str) -> str:
        return evaluator.execute(user_prompt=query)["final_response"]

    return support_function


def build_routing_agent() -> RoutingAgent:
    """Instantiate the routing agent and its role-specific workers."""
    return RoutingAgent(
        agents=[
            {
                "name": "Product Manager",
                "description": "Responsible for defining user stories for a product.",
                "func": create_role_support_function("product_manager"),
            },
            {
                "name": "Program Manager",
                "description": "Responsible for defining features for a product.",
                "func": create_role_support_function("program_manager"),
            },
            {
                "name": "Development Engineer",
                "description": "Responsible for defining development tasks for a product.",
                "func": create_role_support_function("development_engineer"),
            },
        ]
    )


# ============================================================================
# WORKFLOW EXECUTION
# ============================================================================


def execute_workflow_step(
    step: str, idx: int, total: int, routing_agent: RoutingAgent
) -> Tuple[bool, str, str]:
    """Execute a single workflow step with error handling.
    Returns (success, output, agent_name)."""
    try:
        logger.info(f"[Step {idx}/{total}] Executing: {step}")
        output, agent_name = routing_agent.route(step)
        logger.info(f"[Step {idx}/{total}] Completed by {agent_name}")
        return True, output, agent_name
    except Exception as exc:
        logger.error(f"[Step {idx}/{total}] Error: {str(exc)}")
        return False, str(exc), ""


def main() -> None:
    """Main entry point for the agentic workflow."""
    logger.info("*** Workflow execution started ***\n")

    # Workflow Configuration
    workflow_prompt = "Create a complete development plan"

    logger.info(f"Workflow prompt: {workflow_prompt}\n")

    # Initialize agents
    action_planning_agent = ActionPlanningAgent(
        AGENT_CONFIG["action_planner"]["name"],
        AGENT_CONFIG["action_planner"]["knowledge"],
    )
    routing_agent = build_routing_agent()

    # Extract workflow steps
    logger.info("Extracting workflow steps from action planning agent...")
    steps = action_planning_agent.execute(workflow_prompt)

    if not steps:
        logger.error("No workflow steps extracted. Exiting.")
        return

    logger.info(f"Found {len(steps)} workflow steps\n")

    plan = PlanState(prompt=workflow_prompt)
    plan.add_steps(steps)
    successful = 0

    for idx, step in enumerate(steps, 1):
        plan.mark_in_progress(idx)
        success, result, agent = execute_workflow_step(step, idx, len(steps), routing_agent)
        if success:
            plan.mark_completed(idx, agent, result)
            successful += 1
            logger.info(f"Result: {result}\n")
        else:
            plan.mark_failed(idx, result)
            logger.warning(f"Step {idx} failed, continuing...\n")

    full_result = "\n\n".join(
        f"=== Step {entry.step_number} ({entry.assigned_worker or '?'}) ===\n"
        f"{entry.final_output or entry.error_message or '(no output)'}"
        for entry in plan.entries
    )
    plan.set_final_output(full_result)

    logger.info("*** Workflow execution completed ***")
    logger.info(f"Summary: {successful}/{len(steps)} steps completed successfully")


if __name__ == "__main__":
    main()
