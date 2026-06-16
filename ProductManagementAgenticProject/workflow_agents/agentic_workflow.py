import logging
from typing import Callable, List, Tuple

from workflow_agents.base_agents import (
    ActionPlanningAgent,
    AGENT_CONFIG,
    EvaluationAgent,
    KnowledgeAugmentedPromptAgent,
    RoutingAgent,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# AGENT FACTORY METHODS
# ============================================================================

def create_knowledge_augmented_agent(
    agent_name: str, persona: str, knowledge: str
) -> KnowledgeAugmentedPromptAgent:
    """Factory method to create a Knowledge Augmented Prompt Agent."""
    return KnowledgeAugmentedPromptAgent(agent_name, persona, knowledge)


def create_evaluation_agent(
    max_interactions: int = 5,
) -> EvaluationAgent:
    """Factory method to create an Evaluation Agent."""
    return EvaluationAgent(max_interactions=max_interactions)


def create_support_function(
    knowledge_agent: KnowledgeAugmentedPromptAgent,
    evaluation_agent: EvaluationAgent,
    criteria: str = None,
) -> Callable:
    """Factory method to create a support function (higher-order function)."""
    def support_function(query: str) -> str:
        """Execute knowledge agent and evaluate response."""
        result = evaluation_agent.respond(
            worker_agent=knowledge_agent,
            user_prompt=query,
            criteria=criteria,
        )
        return result["final_response"]
    return support_function


# ============================================================================
# AGENT INITIALIZATION
# ============================================================================

# Action Planning Agent
action_planning_agent = ActionPlanningAgent(
    AGENT_CONFIG["action_planner"]["name"],
    AGENT_CONFIG["action_planner"]["knowledge"],
)

# Product Manager Agents
product_manager_knowledge_agent = create_knowledge_augmented_agent(
    AGENT_CONFIG["product_manager"]["name"],
    AGENT_CONFIG["product_manager"]["persona"],
    AGENT_CONFIG["product_manager"]["knowledge"],
)
product_manager_evaluation_agent = create_evaluation_agent(max_interactions=1)

product_manager_support_function = create_support_function(
    product_manager_knowledge_agent,
    product_manager_evaluation_agent,
    criteria=AGENT_CONFIG["product_manager"]["eval_criteria"],
)

# Program Manager Agents
program_manager_knowledge_agent = create_knowledge_augmented_agent(
    AGENT_CONFIG["program_manager"]["name"],
    AGENT_CONFIG["program_manager"]["persona"],
    AGENT_CONFIG["program_manager"]["knowledge"],
)
program_manager_evaluation_agent = create_evaluation_agent(max_interactions=1)
program_manager_support_function = create_support_function(
    program_manager_knowledge_agent,
    program_manager_evaluation_agent,
    criteria=AGENT_CONFIG["program_manager"]["eval_criteria"],
)

# Development Engineer Agents
development_engineer_knowledge_agent = create_knowledge_augmented_agent(
    AGENT_CONFIG["development_engineer"]["name"],
    AGENT_CONFIG["development_engineer"]["persona"],
    AGENT_CONFIG["development_engineer"]["knowledge"],
)
development_engineer_evaluation_agent = create_evaluation_agent(max_interactions=1)
development_engineer_support_function = create_support_function(
    development_engineer_knowledge_agent,
    development_engineer_evaluation_agent,
    criteria=AGENT_CONFIG["development_engineer"]["eval_criteria"],
)

# ============================================================================
# ROUTING AGENT
# ============================================================================

routing_agent = RoutingAgent(
    agents=[
        {
            "name": "Product Manager",
            "description": "Responsible for defining user stories for a product.",
            "func": product_manager_support_function,  # ✅ FIXED: Corrected function reference
        },
        {
            "name": "Program Manager",
            "description": "Responsible for defining features for a product.",
            "func": program_manager_support_function,  # ✅ FIXED: Corrected function reference
        },
        {
            "name": "Development Engineer",
            "description": "Responsible for defining development tasks for a product.",
            "func": development_engineer_support_function,  # ✅ FIXED: Corrected function reference
        },
    ]
)


# ============================================================================
# WORKFLOW EXECUTION
# ============================================================================


def execute_workflow_step(step: str, idx: int, total: int) -> Tuple[bool, str]:
    """Execute a single workflow step with error handling."""
    try:
        logger.info(f"[Step {idx}/{total}] Executing: {step}")
        routed_response = routing_agent.route(step)
        logger.info(f"[Step {idx}/{total}] Completed successfully")
        return True, routed_response
    except Exception as e:
        logger.error(f"[Step {idx}/{total}] Error: {str(e)}")
        return False, str(e)


def main() -> None:
    """Main entry point for the agentic workflow."""
    logger.info("*** Workflow execution started ***\n")
    
    # Workflow Configuration
    workflow_prompt = "What would the development tasks for this product be?"
    logger.info(f"Workflow prompt: {workflow_prompt}\n")
    
    # Extract workflow steps
    logger.info("Extracting workflow steps from action planning agent...")
    steps = action_planning_agent.respond(workflow_prompt)
    
    if not steps:
        logger.error("No workflow steps extracted. Exiting.")
        return
    
    logger.info(f"Found {len(steps)} workflow steps\n")
    

    completed_steps: List[str] = []
    successful_steps = 0
    context = ""
    
    for idx, step in enumerate(steps, 1):
     
        enhanced_step = f"""
        Current context:
        {context}
        New instruction:
        {step}
        """

        success, result = execute_workflow_step(enhanced_step, idx, len(steps))
        if success:
            completed_steps.append(result)
            context += f"\n\nStep {idx} output:\n{result}"
            successful_steps += 1
            logger.info(f"Result: {result}\n")
        else:
            logger.warning(f"Step failed, continuing to next step...\n")
    
    # Output Summary
    logger.info("*** Workflow execution completed ***")
    logger.info(f"Summary: {successful_steps}/{len(steps)} steps completed successfully")
    
    if completed_steps:
        logger.info(f"\nFinal Workflow Output:\n{completed_steps[-1]}")
    else:
        logger.error("Workflow failed - no completed steps")


if __name__ == "__main__":
    main()
