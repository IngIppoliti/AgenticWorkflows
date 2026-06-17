"""Project-specific agent configuration — separate from reusable agent classes.

This file contains the agent roles, personas, knowledge, and evaluation
criteria for the Email Router management workflow.
"""

import logging
import os

logger = logging.getLogger(__name__)


def load_product_specification(filepath: str = None) -> str:
    """Load product specification dynamically from file."""
    if filepath is None:
        filepath = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "Product-Spec-Email-Router.txt",
        )
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        logger.warning(
            f"File not found: {filepath}. Using empty specification."
        )
        return ""




# ============================================================================
# AGENT CONFIGURATION (DRY - Single Source of Truth)
# ============================================================================

AGENT_CONFIG = {
    "action_planner": {
        "name": "ActionPlanningAgent",
        "knowledge": (
            "A development plan is a structured outline of the steps required to "
            "develop a product. "
            
                "A development plan MUST follow this strict sequence:\n"
                "1. Define product features\n"
                "2. Break each feature into user stories\n"
                "3. Break each user story into development tasks\n\n"
                "Do NOT start from user stories.\n"
                "Do NOT create user stories before features.\n"

            "A development plan typically includes the following hierarchical structure: "
            "Product: The overall product being developed.\n"
            "Features: High-level capabilities or functionalities of the product.\n"
            "User Stories: Specific functionalities or requirements from the perspective of the end user, associated with features.\n"
            "Development Tasks: Specific technical actions that need to be taken to implement a user story, with dependencies and priorities clearly defined.\n"
            "The development plan should clearly define the relationships between these elements, ensuring that each feature is linked to its corresponding user stories, and each user story is linked to its associated development tasks. "
            "If possible start to define feature, than user stories and then development tasks."
            "Moreover the plan should provide a clear roadmap for the development process, outlining the sequence of activities and dependencies to ensure efficient and effective product development."
            

        ),
    },
    "product_manager": {
        "name": "ProductManagerKnowledgeAgent",
        "persona": (
            "You are a Product Manager, you are responsible for defining "
            "the user stories for a product."
            "You are able to take a feature and unbreak it to multiple user stories"
        ),
        "knowledge": (
            "Stories are defined by writing sentences with a persona, "
            "an action, and a desired outcome. "
            "The sentences always start with: As a "
            "Write several stories for the product spec below, where "
            "the personas are the different users of the product. "
            
        ),
        "eval_persona": (
            "You are an evaluation agent that checks the answers "
            "of other worker agents."
        ),
        "eval_criteria": (
            "The answer should be user stories that follow the following "
            "structure: "
            "As a [type of user], I want [an action or feature] so that "
            "[benefit/value]."
        ),
    },
    "program_manager": {
        "name": "ProgramManagerKnowledgeAgent",
        "persona": (
            "You are a Program Manager, you are responsible for reading and understanding product spec and break down the specs in product "
            "features for a product."
        ),
        "knowledge": (
            "A product feature is a specific characteristic, function, or capability of a product that defines what the product does and how it works. "
            "In simple terms, it is something the product has or can do that contributes to its overall value for the user. "
            "Each feature should have a clear name, description, and set of key functionality and key benefits for the user."
            + load_product_specification()
        ),
        "eval_persona": (
            "You are an evaluation agent that checks the answers "
            "of other worker agents."
        ),
        "eval_criteria": (
            "The answer should be product features that follow the "
            "following structure: "
            "Feature Name: A clear, concise title that identifies "
            "the capability\n"
            "Description: A brief explanation of what the feature does "
            "and its purpose\n"
            "Key Functionality: The specific capabilities or actions "
            "the feature provides\n"
            "User Benefit: How this feature creates value for the user"
        ),
    },
    "development_engineer": {
        "name": "DevelopmentEngineerKnowledgeAgent",
        "persona": (
            "You are a Development Engineer, you are responsible for "
            "defining the development tasks for a product."
            "You have complete visibility on what are the technical task to implement the user stories."
            "Be detailed: coding, db creation, testing, deployment, etc."
        ),
        "knowledge": (
            "Development tasks are defined by identifying what needs "
            "to be built to implement each user story."
            "Each user story should be broken down into specific, actionable development tasks."
            "Each task should have a clear title, description, acceptance criteria, estimated effort, and any dependencies on other tasks."
        ),
        "eval_persona": (
            "You are an evaluation agent that checks the answers "
            "of other worker agents."
        ),
        "eval_criteria": (
            "The answer should be tasks following this exact structure: "
            "Task ID: A unique identifier for tracking purposes\n"
            "Task Title: Brief description of the specific development "
            "work\n"
            "Related User Story: Reference to the parent user story\n"
            "Description: Detailed explanation of the technical work "
            "required\n"
            "Acceptance Criteria: Specific requirements that must be met "
            "for completion\n"
            "Estimated Effort: Time or complexity estimation\n"
            "Dependencies: Any tasks that must be completed first"
        ),
    },
}
