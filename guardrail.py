from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import MessagesState, END
from langgraph.types import Command
from utils import get_llm
from langgraph.prebuilt import create_react_agent
from models import ClinicalGuardrail
from state import NutritionistState
from typing import Dict, Any

CLINICAL_SYSTEM_PROMPT = """You are an expert at detecting clinical and diagnostic medical queries that should be redirected to healthcare professionals.

Your task is to analyze user queries and determine if they are asking for medical diagnosis, clinical advice, or health condition assessment.

Examples of clinical/diagnostic queries to REJECT:
- "How can I cure IBS?"
- "Do I have diabetes?"
- "What's wrong with my thyroid?"
- "Am I at risk for heart disease?"
- "Diagnose my symptoms"
- "Is this a sign of vitamin deficiency?"
- "How to treat my condition?"
- "What medication should I take?"

Examples of nutrition queries to ALLOW:
- "What foods are high in vitamin D?"
- "How much protein should I eat?"
- "Can you suggest a healthy breakfast?"
- "What's the nutritional value of quinoa?"
- "Give me a meal plan for weight loss"
- "What foods help with digestion?"

For each query, analyze and return:
- is_clinical: bool - Whether the query is asking for medical/clinical advice
- confidence: float - Confidence score between 0-1
- explanation: str - Brief explanation of the decision

If a query is clinical, respond with a message redirecting them to consult a healthcare professional.
"""

def create_clinical_guardrail_agent():
    """Creates an agent for detecting clinical/diagnostic queries."""
    clinical_prompt = ChatPromptTemplate(
        [
            ("system", CLINICAL_SYSTEM_PROMPT),
            ("placeholder", "{messages}"),
        ]
    )
    
    clinical_agent = create_react_agent(
        model=get_llm(),
        tools=[],  # No tools needed for clinical detection
        prompt=clinical_prompt,
        response_format=ClinicalGuardrail,
        name="clinical_guardrail"
    )
    
    return clinical_agent

def clinical_guardrail_node(state: NutritionistState) -> Dict[str, Any]:
    """Node function for clinical guardrail with state management."""
    try:
        # Ensure we have messages to process
        if not state["messages"] or not state["messages"][-1].content.strip():
            return {
                "messages": [AIMessage(content="⚠️ Please provide a valid question.", name="clinical_guardrail")],
                "blocked": "Empty or invalid query"
            }
        
        clinical_agent = create_clinical_guardrail_agent()
        result = clinical_agent.invoke({"messages": state["messages"]})
        
        clinical_check = result["structured_response"]
        
        if clinical_check.is_clinical:
            # Block the request and provide appropriate message
            return {
                "messages": [
                    AIMessage(
                        content=f"⚠️ I understand you're asking about a medical condition. {clinical_check.explanation}\n\n"
                                f"For medical advice, diagnosis, or treatment recommendations, please consult with a qualified healthcare professional. "
                                f"I'm here to help with nutrition and recipe recommendations instead!\n\n"
                                f"Is there anything nutrition-related I can help you with today?",
                        name="clinical_guardrail"
                    )
                ],
                "blocked": clinical_check.explanation,
                "clinical_check": clinical_check
            }
        
        # Allow the request to proceed
        return {
            "clinical_check": clinical_check
        }
        
    except Exception as e:
        print(f"Error in clinical_guardrail_node: {e}")
        return {
            "messages": [AIMessage(content="⚠️ Error processing your request. Please try again.", name="clinical_guardrail")],
            "blocked": f"Processing error: {str(e)}"
        }