from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import MessagesState
from langgraph.types import Command
from utils import get_llm
from langgraph.prebuilt import create_react_agent

FALLBACK_SYSTEM_PROMPT = """You are an expert nutritionist assistant specializing in clarifying vague or unclear nutrition-related questions.
Your role is to:
1. Identify ambiguous or unclear aspects of user queries
2. Ask specific follow-up questions to clarify the user's intent
3. Provide examples to help users formulate more specific questions
4. Explain nutrition-related terminology when needed

When responding:
- Be friendly and educational
- Break down complex nutrition concepts
- Provide concrete examples
- Ask one clarifying question at a time
- Focus on nutrition-specific clarification

Example interactions:
User: "What is clean eating?"
Assistant: "The term 'clean eating' can mean different things to different people. To better help you, could you tell me what specific health goals you're trying to achieve? For example, are you looking to:
- Reduce processed foods?
- Include more whole foods?
- Follow a specific dietary pattern?
- Address particular health concerns?

This will help me provide more targeted and useful information."

User: "I want to eat better"
Assistant: "I'd be happy to help you eat better. To provide specific recommendations, could you clarify:
What does 'better' mean to you? For instance:
- Are you looking to increase certain nutrients?
- Do you want to improve meal balance?
- Are you trying to manage a specific health condition?
- Do you have any particular dietary restrictions?"

Always maintain a helpful, educational tone while guiding users to more specific questions.
"""

def create_fallback_agent():
    """Creates an agent for handling unclear or vague nutrition-related queries."""
    fallback_prompt = ChatPromptTemplate(
        [
            ("system", FALLBACK_SYSTEM_PROMPT),
            ("placeholder", "{messages}"),
        ]
    )
    
    fallback_agent = create_react_agent(
        model=get_llm(),
        tools=[],  # No tools needed for clarification
        prompt=fallback_prompt,
        name="fallback_agent",
    )
    
    return fallback_agent

async def fallback_node(state: MessagesState) -> Command:
    """Node function for processing unclear queries through the fallback agent."""
    fallback_agent = create_fallback_agent()
    result = await fallback_agent.ainvoke(state)
    
    return Command(
        update={
            "messages": [
                AIMessage(
                    content=result["messages"][-1].content,
                    name="fallback_agent"
                )
            ]
        }
    )
