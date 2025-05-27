from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent
from langchain_community.tools import DuckDuckGoSearchRun
import pandas as pd
import config
from utils import get_llm
from langchain_core.messages import AIMessage, HumanMessage
from models import NutritionalInfo
from state import NutritionistState
from typing import Dict, Any

NUTRITIONAL_INFO_PROMPT = """You are a nutrition expert specializing in providing detailed nutritional information about foods and nutrients.

Your task is to provide comprehensive nutritional information based on user queries. You should:

1. **Identify the nutritional focus** (e.g., calcium, protein, vitamins)
2. **List specific food recommendations** that meet the criteria
3. **Provide quantitative nutritional data** when possible
4. **Consider dietary restrictions** mentioned by the user
5. **Avoid suggesting recipes** - focus on food lists and nutritional facts

Dataset Reference for nutritional values:
{df_str}

Guidelines:
- Prioritize the provided dataset for nutritional values
- Use the search tool if specific foods aren't in the dataset
- Provide specific food names, not recipes
- Include quantitative values (mg, g, % daily value) when available
- Consider excluded ingredients mentioned by the user
- Focus on education about nutrition rather than cooking

Response Format:
- query_summary: Brief summary of what the user is asking for
- food_recommendations: List of specific foods that meet the criteria
- nutritional_breakdown: Key nutritional facts with quantities
- food_sources: Categorized list of food sources
- additional_notes: Important nutritional context or tips
"""

def create_nutritional_info_agent():
    """Create a nutritional information agent"""
    df = pd.read_csv("clean-food.csv")
    
    nutritional_prompt = ChatPromptTemplate(
        [
            ("system", NUTRITIONAL_INFO_PROMPT),
            ("placeholder", "{messages}"),
        ],
        partial_variables={"df_str": df.to_markdown(index=True)}
    )
    
    nutritional_agent = create_react_agent(
        model=get_llm(),
        tools=[DuckDuckGoSearchRun()],
        prompt=nutritional_prompt,
        response_format=NutritionalInfo
    )
    
    return nutritional_agent

def nutritional_info_node(state: NutritionistState) -> Dict[str, Any]:
    """
    Nutritional information node that provides food recommendations and nutritional data.
    """
    intent = state["intent"]
    nutritional_agent = create_nutritional_info_agent()
    
    # Build message with intent context
    original_message = state["messages"][-1].content if state["messages"] else ""
    
    enhanced_message_parts = [f"Original query: {original_message}"]
    
    if intent:
        enhanced_message_parts.append("\nContext from intent analysis:")
        if intent.nutritional_requirements:
            enhanced_message_parts.append(f"- Nutritional focus: {intent.nutritional_requirements}")
        if intent.dietary_restrictions:
            enhanced_message_parts.append(f"- Dietary restrictions: {', '.join(intent.dietary_restrictions)}")
        if intent.excluded_ingredients:
            enhanced_message_parts.append(f"- Excluded ingredients: {', '.join(intent.excluded_ingredients)}")
        if intent.specific_foods:
            enhanced_message_parts.append(f"- Specific foods mentioned: {', '.join(intent.specific_foods)}")
    
    enhanced_message_parts.append("""
Please provide:
1. A list of specific foods (not recipes) that meet the criteria
2. Nutritional information with quantities where possible
3. Food sources organized by category
4. Important nutritional context
""")
    
    enhanced_message = "\n".join(enhanced_message_parts)
    
    try:
        result = nutritional_agent.invoke({
            "messages": [HumanMessage(content=enhanced_message)]
        })
        
        nutritional_data = result['structured_response']
        
        return {
            "nutritional_info": nutritional_data,
            "messages": result.get("messages", [])
        }
    except Exception as e:
        print(f"Error in nutritional_info_node: {e}")
        return {
            "messages": [AIMessage(content=f"Error getting nutritional information: {str(e)}", name="nutritional_info_node")],
            "error": str(e)
        }
