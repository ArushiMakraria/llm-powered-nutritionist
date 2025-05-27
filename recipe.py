from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent
from langchain_community.tools import DuckDuckGoSearchRun
import pandas as pd
import config
from utils import get_llm
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import Command
from models import Recipe
from state import NutritionistState
from typing import Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI

ENHANCED_RECIPE_PROMPT = """You are a culinary expert specializing in creating detailed, specific recipes that exactly match user requests.

Your task is to create recipes that precisely match what the user is asking for:

1. **EXACT RECIPE MATCHING**: 
   - If user asks for "Calcium-Rich Fig and Tahini Power Bowl" → Create a POWER BOWL (not a salad)
   - If user asks for "Palak Paneer" → Create the traditional Indian dish
   - If user asks for "Tahini Dressing" → Create a dressing/sauce, not a dish that uses tahini

2. **Recipe Format Guidelines**:
   - Power Bowl = Base (grains/greens) + Protein + Toppings + Dressing
   - Salad = Mixed greens + Vegetables + Dressing
   - Dressing = Sauce/condiment for other dishes

3. **Nutritional Accuracy**: Use the dataset below for precise nutritional calculations

Dataset Reference for nutritional values:
{df_str}

4. **Recipe Requirements**:
   - Match the EXACT dish type requested
   - Complete ingredient list with precise quantities
   - Step-by-step instructions
   - Accurate timing estimates
   - Detailed nutritional information per serving (calories, protein, fat, carbs, fiber, key nutrients)
   - Consider all dietary restrictions and excluded ingredients

5. **Search Usage**: Use the search tool to find authentic recipes for specific dishes you're not familiar with.

IMPORTANT: Always create the exact type of dish requested. Don't substitute a salad for a power bowl, or a dish for a dressing.
"""

def create_recipe_agent():
    """Create a recipe agent with enhanced prompt for better specificity handling"""
    df = pd.read_csv("clean-food.csv")
    recipe_prompt = ChatPromptTemplate(
        [
            ("system", ENHANCED_RECIPE_PROMPT),
            ("placeholder", "{messages}"),
        ],
        partial_variables={"df_str": df.to_markdown(index=True)}
    )
    recipe_agent = create_react_agent(
        model=get_llm(),
        tools=[DuckDuckGoSearchRun()],
        prompt=recipe_prompt,
        response_format=Recipe
    )
    return recipe_agent

def recipe_node(state: NutritionistState) -> Dict[str, Any]:
    """
    Enhanced recipe generation node that handles specific recipes and ingredient constraints.
    """
    intent = state["intent"]
    recipe_agent = create_recipe_agent()
    
    # Build enhanced message with better context
    original_message = state["messages"][-1].content if state["messages"] else "Create a recipe"
    
    user_message_parts = [f"RECIPE REQUEST: {original_message}"]
    
    if intent:
        user_message_parts.append("\n🎯 ANALYSIS RESULTS:")
        
        # Handle recipe specificity with emphasis
        if intent.recipe_specificity == "specific_recipe":
            user_message_parts.append(f"⚠️ SPECIFIC RECIPE REQUESTED: Create the EXACT dish mentioned in the request.")
            user_message_parts.append(f"   - Do NOT substitute with similar dishes")
            user_message_parts.append(f"   - Match the exact format (bowl vs salad vs dressing)")
        elif intent.recipe_specificity == "general_dish":
            user_message_parts.append(f"📋 GENERAL DISH: Create a suitable recipe that meets the criteria.")
        
        # Handle excluded ingredients prominently
        if intent.excluded_ingredients:
            user_message_parts.append(f"🚫 EXCLUDED INGREDIENTS: {', '.join(intent.excluded_ingredients)}")
        
        if intent.specific_foods:
            user_message_parts.append(f"✅ INCLUDE THESE FOODS: {', '.join(intent.specific_foods)}")
        
        if intent.dietary_restrictions:
            user_message_parts.append(f"🥗 DIETARY RESTRICTIONS: {', '.join(intent.dietary_restrictions)}")
        
        if intent.nutritional_requirements:
            user_message_parts.append(f"📊 NUTRITIONAL FOCUS: {intent.nutritional_requirements}")
        
        if intent.meal_type:
            user_message_parts.append(f"🍽️ MEAL TYPE: {', '.join(intent.meal_type)}")
    
    user_message_parts.append("""
📝 RECIPE REQUIREMENTS:
1. Create the EXACT dish type requested (don't substitute formats)
2. Include complete ingredients with precise measurements
3. Provide clear step-by-step instructions
4. Calculate detailed nutritional information per serving
5. Respect all dietary restrictions and excluded ingredients
6. Use search tool if you need authentic recipe details

🔍 If this is a specific dish you're not familiar with, please search for it first to ensure authenticity.
""")
    
    enhanced_user_message = "\n".join(user_message_parts)
    
    try:
        result = recipe_agent.invoke({
            "messages": [HumanMessage(content=enhanced_user_message)]
        })
        
        recipe_data = result['structured_response']
        
        return {
            "recipe": recipe_data,
            "messages": result.get("messages", [])
        }
    except Exception as e:
        print(f"Error in recipe_node: {e}")
        return {
            "messages": [AIMessage(content=f"Error generating recipe: {str(e)}", name="recipe_node")],
            "error": str(e)
        }