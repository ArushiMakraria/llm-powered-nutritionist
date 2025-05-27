from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent
from langchain_community.tools import DuckDuckGoSearchRun
import pandas as pd
import config
from utils import get_llm
from langchain_core.messages import AIMessage, HumanMessage
from models import DietPlan, MealPlanDay, Recipe
from state import NutritionistState
from typing import Dict, Any

DIET_PLAN_PROMPT = """You are a nutrition expert specializing in creating comprehensive diet plans and meal prep guidance.

Your task is to create detailed diet plans that include multiple recipes organized by day and meal type.

Dataset Reference for nutritional values:
{df_str}

⚠️ CRITICAL DURATION INSTRUCTIONS:
- If user asks for "3-day" or "3 day": Create EXACTLY 3 days (Day 1, Day 2, Day 3)
- If user asks for "week" or "7-day": Create EXACTLY 7 days
- If user asks for "5-day": Create EXACTLY 5 days
- DO NOT create more or fewer days than requested!

You must return a structured DietPlan with the following exact format:

For a 3-day plan:
{{
  "plan_name": "3-Day Vegetarian Low-Calorie Plan",
  "duration": "3 days",
  "daily_plans": [
    {{
      "day": "Day 1",
      "breakfast": [
        {{
          "name": "Recipe Name",
          "ingredients": ["ingredient 1", "ingredient 2"],
          "instructions": ["step 1", "step 2"],
          "prep_time": "10 minutes",
          "cook_time": "5 minutes", 
          "total_time": "15 minutes",
          "servings": 1,
          "nutritional_info": "calories: 300, protein: 15g, carbs: 30g, fat: 10g"
        }}
      ],
      "lunch": [...],
      "dinner": [...],
      "snack": [...]
    }},
    {{
      "day": "Day 2",
      "breakfast": [...],
      "lunch": [...],
      "dinner": [...],
      "snack": [...]
    }},
    {{
      "day": "Day 3",
      "breakfast": [...],
      "lunch": [...],
      "dinner": [...],
      "snack": [...]
    }}
  ],
  "total_nutritional_info": "Daily average: 1450 calories, 80g protein, 180g carbs, 45g fat",
  "shopping_list": ["ingredient 1", "ingredient 2", "ingredient 3"]
}}

Guidelines:
- Create EXACTLY the number of days requested (NO MORE, NO LESS)
- Each day should have breakfast, lunch, dinner, and optionally snacks
- Each meal should have 1 recipe
- Keep total daily calories within the specified limit
- Ensure variety across days
- Consider all dietary restrictions and excluded ingredients
- Provide complete recipes with ingredients, instructions, and nutritional info
- Calculate realistic nutritional information per serving
- Generate a comprehensive shopping list for all recipes

REMEMBER: If user asks for 3 days, create exactly 3 daily_plans, not 7!
"""

def create_diet_plan_agent():
    """Create a diet plan agent"""
    print(f"🔍 DEBUG: Creating diet plan agent...")
    df = pd.read_csv("clean-food.csv")
    
    diet_plan_prompt = ChatPromptTemplate(
        [
            ("system", DIET_PLAN_PROMPT),
            ("placeholder", "{messages}"),
        ],
        partial_variables={"df_str": df.to_markdown(index=True)}
    )
    
    diet_plan_agent = create_react_agent(
        model=get_llm(),
        tools=[DuckDuckGoSearchRun()],
        prompt=diet_plan_prompt,
        response_format=DietPlan
    )
    
    print(f"✅ DEBUG: Diet plan agent created successfully")
    return diet_plan_agent

def diet_plan_node(state: NutritionistState) -> Dict[str, Any]:
    """
    Diet plan node that creates comprehensive meal plans with multiple recipes.
    """
    print(f"🔍 DEBUG: Starting diet_plan_node...")
    intent = state["intent"]
    print(f"🔍 DEBUG: Intent received: {intent}")
    
    diet_plan_agent = create_diet_plan_agent()
    
    # Build comprehensive message
    original_message = state["messages"][-1].content if state["messages"] else ""
    print(f"🔍 DEBUG: Original message: {original_message}")
    
    # Detect the exact number of days requested
    days_requested = None
    if "3-day" in original_message.lower() or "3 day" in original_message.lower():
        days_requested = 3
    elif "5-day" in original_message.lower() or "5 day" in original_message.lower():
        days_requested = 5
    elif "week" in original_message.lower() or "7-day" in original_message.lower() or "7 day" in original_message.lower():
        days_requested = 7
    
    enhanced_message_parts = [f"DIET PLAN REQUEST: {original_message}"]
    
    if days_requested:
        enhanced_message_parts.append(f"\n⚠️ CRITICAL: Create EXACTLY {days_requested} days of meal plans!")
        enhanced_message_parts.append(f"The user specifically asked for {days_requested} days, so create exactly {days_requested} daily_plans.")
    
    if intent:
        enhanced_message_parts.append("\n🎯 PLAN REQUIREMENTS:")
        
        if intent.meal_type:
            enhanced_message_parts.append(f"🍽️ Meal types needed: {', '.join(intent.meal_type)}")
        else:
            enhanced_message_parts.append("🍽️ Meal types needed: breakfast, lunch, dinner, snack")
        
        if intent.dietary_restrictions:
            enhanced_message_parts.append(f"🥗 Dietary restrictions: {', '.join(intent.dietary_restrictions)}")
        
        if intent.excluded_ingredients:
            enhanced_message_parts.append(f"🚫 Excluded ingredients: {', '.join(intent.excluded_ingredients)}")
        
        if intent.nutritional_requirements:
            enhanced_message_parts.append(f"📊 Nutritional focus: {intent.nutritional_requirements}")
        
        if intent.specific_foods:
            enhanced_message_parts.append(f"✅ Include these foods: {', '.join(intent.specific_foods)}")
    
    enhanced_message_parts.append(f"""
📝 DIET PLAN REQUIREMENTS:
1. Create EXACTLY {days_requested if days_requested else 'the requested number of'} days
2. Each day must have breakfast, lunch, dinner, and snack arrays
3. Each recipe must include complete ingredients, instructions, timing, and nutrition
4. Ensure variety across days while meeting calorie and dietary requirements
5. Provide a comprehensive shopping list
6. Calculate total nutritional summary

⚠️ FINAL REMINDER: If user asked for 3 days, create exactly 3 daily_plans. Do not create 7 days!

CRITICAL: Return the exact JSON structure specified in the system prompt.
""")
    
    enhanced_message = "\n".join(enhanced_message_parts)
    print(f"🔍 DEBUG: Enhanced message length: {len(enhanced_message)}")
    
    try:
        print(f"🔍 DEBUG: Invoking diet plan agent...")
        result = diet_plan_agent.invoke({
            "messages": [HumanMessage(content=enhanced_message)]
        })
        
        print(f"🔍 DEBUG: Agent result received")
        diet_plan_data = result['structured_response']
        print(f"🔍 DEBUG: Diet plan has {len(diet_plan_data.daily_plans)} days")
        
        return {
            "diet_plan": diet_plan_data,
            "messages": result.get("messages", [])
        }
    except Exception as e:
        print(f"❌ ERROR in diet_plan_node: {e}")
        print(f"🔍 DEBUG: Exception type: {type(e)}")
        # Return a fallback response
        return {
            "messages": [AIMessage(content=f"I encountered an issue creating your diet plan: {str(e)}. Let me try creating a single recipe instead.", name="diet_plan_node")],
            "error": str(e)
        }
