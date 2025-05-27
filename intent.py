from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import MessagesState
from langgraph.types import Command
from state import NutritionistState
from utils import get_llm
from langgraph.prebuilt import create_react_agent
from models import Intent
from typing import Dict, Any

INTENT_SYSTEM_PROMPT = """You are an expert nutritionist assistant that specializes in understanding user queries about nutrition, diet, and food recommendations.
Your task is to extract relevant entities and intents from user queries to help provide personalized nutrition advice.

CRITICAL: Pay close attention to time durations mentioned in the query:
- "3-day" or "3 day" → time_context: "daily_plan" (NOT weekly_plan)
- "week" or "7-day" → time_context: "weekly_plan"
- "meal prep" → time_context: "meal_prep"
- Single meal → time_context: "single_meal"

For each user query, you should extract:

1. **Primary Intent**: Classify the main purpose:
   - "single_recipe": User wants one specific recipe
   - "diet_plan": User wants multiple meals/recipes for meal planning
   - "nutritional_info": User wants nutritional information about foods (not recipes)
   - "ingredient_substitution": User wants alternatives for ingredients
   - "meal_prep": User wants meal prep guidance

2. **Recipe Specificity**: How specific is the recipe request:
   - "specific_recipe": User mentions a specific dish name (e.g., "palak paneer", "tahini dressing")
   - "general_dish": User describes a type of dish (e.g., "high-protein breakfast")
   - "food_category": User asks about food categories (e.g., "calcium-rich foods")

3. **Time Context**: Temporal scope (BE PRECISE):
   - "single_meal": One meal/recipe
   - "daily_plan": 1-3 days of meals
   - "weekly_plan": 4-7 days or "week"
   - "meal_prep": Batch cooking focus
   - "none": No specific time context

4. **Ingredient Constraints**: 
   - specific_foods: Ingredients user wants to include
   - excluded_ingredients: Ingredients user doesn't have or wants to avoid (look for "don't have", "without", "no", etc.)

Examples:
- "Make me a 3-day vegetarian plan" → primary_intent: "diet_plan", time_context: "daily_plan" (NOT weekly_plan!)
- "I need a week of meal prep" → primary_intent: "diet_plan", time_context: "weekly_plan"
- "How to make palak paneer" → primary_intent: "single_recipe", recipe_specificity: "specific_recipe"
- "Foods high in calcium but no dairy" → primary_intent: "nutritional_info", excluded_ingredients: ["dairy"]

Response Schema (use exactly these field names):
- primary_intent: "single_recipe" | "diet_plan" | "nutritional_info" | "ingredient_substitution" | "meal_prep"
- meal_type: ["breakfast", "lunch", "dinner", "snack"] (can be multiple for diet plans)
- dietary_restrictions: [str] (vegetarian, gluten-free, dairy-free, etc.)
- nutritional_requirements: str ("high protein", "low calorie", etc.)
- health_goals: [str] (weight loss, muscle gain, etc.)
- specific_foods: [str] (foods user wants to include)
- excluded_ingredients: [str] (foods user doesn't have or wants to avoid)
- time_context: "single_meal" | "daily_plan" | "weekly_plan" | "meal_prep" | "none"
- recipe_specificity: "specific_recipe" | "general_dish" | "food_category"

IMPORTANT: For "3-day" requests, ALWAYS use time_context: "daily_plan", NOT "weekly_plan"!
"""

def create_intent_agent():
    """Creates an agent for extracting intents and entities from user queries."""
    intent_prompt = ChatPromptTemplate(
        [
            ("system", INTENT_SYSTEM_PROMPT),
            ("placeholder", "{messages}"),
        ]
    )
    
    intent_agent = create_react_agent(
        model=get_llm(),
        tools=[],  # No tools needed for intent extraction
        prompt=intent_prompt,
        response_format=Intent,   
        name="intent_agent"
    )
    
    return intent_agent

def intent_node(state: NutritionistState) -> Dict[str, Any]:
    """Node function for processing user queries through the intent agent."""
    print(f"🔍 DEBUG: Starting intent_node...")
    try:
        intent_agent = create_intent_agent()
        print(f"🔍 DEBUG: Intent agent created successfully")
        
        result = intent_agent.invoke({"messages": state["messages"]})
        print(f"🔍 DEBUG: Intent agent invoked successfully")
        
        # Extract the structured response and return state update
        intent_data = result['structured_response']
        print(f"🔍 DEBUG: Intent data extracted: {intent_data}")
        
        return {
            "intent": intent_data,
            "messages": result.get("messages", [])
        }
    except Exception as e:
        print(f"❌ ERROR in intent_node: {e}")
        print(f"🔍 DEBUG: Exception type: {type(e)}")
        # Return a default intent on error
        from models import Intent
        default_intent = Intent(
            primary_intent="single_recipe",
            meal_type=["breakfast"],
            dietary_restrictions=[],
            nutritional_requirements="",
            health_goals=[],
            specific_foods=[],
            excluded_ingredients=[],
            time_context="single_meal",
            recipe_specificity="general_dish"
        )
        return {
            "intent": default_intent,
            "messages": [AIMessage(content=f"Intent analysis error: {str(e)}", name="intent_node")]
        }