# from intent import intent_node
# from recipe import recipe_node
# from state import NutritionistState
# from langgraph.types import Command
# from langchain_core.messages import HumanMessage
# from dotenv import load_dotenv
# load_dotenv()

# state = NutritionistState()
# state.add_message(HumanMessage(content="Hello, I want a nutritious meal plan for my 10 year old, it should be high in protein and low in sugar."))
# result = intent_node(state)
# print("intent", result)
# print("================================")
# state.update_intent(result['structured_response'])
# print("State after intent", state)
# print("================================")
# result = recipe_node(state)
# print("recipe", result['structured_response'])


from visualization import visual_node
from state import NutritionistState
from models import Recipe
state = NutritionistState()
# state.add_message(HumanMessage(content="Hello, I want a nutritious meal plan for my 10 year old, it should be high in protein and low in sugar."))
state.update_recipe(Recipe(
    name="Chicken Salad",
    ingredients=["chicken", "lettuce", "tomato", "cucumber"],
    instructions=["chop the chicken", "chop the lettuce", "chop the tomato", "chop the cucumber"],
    prep_time="10 minutes",
    cook_time="10 minutes",
    total_time="20 minutes",
    servings=2,
    nutritional_info='Calories: 693 kcal, Protein: 41.7 g, Fat: 34.1 g, Carbohydrates: 52.8 g, Fiber: 11.1 g, Vitamin C: 81.2 mg'))
result = visual_node(state)
print("visualization", result)