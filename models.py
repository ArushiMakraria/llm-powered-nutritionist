from typing import Dict, List, Literal, Optional, Tuple, Union
from pydantic import BaseModel, Field

class ClinicalGuardrail(BaseModel):
    is_clinical: bool
    confidence: float
    explanation: str

class Intent(BaseModel):
    primary_intent: Literal["single_recipe", "diet_plan", "nutritional_info", "ingredient_substitution", "meal_prep"]
    meal_type: List[Literal["breakfast", "lunch", "dinner", "snack"]]
    dietary_restrictions: List[str]
    nutritional_requirements: str
    health_goals: List[str]
    specific_foods: List[str]  # Foods user wants to include
    excluded_ingredients: List[str]  # Foods/ingredients user doesn't have or wants to avoid
    time_context: Literal["single_meal", "daily_plan", "weekly_plan", "meal_prep", "none"]
    recipe_specificity: Literal["specific_recipe", "general_dish", "food_category"]  # New field

class Recipe(BaseModel):
    name: str
    ingredients: List[str]
    instructions: List[str]
    prep_time: str
    cook_time: str
    total_time: str
    servings: int
    nutritional_info: str

class MealPlanDay(BaseModel):
    """Represents one day of meals"""
    day: str  # "Day 1", "Day 2", etc.
    breakfast: List[Recipe]
    lunch: List[Recipe] 
    dinner: List[Recipe]
    snack: Optional[List[Recipe]] = []

class DietPlan(BaseModel):
    """Represents a multi-day diet plan with simplified structure"""
    plan_name: str
    duration: str  # "3 days", "1 week", etc.
    daily_plans: List[MealPlanDay] = Field(description="List of daily meal plans")
    total_nutritional_info: str
    shopping_list: List[str]

class NutritionalInfo(BaseModel):
    """Represents nutritional information response"""
    query_summary: str
    food_recommendations: List[str]
    nutritional_breakdown: Dict[str, str]  # {"calcium": "1000mg daily recommended", ...}
    food_sources: Dict[str, List[str]]  # {"calcium": ["dairy", "leafy greens", ...]}
    additional_notes: str

class GroceryItem(BaseModel):
    """Represents a single grocery item with quantity and unit"""
    name: str
    quantity: float
    unit: str
    category: str = "Other"  # e.g., Produce, Dairy, Meat, Pantry, etc.
    optional: bool = False

class Visualization(BaseModel):
    plot_path: str
