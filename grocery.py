from typing import Dict, List, Optional, Union
from pydantic import BaseModel
from models import Recipe, GroceryItem
import re
from collections import defaultdict
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import AIMessage
from langgraph.types import Command
from utils import get_llm
from state import NutritionistState
from langchain.tools import tool

    
class GroceryList(BaseModel):
    """Represents a complete grocery list"""
    items: List[GroceryItem]
    recipe_source: Optional[str] = None  # Name of the recipe this list is for
    servings: int = 1
    
    def add_item(self, item: GroceryItem) -> None:
        """Add a single item to the grocery list"""
        self.items.append(item)
    
    def remove_item(self, item_name: str) -> None:
        """Remove an item from the grocery list by name"""
        self.items = [item for item in self.items if item.name != item_name]
    
    def get_by_category(self) -> Dict[str, List[GroceryItem]]:
        """Group items by their category"""
        categorized = defaultdict(list)
        for item in self.items:
            categorized[item.category].append(item)
        return dict(categorized)
    
    def scale_quantities(self, servings: int) -> None:
        """Scale quantities based on desired number of servings"""
        factor = servings / self.servings
        for item in self.items:
            item.quantity *= factor
        self.servings = servings

@tool
def parse_ingredient(ingredient: str) -> GroceryItem:
    """
    Parse an ingredient string into a GroceryItem.
    Example: "2 cups milk" -> GroceryItem(name="milk", quantity=2, unit="cups")
    
    Args:
        ingredient: String containing ingredient information
        
    Returns:
        GroceryItem object with parsed information
    """
    # Common units of measurement
    units = {
        'cup': 'cups', 'cups': 'cups',
        'tbsp': 'tablespoons', 'tablespoon': 'tablespoons', 'tablespoons': 'tablespoons',
        'tsp': 'teaspoons', 'teaspoon': 'teaspoons', 'teaspoons': 'teaspoons',
        'oz': 'ounces', 'ounce': 'ounces', 'ounces': 'ounces',
        'lb': 'pounds', 'pound': 'pounds', 'pounds': 'pounds',
        'g': 'grams', 'gram': 'grams', 'grams': 'grams',
        'ml': 'milliliters', 'milliliter': 'milliliters', 'milliliters': 'milliliters'
    }
    
    # Regular expression to match quantity, unit, and ingredient name
    pattern = r'^(\d+(?:\.\d+)?)\s*([a-zA-Z]+)?\s+(.+)$'
    match = re.match(pattern, ingredient.strip())
    
    if match:
        quantity = float(match.group(1))
        unit = match.group(2).lower() if match.group(2) else ""
        name = match.group(3)
        
        # Standardize unit
        unit = units.get(unit, unit) if unit else ""
        
        # Determine category based on common ingredients (can be expanded)
        categories = {
            'produce': ['apple', 'banana', 'lettuce', 'tomato', 'onion', 'garlic', 'vegetable', 'fruit'],
            'dairy': ['milk', 'cheese', 'yogurt', 'cream', 'butter'],
            'meat': ['chicken', 'beef', 'pork', 'fish', 'meat'],
            'pantry': ['flour', 'sugar', 'salt', 'oil', 'spice', 'herb'],
            'grains': ['rice', 'pasta', 'bread', 'cereal']
        }
        
        category = "Other"
        for cat, keywords in categories.items():
            if any(keyword in name.lower() for keyword in keywords):
                category = cat.capitalize()
                break
                
        return GroceryItem(
            name=name,
            quantity=quantity,
            unit=unit,
            category=category
        )
    else:
        # If no quantity/unit found, treat as just the item name
        return GroceryItem(
            name=ingredient.strip(),
            quantity=1,
            unit="",
            category="Other"
        )

@tool
def create_grocery_list(recipe: Recipe, servings: Optional[int] = None) -> GroceryList:
    """
    Create a grocery list from a recipe.
    
    Args:
        recipe: Recipe object containing ingredients
        servings: Optional number of servings to scale the recipe to
    
    Returns:
        GroceryList object containing all needed ingredients
    """
    items = [parse_ingredient(ingredient) for ingredient in recipe.ingredients]
    grocery_list = GroceryList(
        items=items,
        recipe_source=recipe.name,
        servings=recipe.servings
    )
    
    # Scale quantities if different number of servings requested
    if servings and servings != recipe.servings:
        grocery_list.scale_quantities(servings)
    
    return grocery_list

@tool
def merge_grocery_lists(lists: List[GroceryList]) -> GroceryList:
    """
    Merge multiple grocery lists into one, combining similar items.
    
    Args:
        lists: List of GroceryList objects to merge
    
    Returns:
        Combined GroceryList object
    """
    merged_items: Dict[str, GroceryItem] = {}
    
    for grocery_list in lists:
        for item in grocery_list.items:
            key = f"{item.name}_{item.unit}"
            if key in merged_items:
                # If item exists, add quantities
                merged_items[key].quantity += item.quantity
            else:
                # If new item, add to merged items
                merged_items[key] = GroceryItem(
                    name=item.name,
                    quantity=item.quantity,
                    unit=item.unit,
                    category=item.category
                )
    
    return GroceryList(
        items=list(merged_items.values()),
        recipe_source="Multiple Recipes"
    )

GROCERY_SYSTEM_PROMPT = """You are an expert at creating and managing grocery lists based on recipes.
Your task is to:
1. Parse recipe ingredients into structured grocery items
2. Categorize ingredients appropriately (Produce, Dairy, Meat, Pantry, etc.)
3. Handle unit conversions and quantity scaling
4. Combine items when merging multiple lists
5. Ensure all necessary ingredients are included

For each recipe:
- Extract all ingredients and their quantities
- Standardize units where possible
- Categorize items for shopping convenience
- Account for serving size adjustments
- Combine similar items when appropriate

Always maintain organization by:
- Grouping items by category
- Standardizing units
- Combining duplicate items
- Maintaining clear quantities
"""

def create_grocery_agent():
    """Creates an agent for managing grocery lists."""
    grocery_prompt = ChatPromptTemplate(
        [
            ("system", GROCERY_SYSTEM_PROMPT),
            ("placeholder", "{messages}"),
        ]
    )
    
    grocery_agent = create_react_agent(
        model=get_llm(),
        tools=[parse_ingredient, create_grocery_list, merge_grocery_lists],
        prompt=grocery_prompt,
        response_format=GroceryList
    )
    
    return grocery_agent

def grocery_node(state: NutritionistState) -> Command:
    """Node function for generating grocery lists with state management."""
    grocery_agent = create_grocery_agent()
    result = grocery_agent.invoke({
        "messages": state["messages"],
        "recipe": state["recipe"]
    })
    
    grocery_list = result["output"]
    metadata = state["metadata"].copy()
    metadata["grocery_list"] = grocery_list
    
    return Command(
        update={
            "messages": [
                AIMessage(
                    content=str(grocery_list),
                    name="grocery_node"
                )
            ],
            "metadata": metadata
        }
    ) 