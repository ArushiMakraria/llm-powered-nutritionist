from typing import Dict, List, Optional, Any
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from models import Intent, Recipe, DietPlan, NutritionalInfo, Visualization, ClinicalGuardrail

class NutritionistState(TypedDict):
    """Enhanced state management for the nutritionist workflow"""
    messages: List[BaseMessage]
    intent: Optional[Intent]
    recipe: Optional[Recipe]
    diet_plan: Optional[DietPlan]
    nutritional_info: Optional[NutritionalInfo]
    visualization: Optional[str]
    clinical_check: Optional[ClinicalGuardrail]
    metadata: Dict[str, Any]