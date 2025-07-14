"""
LangMem memory management tools for Agent Beeb - Personal Grocery Shopping Assistant.
These tools enable Agent Beeb to manage long-term memories about user preferences,
dietary restrictions, shopping habits, and meal planning preferences.
"""

import os
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

from langchain_core.tools import tool
from langmem import create_manage_memory_tool, create_search_memory_tool


class DietaryPreference(BaseModel):
    """Store user dietary preferences and restrictions."""
    preference_type: str  # e.g., "vegetarian", "vegan", "gluten-free", "keto", "paleo"
    severity: str  # e.g., "strict", "moderate", "flexible"
    context: Optional[str] = None


class ShoppingHabit(BaseModel):
    """Store user shopping habits and preferences."""
    habit_type: str  # e.g., "budget", "store_preference", "shopping_frequency"
    value: str
    importance: int = 1  # 1-5 scale
    context: Optional[str] = None


class FoodPreference(BaseModel):
    """Store user food likes, dislikes, and allergies."""
    food_item: str
    preference_type: str  # e.g., "likes", "dislikes", "allergic"
    intensity: int = 1  # 1-5 scale
    context: Optional[str] = None


class MealPlan(BaseModel):
    """Store meal planning preferences and past meal plans."""
    meal_type: str  # e.g., "breakfast", "lunch", "dinner", "snack"
    recipe_or_meal: str
    ingredients: List[str]
    rating: Optional[int] = None  # 1-5 scale
    context: Optional[str] = None


class ShoppingList(BaseModel):
    """Store shopping list items and patterns."""
    item: str
    category: str  # e.g., "produce", "dairy", "meat", "pantry"
    frequency: str  # e.g., "weekly", "monthly", "occasional"
    brand_preference: Optional[str] = None
    context: Optional[str] = None


class BudgetInfo(BaseModel):
    """Store budget-related information."""
    budget_type: str  # e.g., "weekly", "monthly", "per_meal"
    amount: str
    priority: str  # e.g., "high", "medium", "low"
    context: Optional[str] = None


def _format_product_results(products: List[Dict]) -> List[Dict[str, Any]]:
    """Format product search results for Agent Beeb."""
    formatted_products = []
    
    for product in products:
        # Extract information from smart_grocery_search result
        formatted_product = {
            "id": product.get("product_id"),
            "gtin": product.get("gtin"),
            "title": product.get("product_title"),  # Fixed: was "title", now "product_title"
            "brand": product.get("brand"),
            "best_price": product.get("price"),
            "best_store": product.get("store_name"),
            "search_type": product.get("search_type"),
            "relevance_score": product.get("relevance_score"),
            "price_rank": product.get("price_rank"),
            "suggestion": product.get("suggestion"),
            "description": f"Relevance: {product.get('relevance_score', 0):.2f} - {product.get('suggestion', '')}"
        }
        
        formatted_products.append(formatted_product)
    
    return formatted_products


@tool
def search_grocery_products(
    query: str,
    dietary_filters: Optional[List[str]] = None,
    category_filter: Optional[str] = None,
    max_price: Optional[float] = None,
    store_preference: Optional[str] = None,
    limit: int = 10
) -> List[Dict[str, Any]]:
    """Search for products in the grocery database with 40,000+ items from multiple stores.
    
    Args:
        query: Search query for products (name, brand, category, etc.)
        dietary_filters: List of dietary filters like 'vegetarian', 'vegan', 'gluten-free', 'organic'
        category_filter: Product category filter (e.g., 'dairy', 'meat', 'produce')
        max_price: Maximum price filter
        store_preference: Preferred store (e.g., 'Albert Heijn', 'Jumbo')
        limit: Maximum number of products to return (default: 10)
    
    Returns:
        List of product information with prices, nutrition, ingredients, and store availability
    """
    try:
        from supabase import create_client
        
        # Get Supabase credentials from environment
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        
        if not supabase_url or not supabase_key:
            return [{"error": "Supabase credentials not configured"}]
        
        supabase = create_client(supabase_url, supabase_key)
        
        # For simple, short queries, use faster basic search to avoid 122+ second delays
        if len(query.split()) <= 2 and not dietary_filters and not category_filter:
            # Use basic text search for simple queries like "eggs", "milk", etc.
            query_builder = supabase.table('products').select('*')
            query_builder = query_builder.ilike('product_name', f'%{query}%')
            
            if max_price:
                query_builder = query_builder.lte('price', max_price)
            if store_preference:
                query_builder = query_builder.eq('store_name', store_preference)
            
            # Order by price and limit results
            result = query_builder.order('price', desc=False).limit(limit).execute()
            
            # Format basic results
            if result.data:
                basic_results = []
                for item in result.data:
                    basic_results.append({
                        'product_id': item.get('id'),
                        'gtin': item.get('gtin'),
                        'product_title': item.get('product_name'),
                        'brand': item.get('brand'),
                        'price': item.get('price'),
                        'store_name': item.get('store_name'),
                        'search_type': 'basic_text_search',
                        'relevance_score': 1.0,
                        'price_rank': 1,
                        'suggestion': f'Found via basic search'
                    })
                return _format_product_results(basic_results)
        
        # Use the existing smart_grocery_search function for complex queries
        result = supabase.rpc('smart_grocery_search', {
            'user_query': query,
            'max_budget': max_price,
            'store_preference': store_preference,
            'result_limit': limit  # Add limit parameter to improve performance
        }).execute()
        
        if result.data:
            # Apply limit as backup in case database function doesn't support it
            limited_results = result.data[:limit] if len(result.data) > limit else result.data
            return _format_product_results(limited_results)
        else:
            return []
            
    except Exception as e:
        return [{"error": f"Product search failed: {str(e)}"}]


def search_products(
    query: str,
    dietary_filters: Optional[List[str]] = None,
    category_filter: Optional[str] = None,
    max_price: Optional[float] = None,
    store_preference: Optional[str] = None,
    limit: int = 10
) -> List[Dict[str, Any]]:
    """
    Search for products in the grocery database.
    
    Args:
        query: Search query for products
        dietary_filters: List of dietary preferences to filter by
        category_filter: Product category to filter by
        max_price: Maximum price filter
        store_preference: Preferred store name
        limit: Maximum number of results
    
    Returns:
        List of product information dictionaries
    """
    return search_grocery_products(
        query=query,
        dietary_filters=dietary_filters,
        category_filter=category_filter,
        max_price=max_price,
        store_preference=store_preference,
        limit=limit
    )


# Memory management tools for different namespaces
def create_memory_tools(namespace: tuple = ("grocery_memories",)):
    """
    Create LangMem memory management tools for Agent Beeb.
    
    Args:
        namespace: Memory namespace for organization
        
    Returns:
        List of memory management tools
    """
    
    # Tool to manage memories (create, update, delete)
    manage_memory = create_manage_memory_tool(namespace=namespace)
    
    # Tool to search existing memories
    search_memory = create_search_memory_tool(namespace=namespace)
    
    return [manage_memory, search_memory]


def create_user_grocery_tools(user_id: str):
    """
    Create user-specific grocery shopping memory tools.
    
    Args:
        user_id: User identifier for namespacing
        
    Returns:
        List of user-specific grocery memory tools
    """
    
    # User dietary preferences namespace
    dietary_namespace = ("users", user_id, "dietary_preferences")
    dietary_tools = create_memory_tools(dietary_namespace)
    
    # User shopping habits namespace
    shopping_namespace = ("users", user_id, "shopping_habits")
    shopping_tools = create_memory_tools(shopping_namespace)
    
    # User food preferences namespace
    food_namespace = ("users", user_id, "food_preferences")
    food_tools = create_memory_tools(food_namespace)
    
    # User meal plans namespace
    meal_namespace = ("users", user_id, "meal_plans")
    meal_tools = create_memory_tools(meal_namespace)
    
    # User shopping lists namespace
    list_namespace = ("users", user_id, "shopping_lists")
    list_tools = create_memory_tools(list_namespace)
    
    # User budget information namespace
    budget_namespace = ("users", user_id, "budget_info")
    budget_tools = create_memory_tools(budget_namespace)
    
    return (dietary_tools + shopping_tools + food_tools + 
            meal_tools + list_tools + budget_tools)


# Grocery shopping specific memory schemas
GROCERY_MEMORY_SCHEMAS = [
    DietaryPreference, 
    ShoppingHabit, 
    FoodPreference, 
    MealPlan, 
    ShoppingList, 
    BudgetInfo
]


# Instructions for grocery shopping memory extraction
GROCERY_MEMORY_INSTRUCTIONS = """
Extract important grocery shopping information from conversations including:

1. **Dietary Preferences**: vegetarian, vegan, gluten-free, keto, paleo, allergies, etc.
2. **Food Preferences**: likes, dislikes, favorite brands, specific products
3. **Shopping Habits**: preferred stores, budget constraints, shopping frequency
4. **Meal Planning**: favorite recipes, meal prep preferences, cooking skills
5. **Shopping Lists**: recurring items, seasonal preferences, quantity preferences
6. **Budget Information**: weekly/monthly budgets, cost-conscious choices

Focus on information that will help Agent Beeb provide personalized grocery shopping 
assistance and meal planning recommendations in future conversations.
""" 


@tool
def create_grocery_list(
    items: List[str],
    dietary_preferences: Optional[List[str]] = None,
    budget_limit: Optional[float] = None,
    store_preference: Optional[str] = None
) -> Dict[str, Any]:
    """Create a grocery list by searching for specific items with prices and store availability.
    
    Args:
        items: List of items to search for (e.g., ['milk', 'bread', 'chicken breast'])
        dietary_preferences: Dietary filters like 'organic', 'vegetarian', 'gluten-free'
        budget_limit: Maximum budget for the entire list
        store_preference: Preferred store name
    
    Returns:
        Dictionary with grocery list containing products, total cost, and store breakdown
    """
    try:
        grocery_list = {
            "items": [],
            "total_cost": 0.0,
            "stores": {},
            "budget_status": "within_budget",
            "suggestions": []
        }
        
        for item in items:
            # Search for each item
            products = search_grocery_products.invoke({
                "query": item,
                "dietary_filters": dietary_preferences,
                "store_preference": store_preference,
                "limit": 3  # Get top 3 options per item
            })
            
            if products and len(products) > 0:
                # Take the best option (first result)
                best_product = products[0]
                
                grocery_list["items"].append({
                    "searched_for": item,
                    "product": best_product.get("title", "Unknown"),
                    "brand": best_product.get("brand"),
                    "price": best_product.get("best_price", 0),
                    "store": best_product.get("best_store"),
                    "suggestion": best_product.get("suggestion"),
                    "alternatives": products[1:] if len(products) > 1 else []
                })
                
                # Add to total cost
                if best_product.get("best_price"):
                    grocery_list["total_cost"] += best_product.get("best_price")
                
                # Track stores
                store = best_product.get("best_store", "Unknown")
                if store not in grocery_list["stores"]:
                    grocery_list["stores"][store] = []
                grocery_list["stores"][store].append({
                    "item": item,
                    "product": best_product.get("title"),
                    "price": best_product.get("best_price")
                })
            else:
                grocery_list["items"].append({
                    "searched_for": item,
                    "product": "Not found",
                    "alternatives": []
                })
        
        # Check budget
        if budget_limit and grocery_list["total_cost"] > budget_limit:
            grocery_list["budget_status"] = "over_budget"
            overage = grocery_list["total_cost"] - budget_limit
            grocery_list["suggestions"].append(f"Over budget by €{overage:.2f}. Consider alternatives.")
        
        # Add shopping suggestions
        if len(grocery_list["stores"]) > 1:
            grocery_list["suggestions"].append("You'll need to visit multiple stores for the best prices.")
        
        return grocery_list
        
    except Exception as e:
        return {"error": f"Failed to create grocery list: {str(e)}"}


@tool
def plan_meal_with_products(
    meal_name: str,
    ingredients_needed: List[str],
    servings: int = 4,
    dietary_preferences: Optional[List[str]] = None,
    budget_limit: Optional[float] = None
) -> Dict[str, Any]:
    """Plan a meal by finding actual products for all required ingredients.
    
    Args:
        meal_name: Name of the meal/recipe
        ingredients_needed: List of ingredients required
        servings: Number of servings (default: 4)
        dietary_preferences: Dietary restrictions/preferences
        budget_limit: Maximum budget for the meal
    
    Returns:
        Complete meal plan with products, costs, and shopping information
    """
    try:
        meal_plan = {
            "meal_name": meal_name,
            "servings": servings,
            "ingredients": [],
            "total_cost": 0.0,
            "cost_per_serving": 0.0,
            "stores_needed": {},
            "budget_status": "within_budget",
            "preparation_notes": []
        }
        
        for ingredient in ingredients_needed:
            # Search for ingredient products
            products = search_grocery_products.invoke({
                "query": ingredient,
                "dietary_filters": dietary_preferences,
                "limit": 3
            })
            
            if products and len(products) > 0:
                best_product = products[0]
                
                ingredient_info = {
                    "ingredient": ingredient,
                    "product_found": best_product.get("title", "Unknown"),
                    "brand": best_product.get("brand"),
                    "price": best_product.get("best_price", 0),
                    "store": best_product.get("best_store"),
                    "suggestion": best_product.get("suggestion"),
                    "alternatives": [
                        {
                            "product": p.get("title"),
                            "price": p.get("best_price"),
                            "store": p.get("best_store")
                        } for p in products[1:] if p.get("best_price")
                    ]
                }
                
                meal_plan["ingredients"].append(ingredient_info)
                
                # Add to total cost
                if best_product.get("best_price"):
                    meal_plan["total_cost"] += best_product.get("best_price")
                
                # Track stores
                store = best_product.get("best_store", "Unknown")
                if store not in meal_plan["stores_needed"]:
                    meal_plan["stores_needed"][store] = []
                meal_plan["stores_needed"][store].append(ingredient)
                
            else:
                meal_plan["ingredients"].append({
                    "ingredient": ingredient,
                    "product_found": "Not available",
                    "alternatives": []
                })
        
        # Calculate cost per serving
        if meal_plan["total_cost"] > 0:
            meal_plan["cost_per_serving"] = meal_plan["total_cost"] / servings
        
        # Budget check
        if budget_limit and meal_plan["total_cost"] > budget_limit:
            meal_plan["budget_status"] = "over_budget"
            overage = meal_plan["total_cost"] - budget_limit
            meal_plan["preparation_notes"].append(f"Meal exceeds budget by €{overage:.2f}")
        
        # Add helpful notes
        if len(meal_plan["stores_needed"]) == 1:
            store_name = list(meal_plan["stores_needed"].keys())[0]
            meal_plan["preparation_notes"].append(f"All ingredients available at {store_name}")
        elif len(meal_plan["stores_needed"]) > 1:
            meal_plan["preparation_notes"].append(f"Ingredients available across {len(meal_plan['stores_needed'])} stores")
        
        return meal_plan
        
    except Exception as e:
        return {"error": f"Failed to plan meal: {str(e)}"}


@tool 
def suggest_weekly_meal_plan(
    number_of_meals: int = 7,
    dietary_preferences: Optional[List[str]] = None,
    budget_per_meal: Optional[float] = None,
    cuisine_preferences: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Suggest a weekly meal plan with actual products and shopping list.
    
    Args:
        number_of_meals: Number of meals to plan (default: 7 for a week)
        dietary_preferences: Dietary restrictions/preferences  
        budget_per_meal: Maximum budget per meal
        cuisine_preferences: Preferred cuisines (e.g., ['Italian', 'Asian'])
    
    Returns:
        Weekly meal plan with shopping list and cost breakdown
    """
    try:
        # Sample meal ideas with common ingredients
        meal_ideas = [
            {
                "name": "Spaghetti Bolognese",
                "ingredients": ["pasta", "ground beef", "tomato sauce", "onion", "garlic"]
            },
            {
                "name": "Chicken Stir Fry", 
                "ingredients": ["chicken breast", "mixed vegetables", "soy sauce", "rice"]
            },
            {
                "name": "Vegetable Soup",
                "ingredients": ["vegetable broth", "carrots", "celery", "onion", "potatoes"]
            },
            {
                "name": "Grilled Salmon",
                "ingredients": ["salmon fillet", "lemon", "asparagus", "olive oil"]
            },
            {
                "name": "Beef Tacos",
                "ingredients": ["ground beef", "taco shells", "cheese", "lettuce", "tomatoes"]
            },
            {
                "name": "Chicken Caesar Salad",
                "ingredients": ["chicken breast", "romaine lettuce", "caesar dressing", "croutons"]
            },
            {
                "name": "Vegetable Pasta",
                "ingredients": ["pasta", "zucchini", "bell peppers", "olive oil", "parmesan cheese"]
            }
        ]
        
        # Select meals for the week
        selected_meals = meal_ideas[:number_of_meals]
        
        weekly_plan = {
            "meals": [],
            "consolidated_shopping_list": {},
            "total_weekly_cost": 0.0,
            "stores_needed": {},
            "budget_analysis": {}
        }
        
        # Plan each meal
        for i, meal in enumerate(selected_meals):
            meal_plan = plan_meal_with_products.invoke({
                "meal_name": meal["name"],
                "ingredients_needed": meal["ingredients"],
                "dietary_preferences": dietary_preferences,
                "budget_limit": budget_per_meal
            })
            
            if not meal_plan.get("error"):
                weekly_plan["meals"].append(meal_plan)
                weekly_plan["total_weekly_cost"] += meal_plan.get("total_cost", 0)
                
                # Consolidate shopping list
                for ingredient_info in meal_plan.get("ingredients", []):
                    ingredient = ingredient_info.get("ingredient")
                    if ingredient and ingredient not in weekly_plan["consolidated_shopping_list"]:
                        weekly_plan["consolidated_shopping_list"][ingredient] = ingredient_info
        
        # Analyze budget
        if budget_per_meal:
            total_budget = budget_per_meal * number_of_meals
            weekly_plan["budget_analysis"] = {
                "total_budget": total_budget,
                "actual_cost": weekly_plan["total_weekly_cost"],
                "under_over": "under" if weekly_plan["total_weekly_cost"] <= total_budget else "over",
                "difference": abs(total_budget - weekly_plan["total_weekly_cost"])
            }
        
        return weekly_plan
        
    except Exception as e:
        return {"error": f"Failed to create weekly meal plan: {str(e)}"} 