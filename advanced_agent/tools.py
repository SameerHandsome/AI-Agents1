"""
tools.py - All Tools with Retry Logic and Validation
"""

from functools import wraps
import time
from langchain_core.tools import tool
import requests
import os
from datetime import datetime
import pytz

def retry_on_failure(max_retries=3):
    """Retry decorator with exponential backoff"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for i in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if i == max_retries - 1:
                        return f"Error after {max_retries} retries: {str(e)}"
                    time.sleep(2 ** i)  
            return "Max retries exceeded"
        return wrapper
    return decorator


def validate_output(validator):
    """Validate tool output before returning"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            if not validator(result):
                return "Tool validation failed - invalid or incomplete data"
            return result
        return wrapper
    return decorator


def is_valid_weather(result: str) -> bool:
    """Validate weather tool output"""
    return (
        "Temperature:" in result and 
        "°C" in result and 
        not result.startswith("Error") and
        not result.startswith("Could not")
    )


def is_valid_currency(result: str) -> bool:
    """Validate currency conversion output"""
    return (
        "=" in result and 
        not result.startswith("Currency conversion failed") and
        not result.startswith("Error")
    )


def is_valid_search(result: str) -> bool:
    """Validate search results"""
    return (
        len(result) > 50 and 
        not result.startswith("Search failed") and
        not result.startswith("Error")
    )


def is_valid_wikipedia(result: str) -> bool:
    """Validate Wikipedia summary"""
    return (
        len(result) > 30 and
        not result.startswith("Wikipedia lookup failed") and
        not result.startswith("Error")
    )


def is_valid_time(result: str) -> bool:
    """Validate time lookup"""
    return (
        "Current time" in result and
        not result.startswith("Timezone lookup failed") and
        not result.startswith("Error")
    )


# ============================================================================
# TOOLS WITH RETRY AND VALIDATION
# ============================================================================

@tool
@retry_on_failure(max_retries=3)
@validate_output(is_valid_search)
def tavily_search(query: str) -> str:
    """
    Search the internet using Tavily API for current information.
    Use this for: news, current events, recent information, facts.
    
    Args:
        query: The search query string
        
    Returns:
        Search results as formatted string
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "Error: TAVILY_API_KEY not set in environment variables"
    
    url = "https://api.tavily.com/search"
    payload = {
        "api_key": api_key,
        "query": query,
        "search_depth": "basic",
        "max_results": 5
    }
    response = requests.post(url, json=payload, timeout=10)
    response.raise_for_status()
    data = response.json()
    
    results = []
    for item in data.get("results", [])[:3]:
        results.append(
            f"• {item.get('title', 'No title')}\n"
            f"  {item.get('content', 'No content')}\n"
            f"  Source: {item.get('url', 'N/A')}"
        )
    
    return "\n\n".join(results) if results else "No results found"


@tool
@retry_on_failure(max_retries=3)
@validate_output(is_valid_weather)
def get_weather(city: str) -> str:
    """
    Get current weather information for a city.
    
    Args:
        city: City name (e.g., "London", "New York", "Tokyo")
        
    Returns:
        Current weather information
    """
    url = f"https://wttr.in/{city}?format=j1"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    data = response.json()
    
    current = data["current_condition"][0]
    weather_info = f"""Weather in {city}:
• Temperature: {current['temp_C']}°C / {current['temp_F']}°F
• Condition: {current['weatherDesc'][0]['value']}
• Humidity: {current['humidity']}%
• Wind: {current['windspeedKmph']} km/h
• Feels like: {current['FeelsLikeC']}°C / {current['FeelsLikeF']}°F"""
    
    return weather_info


@tool
@retry_on_failure(max_retries=3)
@validate_output(is_valid_currency)
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """
    Convert currency from one type to another using current exchange rates.
    
    Args:
        amount: Amount to convert
        from_currency: Source currency code (e.g., "USD", "EUR", "GBP")
        to_currency: Target currency code
        
    Returns:
        Converted amount with rate information
    """
    url = f"https://api.exchangerate-api.com/v4/latest/{from_currency.upper()}"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    data = response.json()
    
    rate = data["rates"].get(to_currency.upper())
    if not rate:
        return f"Currency {to_currency} not found"
    
    converted = amount * rate
    return (
        f"{amount} {from_currency.upper()} = {converted:.2f} {to_currency.upper()}\n"
        f"Exchange rate: 1 {from_currency.upper()} = {rate:.4f} {to_currency.upper()}"
    )


@tool
@retry_on_failure(max_retries=3)
@validate_output(is_valid_wikipedia)
def get_wikipedia_summary(topic: str) -> str:
    """
    Get a summary of a topic from Wikipedia.
    
    Args:
        topic: The topic to search for
        
    Returns:
        Wikipedia summary text
    """
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic.replace(' ', '_')}"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    data = response.json()
    
    summary = f"""**{data.get('title', topic)}**

{data.get('extract', 'No summary available')}

Read more: {data.get('content_urls', {}).get('desktop', {}).get('page', 'N/A')}"""
    
    return summary


@tool
@retry_on_failure(max_retries=3)
@validate_output(is_valid_time)
def get_world_time(timezone: str) -> str:
    """
    Get current time in a specific timezone.
    
    Args:
        timezone: Timezone name (e.g., "America/New_York", "Europe/London", "Asia/Tokyo")
        
    Returns:
        Current time in the specified timezone
    """
    tz = pytz.timezone(timezone)
    current_time = datetime.now(tz)
    
    return f"""Current time in {timezone}:
• Date: {current_time.strftime('%A, %B %d, %Y')}
• Time: {current_time.strftime('%I:%M:%S %p')}
• 24h: {current_time.strftime('%H:%M:%S')}
• UTC Offset: {current_time.strftime('%z')}"""


ALL_TOOLS = [
    tavily_search,
    get_weather,
    convert_currency,
    get_wikipedia_summary,
    get_world_time
]