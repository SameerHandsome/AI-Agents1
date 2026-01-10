"""
generate_golden_dataset.py - Generate Golden Dataset for Agent Evaluation
"""

import json
from datetime import datetime


GOLDEN_DATASET = [
    {
        "id": 1,
        "query": "What's the weather in Tokyo?",
        "expected_tools": ["get_weather"],
        "expected_response_contains": ["Tokyo", "temperature", "°C"],
        "task_category": "single_tool",
        "difficulty": "easy"
    },
    {
        "id": 2,
        "query": "Convert 100 USD to EUR",
        "expected_tools": ["convert_currency"],
        "expected_response_contains": ["100", "USD", "EUR", "="],
        "task_category": "single_tool",
        "difficulty": "easy"
    },
    {
        "id": 3,
        "query": "What time is it in New York right now?",
        "expected_tools": ["get_world_time"],
        "expected_response_contains": ["New York", "time"],
        "task_category": "single_tool",
        "difficulty": "easy"
    },
    {
        "id": 4,
        "query": "Tell me about the Eiffel Tower",
        "expected_tools": ["get_wikipedia_summary"],
        "expected_response_contains": ["Eiffel Tower", "Paris"],
        "task_category": "single_tool",
        "difficulty": "easy"
    },
    {
        "id": 5,
        "query": "Search for the latest news about artificial intelligence",
        "expected_tools": ["tavily_search"],
        "expected_response_contains": ["AI", "artificial intelligence"],
        "task_category": "single_tool",
        "difficulty": "easy"
    },
    {
        "id": 6,
        "query": "What's the weather in London and what time is it there?",
        "expected_tools": ["get_weather", "get_world_time"],
        "expected_response_contains": ["London", "temperature", "time"],
        "task_category": "multi_tool",
        "difficulty": "medium"
    },
    {
        "id": 7,
        "query": "Compare the weather in Tokyo and New York",
        "expected_tools": ["get_weather"],  # Called twice
        "expected_response_contains": ["Tokyo", "New York", "temperature"],
        "task_category": "multi_tool",
        "difficulty": "medium"
    },
    {
        "id": 8,
        "query": "Search for recent AI breakthroughs and tell me about neural networks from Wikipedia",
        "expected_tools": ["tavily_search", "get_wikipedia_summary"],
        "expected_response_contains": ["AI", "neural network"],
        "task_category": "multi_tool",
        "difficulty": "medium"
    },
    {
        "id": 9,
        "query": "What's the weather in Paris, convert 50 EUR to USD, and what time is it in Paris?",
        "expected_tools": ["get_weather", "convert_currency", "get_world_time"],
        "expected_response_contains": ["Paris", "EUR", "USD", "temperature", "time"],
        "task_category": "complex_multi_tool",
        "difficulty": "hard"
    },
    {
        "id": 10,
        "query": "Search for information about climate change, get the weather in Sydney, and tell me about the Great Barrier Reef",
        "expected_tools": ["tavily_search", "get_weather", "get_wikipedia_summary"],
        "expected_response_contains": ["climate", "Sydney", "weather", "Great Barrier Reef"],
        "task_category": "complex_multi_tool",
        "difficulty": "hard"
    }
]


# ============================================================================
# SAVE DATASET
# ============================================================================

def generate_dataset(output_file: str = "golden_dataset.json"):
    """Generate and save the golden dataset"""
    
    dataset = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "total_questions": len(GOLDEN_DATASET),
            "categories": {
                "single_tool": len([q for q in GOLDEN_DATASET if q["task_category"] == "single_tool"]),
                "multi_tool": len([q for q in GOLDEN_DATASET if q["task_category"] == "multi_tool"]),
                "complex_multi_tool": len([q for q in GOLDEN_DATASET if q["task_category"] == "complex_multi_tool"])
            },
            "difficulty_levels": {
                "easy": len([q for q in GOLDEN_DATASET if q["difficulty"] == "easy"]),
                "medium": len([q for q in GOLDEN_DATASET if q["difficulty"] == "medium"]),
                "hard": len([q for q in GOLDEN_DATASET if q["difficulty"] == "hard"])
            }
        },
        "questions": GOLDEN_DATASET
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Golden dataset saved to {output_file}")
    print(f"📊 Total questions: {len(GOLDEN_DATASET)}")
    print(f"   - Easy: {dataset['metadata']['difficulty_levels']['easy']}")
    print(f"   - Medium: {dataset['metadata']['difficulty_levels']['medium']}")
    print(f"   - Hard: {dataset['metadata']['difficulty_levels']['hard']}")
    print(f"\n📋 Categories:")
    print(f"   - Single tool: {dataset['metadata']['categories']['single_tool']}")
    print(f"   - Multi tool: {dataset['metadata']['categories']['multi_tool']}")
    print(f"   - Complex: {dataset['metadata']['categories']['complex_multi_tool']}")
    
    return dataset


if __name__ == "__main__":
    generate_dataset()