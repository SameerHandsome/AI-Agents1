"""
evaluator.py - Agent Evaluation System
Place in root directory (agent-evaluation/)
Evaluates both basic_agent and advanced_agent
"""

import json
import time
import sys
import os
from typing import Dict, List
from datetime import datetime
from collections import defaultdict

# Add paths to import from subdirectories
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'basic_agent'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'advanced_agent'))


# ============================================================================
# AGENT RUNNERS
# ============================================================================

def run_basic_agent(query: str, session_id: str = "eval_basic"):
    """Run the basic agent"""
    # Import from basic_agent folder
    from basic_agent.graph import create_agent as create_basic
    from langchain_core.messages import HumanMessage
    
    agent, memory = create_basic()
    
    result = agent.invoke({"messages": [HumanMessage(content=query)]})
    response = result['messages'][-1].content
    
    # Extract tool calls
    tool_calls = []
    for msg in result['messages']:
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            tool_calls.extend([tc['name'] for tc in msg.tool_calls])
    
    return {
        "response": response,
        "tool_calls": tool_calls
    }


def run_advanced_agent(query: str, session_id: str = "eval_advanced"):
    """Run the advanced agent"""
    # Import from advanced_agent folder
    from advanced_agent.graph import create_agent as create_advanced
    from langchain_core.messages import HumanMessage
    
    agent, memory = create_advanced()
    
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "session_id": session_id,
        "start_time": time.time(),
        "tool_calls_count": 0,
        "reflection_count": 0
    }
    
    config = {"configurable": {"thread_id": session_id}, "recursion_limit": 50}
    result = agent.invoke(initial_state, config)
    
    response = result['messages'][-1].content
    
    # Extract tool calls
    tool_calls = []
    for msg in result['messages']:
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            tool_calls.extend([tc['name'] for tc in msg.tool_calls])
    
    return {
        "response": response,
        "tool_calls": tool_calls,
        "tool_calls_count": result.get('tool_calls_count', 0),
        "reflection_count": result.get('reflection_count', 0)
    }


# ============================================================================
# EVALUATION METRICS
# ============================================================================

class AgentEvaluator:
    """Evaluate agent performance against golden dataset"""
    
    def __init__(self, dataset_path: str = "golden_dataset.json"):
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.dataset = data['questions']
        self.metadata = data['metadata']
    
    def calculate_tool_recall(self, expected_tools: List[str], actual_tools: List[str]) -> float:
        """
        Tool Recall = (Expected tools that were called) / (Total expected tools)
        Measures: Did the agent use all necessary tools?
        """
        if not expected_tools:
            return 1.0
        
        expected_set = set(expected_tools)
        actual_set = set(actual_tools)
        
        correctly_called = len(expected_set & actual_set)
        return correctly_called / len(expected_set)
    
    def calculate_tool_precision(self, expected_tools: List[str], actual_tools: List[str]) -> float:
        """
        Tool Precision = (Correct tool calls) / (Total tool calls)
        Measures: Did the agent avoid unnecessary tool calls?
        """
        if not actual_tools:
            return 1.0 if not expected_tools else 0.0
        
        expected_set = set(expected_tools)
        actual_set = set(actual_tools)
        
        correctly_called = len(expected_set & actual_set)
        return correctly_called / len(actual_set)
    
    def calculate_answer_accuracy(self, expected_contains: List[str], actual_response: str) -> float:
        """
        Answer Accuracy = (Expected keywords found) / (Total expected keywords)
        Measures: Does the response contain expected information?
        """
        if not expected_contains:
            return 1.0
        
        actual_lower = actual_response.lower()
        found = sum(1 for keyword in expected_contains if keyword.lower() in actual_lower)
        
        return found / len(expected_contains)
    
    def evaluate_task_success(self, tool_recall: float, answer_accuracy: float) -> bool:
        """
        Task Success = Tool Recall >= 0.8 AND Answer Accuracy >= 0.6
        Measures: Did the agent complete the task successfully?
        """
        return tool_recall >= 0.8 and answer_accuracy >= 0.6
    
    def evaluate_single_query(self, query_data: Dict, agent_result: Dict) -> Dict:
        """Evaluate a single query"""
        
        tool_recall = self.calculate_tool_recall(
            query_data['expected_tools'],
            agent_result['tool_calls']
        )
        
        tool_precision = self.calculate_tool_precision(
            query_data['expected_tools'],
            agent_result['tool_calls']
        )
        
        answer_accuracy = self.calculate_answer_accuracy(
            query_data['expected_response_contains'],
            agent_result['response']
        )
        
        task_success = self.evaluate_task_success(tool_recall, answer_accuracy)
        
        return {
            "query_id": query_data['id'],
            "query": query_data['query'],
            "difficulty": query_data['difficulty'],
            "category": query_data['task_category'],
            "metrics": {
                "task_success": task_success,
                "tool_recall": tool_recall,
                "tool_precision": tool_precision,
                "answer_accuracy": answer_accuracy
            },
            "details": {
                "expected_tools": query_data['expected_tools'],
                "actual_tools": agent_result['tool_calls'],
                "expected_keywords": query_data['expected_response_contains'],
                "response_preview": agent_result['response'][:200] + "..."
            }
        }
    
    def evaluate_agent(self, agent_name: str, agent_runner) -> Dict:
        """Evaluate an agent on the entire dataset"""
        
        print(f"\n{'='*70}")
        print(f"🔍 Evaluating {agent_name}")
        print(f"{'='*70}\n")
        
        results = []
        start_time = time.time()
        
        for i, query_data in enumerate(self.dataset, 1):
            print(f"[{i}/{len(self.dataset)}] Testing: {query_data['query'][:60]}...")
            
            try:
                agent_result = agent_runner(
                    query_data['query'],
                    session_id=f"eval_{agent_name.replace(' ', '_')}_{i}"
                )
                
                evaluation = self.evaluate_single_query(query_data, agent_result)
                evaluation["agent"] = agent_name
                
                results.append(evaluation)
                
                # Show quick result
                success = "✅" if evaluation["metrics"]["task_success"] else "❌"
                print(f"  {success} Success: {evaluation['metrics']['task_success']}, "
                      f"Recall: {evaluation['metrics']['tool_recall']:.2f}, "
                      f"Precision: {evaluation['metrics']['tool_precision']:.2f}, "
                      f"Accuracy: {evaluation['metrics']['answer_accuracy']:.2f}\n")
                
                # Small delay to avoid rate limits
                time.sleep(1)
                
            except Exception as e:
                print(f"  ❌ ERROR: {str(e)}\n")
                results.append({
                    "query_id": query_data['id'],
                    "query": query_data['query'],
                    "difficulty": query_data['difficulty'],  # Added
                    "category": query_data['task_category'],  # Added
                    "agent": agent_name,
                    "error": str(e),
                    "metrics": {
                        "task_success": False,
                        "tool_recall": 0.0,
                        "tool_precision": 0.0,
                        "answer_accuracy": 0.0
                    },
                    "details": {  # Added
                        "expected_tools": query_data['expected_tools'],
                        "actual_tools": [],
                        "expected_keywords": query_data['expected_response_contains'],
                        "response_preview": f"ERROR: {str(e)}"
                    }
                })
        
        total_time = time.time() - start_time
        
        # Calculate aggregate metrics
        aggregate = self.calculate_aggregate_metrics(results)
        aggregate["total_execution_time"] = total_time
        aggregate["avg_time_per_query"] = total_time / len(self.dataset)
        
        return {
            "agent_name": agent_name,
            "individual_results": results,
            "aggregate_metrics": aggregate,
            "evaluated_at": datetime.now().isoformat()
        }
    
    def calculate_aggregate_metrics(self, results: List[Dict]) -> Dict:
        """Calculate aggregate metrics across all queries"""
        
        # Overall metrics
        task_success_rate = sum(1 for r in results if r["metrics"]["task_success"]) / len(results)
        avg_tool_recall = sum(r["metrics"]["tool_recall"] for r in results) / len(results)
        avg_tool_precision = sum(r["metrics"]["tool_precision"] for r in results) / len(results)
        avg_answer_accuracy = sum(r["metrics"]["answer_accuracy"] for r in results) / len(results)
        
        # By difficulty
        by_difficulty = defaultdict(list)
        for r in results:
            by_difficulty[r["difficulty"]].append(r)
        
        difficulty_metrics = {}
        for diff, items in by_difficulty.items():
            difficulty_metrics[diff] = {
                "task_success_rate": sum(1 for i in items if i["metrics"]["task_success"]) / len(items),
                "avg_tool_recall": sum(i["metrics"]["tool_recall"] for i in items) / len(items),
                "count": len(items)
            }
        
        # By category
        by_category = defaultdict(list)
        for r in results:
            by_category[r["category"]].append(r)
        
        category_metrics = {}
        for cat, items in by_category.items():
            category_metrics[cat] = {
                "task_success_rate": sum(1 for i in items if i["metrics"]["task_success"]) / len(items),
                "avg_tool_recall": sum(i["metrics"]["tool_recall"] for i in items) / len(items),
                "count": len(items)
            }
        
        return {
            "overall": {
                "task_success_rate": task_success_rate,
                "avg_tool_recall": avg_tool_recall,
                "avg_tool_precision": avg_tool_precision,
                "avg_answer_accuracy": avg_answer_accuracy
            },
            "by_difficulty": difficulty_metrics,
            "by_category": category_metrics
        }
    
    def compare_agents(self, basic_results: Dict, advanced_results: Dict):
        """Compare two agents and generate report"""
        
        print(f"\n{'='*70}")
        print(f"📊 COMPARISON REPORT")
        print(f"{'='*70}\n")
        
        # Overall comparison
        print("Overall Performance:")
        print(f"{'Metric':<30} {'Basic Agent':<20} {'Advanced Agent':<20} {'Improvement'}")
        print("-" * 70)
        
        basic_overall = basic_results['aggregate_metrics']['overall']
        advanced_overall = advanced_results['aggregate_metrics']['overall']
        
        metrics_to_compare = [
            ("Task Success Rate", "task_success_rate"),
            ("Avg Tool Recall", "avg_tool_recall"),
            ("Avg Tool Precision", "avg_tool_precision"),
            ("Avg Answer Accuracy", "avg_answer_accuracy")
        ]
        
        for metric_name, metric_key in metrics_to_compare:
            basic_val = basic_overall[metric_key]
            advanced_val = advanced_overall[metric_key]
            improvement = ((advanced_val - basic_val) / basic_val * 100) if basic_val > 0 else 0
            
            arrow = "📈" if improvement > 0 else "📉" if improvement < 0 else "➡️"
            print(f"{metric_name:<30} {basic_val:<20.2%} {advanced_val:<20.2%} {arrow} {improvement:+.1f}%")
        
        print(f"\nExecution Time:")
        print(f"{'Agent':<30} {'Total Time':<20} {'Avg per Query'}")
        print("-" * 70)
        print(f"{'Basic Agent':<30} {basic_results['aggregate_metrics']['total_execution_time']:<20.2f}s "
              f"{basic_results['aggregate_metrics']['avg_time_per_query']:.2f}s")
        print(f"{'Advanced Agent':<30} {advanced_results['aggregate_metrics']['total_execution_time']:<20.2f}s "
              f"{advanced_results['aggregate_metrics']['avg_time_per_query']:.2f}s")
        
        # By difficulty
        print(f"\n\nPerformance by Difficulty:")
        print(f"{'Difficulty':<20} {'Basic Success':<20} {'Advanced Success':<20} {'Improvement'}")
        print("-" * 70)
        
        for diff in ['easy', 'medium', 'hard']:
            if diff in basic_results['aggregate_metrics']['by_difficulty']:
                basic_rate = basic_results['aggregate_metrics']['by_difficulty'][diff]['task_success_rate']
                advanced_rate = advanced_results['aggregate_metrics']['by_difficulty'][diff]['task_success_rate']
                improvement = ((advanced_rate - basic_rate) / basic_rate * 100) if basic_rate > 0 else 0
                
                arrow = "📈" if improvement > 0 else "📉" if improvement < 0 else "➡️"
                print(f"{diff.capitalize():<20} {basic_rate:<20.2%} {advanced_rate:<20.2%} {arrow} {improvement:+.1f}%")
        
        print("\n" + "="*70 + "\n")


def run_full_evaluation():
    """Run complete evaluation on both agents"""
    
    print("🚀 Starting Agent Evaluation")
    print("="*70)
    
    evaluator = AgentEvaluator("golden_dataset.json")
    
    basic_results = evaluator.evaluate_agent("Basic Agent", run_basic_agent)
    
    advanced_results = evaluator.evaluate_agent("Advanced Agent", run_advanced_agent)
    
    evaluator.compare_agents(basic_results, advanced_results)
    
    results = {
        "basic_agent": basic_results,
        "advanced_agent": advanced_results,
        "comparison_date": datetime.now().isoformat()
    }
    
    output_file = f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Full results saved to: {output_file}")
    
    return results


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    run_full_evaluation()