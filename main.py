import sys
import json
import argparse
from typing import List, Dict, Any
from src.agent import CustomerSupportAgent


def run_batch_eval(agent: CustomerSupportAgent, queries_file: str = "data/sample_queries.json"):
    print("=" * 80)
    print(" IIFL FINANCE - POLICY-AWARE CUSTOMER SUPPORT AGENT (BATCH EVALUATION)")
    print("=" * 80)

    try:
        with open(queries_file, "r", encoding="utf-8") as f:
            sample_queries: List[Dict[str, Any]] = json.load(f)
    except Exception as e:
        print(f"Error loading {queries_file}: {e}")
        return

    results = []
    for item in sample_queries:
        q_id = item.get("id")
        query_text = item.get("query", "")
        desc = item.get("description", "")

        print(f"\n--- [Test Case {q_id}] Description: {desc} ---")
        print(f"Input Query: '{query_text}'")

        response = agent.process_query(query_text)
        formatted_json = json.dumps(response.model_dump(), indent=2)
        print("Structured Output:")
        print(formatted_json)
        results.append(response.model_dump())

    print("\n" + "=" * 80)
    print(f"Batch evaluation complete ({len(results)} test cases executed).")
    print("=" * 80)


def run_interactive(agent: CustomerSupportAgent):
    print("=" * 80)
    print(" IIFL FINANCE - INTERACTIVE CUSTOMER SUPPORT AGENT")
    print(" Type 'exit' or 'quit' to end the session.")
    print("=" * 80 + "\n")

    while True:
        try:
            user_input = input("\nCustomer Query > ")
            if user_input.strip().lower() in ["exit", "quit"]:
                print("Exiting support agent session. Goodbye!")
                break
            
            response = agent.process_query(user_input)
            print("\nStructured Response:")
            print(json.dumps(response.model_dump(), indent=2))
        except KeyboardInterrupt:
            print("\nSession ended.")
            break


def main():
    parser = argparse.ArgumentParser(description="IIFL Policy-Aware Customer Support Agent CLI")
    parser.add_argument("--query", "-q", type=str, help="Single query to process")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run interactive mode")
    parser.add_argument("--policy-dir", type=str, default="data/policies", help="Directory containing policy markdown files")
    
    args = parser.parse_args()

    # Initialize agent
    agent = CustomerSupportAgent(policy_dir=args.policy_dir)

    if args.query is not None:
        response = agent.process_query(args.query)
        print(json.dumps(response.model_dump(), indent=2))
    elif args.interactive:
        run_interactive(agent)
    else:
        run_batch_eval(agent)


if __name__ == "__main__":
    is_streamlit = False
    try:
        import streamlit as st
        if st.runtime.exists():
            is_streamlit = True
    except Exception:
        pass

    if is_streamlit:
        import app
    else:
        main()
