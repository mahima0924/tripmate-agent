"""
Entry point for running the TripMate agent interactively from the CLI.

Usage:
    python main.py
Then type natural-language questions. Type 'exit' or 'quit' to stop.
"""

from src.logging_setup import setup_logging
from src.agent.orchestrator import run_agent

logger = setup_logging()


def main():
    print("TripMate Agent -- ask me about destinations, weather, or packing.")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        answer = run_agent(user_input)
        print(f"\nTripMate: {answer}\n")


if __name__ == "__main__":
    main()