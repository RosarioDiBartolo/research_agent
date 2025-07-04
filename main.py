
# === USAGE EXAMPLE ===
from core.agent import ResearchAgent
from core.schemas import ResearchConfig

from models import gemini as model
from utils.logging import setup_logging

setup_logging()  # Set up logging at the start of your script

def main():
    """Example usage of the Advanced Research Agent"""
    
    # Initialize the research agent
     
    agent = ResearchAgent(
        model=model,
        config= ResearchConfig(
        verbose=True,
        max_iterations=5
        )
    )
    
    # Example research questions
    questions = [
        "All python built-in functions",
       ]
    
    # Conduct research
    for question in questions[:1]:  # Just do one for demo
        print(f"\n{'='*100}")
        print(f"RESEARCHING: {question}")
        print(f"{'='*100}")
        
        results = agent.conduct_research(question)
        
        print(f"\n📋 FINAL RESEARCH RESULTS:")
        print(f"📊 Total Sources: {results.total_sources}")
        print(f"🔄 Iterations: {results.iterations_completed}")
        print(f"🎯 Key Concepts: {', '.join(results.key_concepts_discovered[:5])}")
        print(f"\n📄 FINAL SUMMARY:")
        print("-" * 50)
        print(results.final_summary)

        print(f"\n🔗 SOURCES CONSULTED:")
        for i, source in enumerate(results.sources_used[:10], 1):
            print(f"{i}. {source}")

if __name__ == "__main__":
    main()