
# === USAGE EXAMPLE ===
from core.agent import ResearchAgent


def main():
    """Example usage of the Advanced Research Agent"""
    
    # Initialize the research agent
    agent = ResearchAgent(
        model_name="gpt-4",  # or "gpt-3.5-turbo" for faster/cheaper option
        temperature=0.2,
        max_iterations=5
    )
    
    # Example research questions
    questions = [
        "What are the key provisions of Article 23 of the US Constitution and how have they been interpreted by courts?",
        "What are the latest developments in quantum computing error correction techniques?",
        "How do carbon pricing mechanisms work in different countries and what are their effectiveness rates?"
    ]
    
    # Conduct research
    for question in questions[:1]:  # Just do one for demo
        print(f"\n{'='*100}")
        print(f"RESEARCHING: {question}")
        print(f"{'='*100}")
        
        results = agent.conduct_research(question)
        
        print(f"\n📋 FINAL RESEARCH RESULTS:")
        print(f"📊 Total Sources: {results['total_sources']}")
        print(f"🔄 Iterations: {results['iterations_completed']}")
        print(f"🎯 Key Concepts: {', '.join(results['key_concepts_discovered'][:5])}")
        print(f"\n📄 FINAL SUMMARY:")
        print("-" * 50)
        print(results['final_summary'])
        print(f"\n🔗 SOURCES CONSULTED:")
        for i, source in enumerate(results['sources_used'][:10], 1):
            print(f"{i}. {source}")

if __name__ == "__main__":
    main()