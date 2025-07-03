from typing import List, Tuple
from core.models import ResearchContext  # Assuming your ResearchContext is defined here



class PromptTemplates:
    """Collection of all prompt templates used by the research agent, with their respective Pydantic models."""

    @staticmethod
    def search_strategy_prompt(context: ResearchContext, max_iterations: int) -> str:
        prompt = f"""
You are an expert research strategist. Your task is to generate targeted search queries for deep research.

ORIGINAL USER QUESTION: "{context.user_question}"

CURRENT RESEARCH SUMMARY:
{context.current_summary if context.current_summary else "No prior research conducted"}

KEY CONCEPTS ALREADY EXPLORED:
{', '.join(context.key_concepts_found) if context.key_concepts_found else "None"}

SOURCES ALREADY USED (AVOID THESE):
{list(context.used_sources) if context.used_sources else "None"}

ITERATION: {context.iteration_count + 1}/{max_iterations}

Based on the current knowledge and gaps identified, generate 2-3 strategic search queries that will:
1. Fill knowledge gaps from the current summary
2. Explore deeper aspects mentioned but not fully covered
3. Find authoritative sources (laws, academic papers, official documents)
4. Avoid already-used sources
5. Build upon specific details mentioned in the summary

Return your response in this format:
- search_queries: List of queries
- research_rationale: Why these queries will advance our understanding
- expected_findings: What types of information we hope to discover
"""
        return prompt

    @staticmethod
    def concept_extraction_prompt(new_content: str) -> str:
        prompt = f"""
Analyze this research content and extract key concepts, entities, and important details:

CONTENT:
{new_content}

Extract and return a list called `key_concepts` that includes:
- Legal concepts, articles, statutes, or regulations
- Names of people, organizations, institutions
- Numbers, dates, or other specific information
- Technical terms or domain-specific vocabulary
- Cross-references to related research topics

Return: key_concepts: List[str]
"""
        return prompt

    @staticmethod
    def comprehensive_summary_prompt(context: ResearchContext, new_information: str) -> str:
        prompt = f"""
You are an expert research analyst. Create a comprehensive, well-structured summary.

ORIGINAL USER QUESTION: "{context.user_question}"

EXISTING SUMMARY:
{context.current_summary if context.current_summary else "No existing summary"}

NEW RESEARCH FINDINGS:
{new_information}

Create and return a structured summary with the following fields:
- main_answer: A direct response to the user's question
- key_findings: Most important research discoveries
- supporting_evidence: Cited sources and details
- related_concepts: Topics that provide context
- knowledge_gaps: Areas needing further research
- confidence_level: One of "Low", "Medium", "High"
"""
        return prompt

    @staticmethod
    def research_completeness_prompt(context: ResearchContext) -> str:
        prompt = f"""
Evaluate whether our current research is sufficient to provide a comprehensive answer.

ORIGINAL QUESTION: "{context.user_question}"
CURRENT SUMMARY: {context.current_summary}
ITERATIONS COMPLETED: {context.iteration_count}
SOURCES CONSULTED: {len(context.used_sources)}

Return these fields:
- should_continue: True or False
- completeness_score: Integer from 0 to 100
- reasoning: Explanation of this score
- missing_aspects: List of uncovered aspects
- recommended_next_searches: List of suggested queries
"""
        return prompt

    @staticmethod
    def source_validation_prompt(source_url: str, content: str) -> str:
        prompt = f"""
Evaluate the credibility and relevance of this source:

SOURCE URL: {source_url}
CONTENT PREVIEW: {content[:500]}...

Return:
- credibility_score: 0–10
- relevance_score: 0–10
- overall_quality: 0–10
- source_type: "academic", "news", "government", "commercial", "blog", or "other"
- recommendation: "include", "exclude", or "review"
- reasoning: Brief explanation of your rating
"""
        return prompt

    @staticmethod
    def query_refinement_prompt(original_query: str, previous_results: List[str]) -> str:
        prompt = f"""
Refine this search query to get better, more specific results:

ORIGINAL QUERY: "{original_query}"

PREVIOUS RESULTS SUMMARY:
{chr(10).join(previous_results[:3]) if previous_results else "No previous results"}

Return: refined_queries: A list of 2–3 improved search queries
"""
        return prompt

    @staticmethod
    def final_answer_prompt(context: ResearchContext) -> str:
        prompt = f"""
Based on comprehensive research, provide a final, authoritative answer.

ORIGINAL QUESTION: "{context.user_question}"

RESEARCH SUMMARY:
{context.current_summary}

SOURCES CONSULTED: {len(context.used_sources)} sources
RESEARCH DEPTH: {context.iteration_count} iterations

Return a field:
- final_answer: Clear, well-structured response that includes evidence, addresses limitations, and suggests related areas
"""
        return prompt

    @staticmethod
    def error_recovery_prompt(error_context: str, user_question: str) -> str:
        prompt = f"""
An error occurred during research. Suggest alternative approaches.

ERROR CONTEXT: {error_context}
ORIGINAL QUESTION: "{user_question}"

Return:
- alternatives: List of 3 strategies with a description and suggested queries
- explanation: Why these alternatives might work better
"""
        return prompt
