from typing import List, Tuple
from core.schemas import ResearchContext  # Assuming your ResearchContext is defined here



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

For legal questions: Focus on statutes, case law, constitutional articles, regulations
For technical questions: Focus on standards, specifications, research papers, expert analyses
For current events: Focus on recent developments, official statements, expert commentary

Return your response in this format:
- search_queries: List of queries
- research_rationale: Why these queries will advance our understanding
- expected_findings: What types of information we hope to discover
"""
        return prompt

    @staticmethod
    def concept_extraction_prompt(user_question, new_content: str) -> str:
        prompt = f"""
Analyze this research content and extract key concepts, entities, and important details taking into account the original user question:
ORIGINAL USER QUESTION: 
"{user_question}"

CONTENT:
{new_content}

Extract and return:
1. Key legal concepts, articles, statutes, or regulations mentioned
2. Important names, organizations, or institutions
3. Specific numbers, dates, or quantitative information
4. Technical terms or specialized vocabulary
5. Cross-references to other important topics for further research


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

Create an UPDATED COMPREHENSIVE SUMMARY that:

1. **DIRECTLY ADDRESSES** the original user question
2. **INTEGRATES** new findings with existing knowledge
3. **MAINTAINS** all important details and citations
4. **IDENTIFIES** remaining knowledge gaps
5. **SUGGESTS** areas for deeper investigation
6. **ORGANIZES** information logically and clearly

Structure your summary with these sections:
- **Main Answer**: Direct response to the user's original question based on current knowledge
- **Key Findings**: Most important discoveries from research
- **Supporting Evidence**: Citations and sources supporting the findings
- **Related Concepts**: Connected topics that provide context
- **Knowledge Gaps**: Areas requiring further investigation
- **Confidence Level**: Assessment of how complete the answer is (Low/Medium/High)

Ensure every claim is properly cited with source URLs when available.
Return ONLY the updated summary, nothing else.
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

Rate the completeness on these criteria (1-10 scale):
1. **Directness**: Does the summary directly answer the user's question?
2. **Depth**: Is the information sufficiently detailed and comprehensive?
3. **Authority**: Are the sources authoritative and credible?
4. **Coverage**: Are all important aspects of the question covered?
5. **Currency**: Is the information current and up-to-date?
"""
        return prompt

    @staticmethod
    def source_validation_prompt(source_url: str, content: str) -> str:
        prompt = f"""
Evaluate the credibility and relevance of this source:

SOURCE URL: {source_url}
CONTENT PREVIEW: {content[:500]}...

Rate this source on:
1. **Credibility**: Is this from a reputable, authoritative source?
2. **Relevance**: How relevant is this content to research topics?
3. **Recency**: Is the information current and up-to-date?
4. **Depth**: Does it provide substantial, detailed information?
"""
        return prompt

    @staticmethod
    def query_refinement_prompt(original_query: str, previous_results: List[str]) -> str:
        prompt = f"""
Refine this search query to get better, more specific results:

ORIGINAL QUERY: "{original_query}"

PREVIOUS RESULTS SUMMARY:
{chr(10).join(previous_results[:3]) if previous_results else "No previous results"}

Create 2-3 refined search queries that:
1. Are more specific and targeted
2. Use different terminology or approach
3. Focus on gaps in current results
4. Avoid repeating ineffective searches
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

Provide a final answer that:
1. Directly and clearly answers the original question
2. Is well-structured and easy to understand
3. Includes key supporting evidence with citations
4. Acknowledges any limitations or uncertainties
5. Suggests related topics for further exploration

Format as a clear, comprehensive response suitable for the user.
"""
        return prompt

    @staticmethod
    def error_recovery_prompt(error_context: str, user_question: str) -> str:
        prompt = f"""
An error occurred during research. Suggest alternative approaches.

ERROR CONTEXT: {error_context}
ORIGINAL QUESTION: "{user_question}"

Suggest 3 alternative research strategies that could work around this issue:
1. Different search terms or approaches
2. Alternative information sources
3. Modified research methodology"""
        return prompt
