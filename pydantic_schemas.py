from pydantic import BaseModel
from typing import List

class SearchStrategyResponse(BaseModel):
    search_queries: List[str]
    research_rationale: str
    expected_findings: str


class ConceptExtractionResponse(BaseModel):
    key_concepts: List[str]


class ComprehensiveSummaryResponse(BaseModel):
    main_answer: str
    key_findings: str
    supporting_evidence: str
    related_concepts: str
    knowledge_gaps: str
    confidence_level: str  # Should be one of "Low", "Medium", "High"


class ResearchCompletenessResponse(BaseModel):
    should_continue: bool
    completeness_score: int  # 0–100
    reasoning: str
    missing_aspects: List[str]
    recommended_next_searches: List[str]


class SourceValidationResponse(BaseModel):
    credibility_score: int  # 0–10
    relevance_score: int    # 0–10
    overall_quality: int    # 0–10
    source_type: str        # e.g., "academic", "news", "government", "commercial", "blog", or "other"
    recommendation: str     # "include", "exclude", or "review"
    reasoning: str


class QueryRefinementResponse(BaseModel):
    refined_queries: List[str]


class FinalAnswerResponse(BaseModel):
    final_answer: str


class ErrorRecoveryStrategy(BaseModel):
    strategy: str
    queries: List[str]


class ErrorRecoveryResponse(BaseModel):
    alternatives: List[ErrorRecoveryStrategy]
    explanation: str
