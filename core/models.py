"""
Data structures and models for the research agent system.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional
from datetime import datetime
from enum import Enum


class ResearchStatus(Enum):
    """Status of the research process"""
    INITIALIZED = "initialized"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class ConfidenceLevel(Enum):
    """Confidence level of research findings"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class SearchQuery:
    """Represents a search query with metadata"""
    query: str
    rationale: str
    expected_results: str
    iteration: int
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SearchResult:
    """Represents a search result with processed content"""
    url: str
    title: str
    content: str
    snippet: str
    relevance_score: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __hash__(self):
        return hash(self.url)
    
    def __eq__(self, other):
        if not isinstance(other, SearchResult):
            return False
        return self.url == other.url


@dataclass
class IterationResult:
    """Results from a single research iteration"""
    iteration_number: int
    search_queries: List[SearchQuery]
    search_results: List[SearchResult]
    new_sources_count: int
    summary_length: int
    key_concepts_found: List[str]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ResearchAssessment:
    """Assessment of research completeness"""
    should_continue: bool
    completeness_score: float  # 0-100
    reasoning: str
    missing_aspects: List[str] = field(default_factory=list)
    recommended_searches: List[str] = field(default_factory=list)
    confidence_level: ConfidenceLevel = ConfidenceLevel.MEDIUM


@dataclass
class ResearchContext:
    """Holds the complete state of the research process"""
    user_question: str
    current_summary: str = ""
    used_sources: Set[str] = field(default_factory=set)
    iteration_count: int = 0
    status: ResearchStatus = ResearchStatus.INITIALIZED
    key_concepts_found: List[str] = field(default_factory=list)
    research_history: List[IterationResult] = field(default_factory=list)
    search_results: List[SearchResult] = field(default_factory=list)
    last_assessment: Optional[ResearchAssessment] = None
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    
    def add_search_result(self, result: SearchResult) -> bool:
        """Add a search result if it's not already present"""
        if result.url not in self.used_sources:
            self.used_sources.add(result.url)
            self.search_results.append(result)
            return True
        return False
    
    def get_duration(self) -> Optional[float]:
        """Get research duration in seconds"""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
    
    def get_sources_by_iteration(self, iteration: int) -> List[SearchResult]:
        """Get sources discovered in a specific iteration"""
        for iter_result in self.research_history:
            if iter_result.iteration_number == iteration:
                return iter_result.search_results
        return []


@dataclass
class ResearchResult:
    """Final result of the research process"""
    original_question: str
    final_summary: str
    sources_used: List[str]
    total_sources: int
    iterations_completed: int
    key_concepts_discovered: List[str]
    research_history: List[IterationResult]
    final_assessment: Optional[ResearchAssessment]
    duration_seconds: Optional[float]
    status: ResearchStatus
    metadata: Dict = field(default_factory=dict)
    
    @classmethod
    def from_context(cls, context: ResearchContext) -> 'ResearchResult':
        """Create a ResearchResult from a ResearchContext"""
        return cls(
            original_question=context.user_question,
            final_summary=context.current_summary,
            sources_used=list(context.used_sources),
            total_sources=len(context.used_sources),
            iterations_completed=context.iteration_count,
            key_concepts_discovered=context.key_concepts_found,
            research_history=context.research_history,
            final_assessment=context.last_assessment,
            duration_seconds=context.get_duration(),
            status=context.status
        )


@dataclass
class ResearchConfig:
    """Configuration for the research agent"""
    model_name: str = "gpt-4"
    temperature: float = 0.2
    max_iterations: int = 7
    max_search_results_per_query: int = 8
    min_completeness_score: float = 80.0
    search_timeout: int = 30
    enable_concept_extraction: bool = True
    enable_source_validation: bool = True
    verbose: bool = True
    custom_prompts: Dict[str, str] = field(default_factory=dict)