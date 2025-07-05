import json

from typing import Dict, List, Any
from pydantic import ValidationError
from core.schemas import ResearchConfig, ResearchAssessment, ConfidenceLevel, ResearchContext
from pydantic_schemas import (
    ComprehensiveSummaryResponse,
    ConceptExtractionResponse,
    ErrorRecoveryResponse,
    ErrorRecoveryStrategy,
    QueryRefinementResponse,
    ResearchCompletenessResponse,
    SearchStrategyResponse,
    SourceValidationResponse,
) 
from utils.prompts import PromptTemplates
from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
 
load_dotenv()


class LLMService:
    """Service for managing all interactions with language models"""

    def __init__(self, model: BaseChatModel, config: ResearchConfig):
        self.config = config
        self.llm =  model 
        self.search_strategy_model = self.llm.with_structured_output(SearchStrategyResponse)
        self.key_concepts_model = self.llm.with_structured_output(ConceptExtractionResponse)
        self.research_completeness_model = self.llm.with_structured_output(ResearchCompletenessResponse)
        self.source_validation_model = self.llm.with_structured_output(SourceValidationResponse)
        self.error_recovery_model = self.llm.with_structured_output(ErrorRecoveryResponse)
        self.query_refinement_model = self.llm.with_structured_output(QueryRefinementResponse)
 
        self.llm.invoke("Simple test")

        self.prompts = PromptTemplates()

    def generate_search_strategy(self, context: ResearchContext) -> Dict[str, Any]:
        try:
            prompt = self.prompts.search_strategy_prompt(context, self.config.max_iterations)
            strategy = self.search_strategy_model.invoke(prompt)

            for field in ["search_queries", "research_rationale", "expected_findings"]:
                if getattr(strategy, field, None) is None:
                    raise ValueError(f"Missing required field '{field}' in strategy response")

            return strategy.dict()
        except (json.JSONDecodeError, ValidationError, ValueError):
            fallback = SearchStrategyResponse(
                search_queries=[context.user_question],
                research_rationale="Fallback to basic search due to parsing error",
                expected_findings="Basic information about the topic",
            )
            return fallback.dict()
        except Exception:
            raise

    def extract_key_concepts(self, content: str) -> List[str]:
        if not self.config.enable_concept_extraction:
            return []

        try:
            prompt = self.prompts.concept_extraction_prompt(content)
            concepts = self.key_concepts_model.invoke(prompt)
            return concepts.key_concepts if hasattr(concepts, 'key_concepts') else []
        except (json.JSONDecodeError, ValidationError):
            return []
        except Exception:
            return []

    def update_summary(self, context: ResearchContext, new_information: str) -> str:
        if not new_information.strip():
            return context.current_summary or ""

        try:
            prompt = self.prompts.comprehensive_summary_prompt(context, new_information)
            updated_summary = self.llm.invoke(prompt) 
            return   updated_summary 
        except (ValidationError, json.JSONDecodeError):
            return context.current_summary or ""
        except Exception:
            return context.current_summary or ""

    def assess_research_completeness(self, context: ResearchContext) -> ResearchAssessment:
        try:
            prompt = self.prompts.research_completeness_prompt(context)
            data = self.research_completeness_model.invoke(prompt)
            score = data.completeness_score if hasattr(data, 'completeness_score') else 0

            if score >= 80:
                confidence = ConfidenceLevel.HIGH
            elif score >= 60:
                confidence = ConfidenceLevel.MEDIUM
            else:
                confidence = ConfidenceLevel.LOW

            assessment = ResearchAssessment(
                should_continue=data.should_continue,
                completeness_score=score,
                reasoning=data.reasoning,
                missing_aspects=data.missing_aspects,
                recommended_searches=data.recommended_next_searches,
                confidence_level=confidence,
            )
            return assessment
        except (ValidationError, json.JSONDecodeError):
            return ResearchAssessment(
                should_continue=context.iteration_count < 3,
                completeness_score=50,
                reasoning="Assessment parsing failed, using fallback logic",
                missing_aspects=[],
                recommended_searches=[],
                confidence_level=ConfidenceLevel.LOW,
            )
        except Exception:
            raise

    def validate_source(self, source_url: str, content: str) -> Dict[str, Any]:
        if not self.config.enable_source_validation:
            return {"overall_quality": 7, "recommendation": "include"}

        try:
            prompt = self.prompts.source_validation_prompt(source_url, content)
            validation = self.source_validation_model.invoke(prompt)
            return validation.dict()
        except (ValidationError, json.JSONDecodeError):
            return {"overall_quality": 7, "recommendation": "include", "reasoning": "Default due to parsing error"}
        except Exception as e:
            return {"overall_quality": 5, "recommendation": "review", "reasoning": str(e)}

    def generate_final_answer(self, context: ResearchContext) -> str:
        try:
            prompt = self.prompts.final_answer_prompt(context)
            final_answer = self.llm.invoke(prompt)
            return final_answer.strip()
        except (ValidationError, json.JSONDecodeError):
            return context.current_summary or ""
        except Exception:
            return context.current_summary or ""

    def handle_error_recovery(self, error_context: str, user_question: str) -> Dict[str, Any]:
        try:
            prompt = self.prompts.error_recovery_prompt(error_context, user_question)
            recovery = self.error_recovery_model.invoke(prompt)
            return recovery.dict()
        except (ValidationError, json.JSONDecodeError):
            pass
        except Exception:
            pass

        fallback = ErrorRecoveryResponse(
            alternatives=[ErrorRecoveryStrategy(strategy="Basic keyword search", queries=[user_question])],
            explanation="Fallback due to error recovery failure",
        )
        return fallback.dict()

    def refine_query(self, original_query: str, previous_results: List[str]) -> List[str]:
        try:
            prompt = self.prompts.query_refinement_prompt(original_query, previous_results)
            refined = self.query_refinement_model.invoke(prompt)
            return refined.refined_queries if hasattr(refined, 'refined_queries') else [original_query]
        except (ValidationError, json.JSONDecodeError):
            return [original_query]
        except Exception:
            return [original_query]
