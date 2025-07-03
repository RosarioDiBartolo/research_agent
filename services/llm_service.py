import json
import logging
from typing import Dict, List, Optional, Any
from pydantic import ValidationError
from langchain_google_genai import ChatGoogleGenerativeAI

from constant import GeminiKey
from core.models import ResearchConfig, ResearchAssessment, ConfidenceLevel, ResearchContext
from pydantic_schemas import (
    ComprehensiveSummaryResponse,
    ConceptExtractionResponse,
    ErrorRecoveryResponse,
    ErrorRecoveryStrategy,
    FinalAnswerResponse,
    QueryRefinementResponse,
    ResearchCompletenessResponse,
    SearchStrategyResponse,
    SourceValidationResponse,
)
from utils.prompts import PromptTemplates
from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
load_dotenv()

logger = logging.getLogger(__name__)


class LLMService:
    """Service for managing all interactions with language models"""

    def __init__(self , model: BaseChatModel, config: ResearchConfig):
        self.config = config
        self.llm = model

        # Wrap LLM with structured output models
        self.search_strategy_model = self.llm.with_structured_output(SearchStrategyResponse)
        self.key_concepts_model = self.llm.with_structured_output(ConceptExtractionResponse)
        self.research_completeness_model = self.llm.with_structured_output(ResearchCompletenessResponse)
        self.source_validation_model = self.llm.with_structured_output(SourceValidationResponse)
        self.error_recovery_model = self.llm.with_structured_output(ErrorRecoveryResponse)
        self.query_refinement_model = self.llm.with_structured_output(QueryRefinementResponse)
        self.comprehensive_summary_model = self.llm.with_structured_output(ComprehensiveSummaryResponse)
        self.final_answer_model = self.llm.with_structured_output(FinalAnswerResponse)

        try:
            self.llm.invoke("Simple test")
        except Exception as e:
            logger.error("Failed to connect to LLM", exc_info=True)
            raise RuntimeError("LLM connection failed, check your API key and model configuration.") from e

        self.prompts = PromptTemplates()

    def invoke_json(self, prompt: str) -> Dict[str, Any]:
        """Invoke the LLM with a JSON prompt and return parsed response"""
        try:
            response = self.llm.invoke(prompt)
            content = response.content.replace("```json", "").replace("```", "").strip()
            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.error("Failed to parse JSON response", exc_info=True)
            raise ValueError("LLM response is not valid JSON") from e
        except Exception as e:
            logger.error("Error invoking LLM", exc_info=True)
            raise RuntimeError("LLM invocation failed") from e

    def generate_search_strategy(self, context: ResearchContext) -> Dict[str, Any]:
        """Generate strategic search queries based on research context"""
        try:
            prompt = self.prompts.search_strategy_prompt(context, self.config.max_iterations)
            strategy = self.search_strategy_model.invoke(prompt)

            # Validate required fields exist
            for field in ["search_queries", "research_rationale", "expected_findings"]:
                if getattr(strategy, field, None) is None:
                    raise ValueError(f"Missing required field '{field}' in strategy response")

            return strategy.dict()

        except (json.JSONDecodeError, ValidationError, ValueError) as e:
            logger.warning(f"Search strategy parsing failed: {e}", exc_info=True)
            fallback = SearchStrategyResponse(
                search_queries=[context.user_question],
                research_rationale="Fallback to basic search due to parsing error",
                expected_findings="Basic information about the topic",
            )
            return fallback.dict()
        except Exception as e:
            logger.error("Error generating search strategy", exc_info=True)
            raise

    def extract_key_concepts(self, content: str) -> List[str]:
        """Extract key concepts from research content"""
        if not self.config.enable_concept_extraction:
            return []

        try:
            prompt = self.prompts.concept_extraction_prompt(content)
            concepts = self.key_concepts_model.invoke(prompt)
            return concepts.key_concepts if hasattr(concepts, 'key_concepts') else []
        except (json.JSONDecodeError, ValidationError) as e:
            logger.warning(f"Concept extraction parsing failed: {e}", exc_info=True)
            return []
        except Exception as e:
            logger.error("Error extracting concepts", exc_info=True)
            return []

    def update_summary(self, context: ResearchContext, new_information: str) -> str:
        """Update the comprehensive summary with new research findings"""
        if not new_information.strip():
            return context.current_summary or ""

        try:
            prompt = self.prompts.comprehensive_summary_prompt(context, new_information)
            updated_obj = self.comprehensive_summary_model.invoke(prompt)
            return updated_obj.main_answer.strip() if hasattr(updated_obj, 'main_answer') else context.current_summary or ""
        except (ValidationError, json.JSONDecodeError) as e:
            logger.warning(f"Summary update parsing failed: {e}", exc_info=True)
            return context.current_summary or ""
        except Exception as e:
            logger.error("Error updating summary", exc_info=True)
            return context.current_summary or ""

    def assess_research_completeness(self, context: ResearchContext) -> ResearchAssessment:
        """Assess whether research is complete enough to answer the question"""
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

            return ResearchAssessment(
                should_continue=data.should_continue,
                completeness_score=score,
                reasoning=data.reasoning,
                missing_aspects=data.missing_aspects,
                recommended_searches=data.recommended_next_searches,
                confidence_level=confidence,
            )
        except (ValidationError, json.JSONDecodeError) as e:
            logger.warning(f"Research completeness parsing failed: {e}", exc_info=True)
            return ResearchAssessment(
                should_continue=context.iteration_count < 3,
                completeness_score=50,
                reasoning="Assessment parsing failed, using fallback logic",
                missing_aspects=[],
                recommended_searches=[],
                confidence_level=ConfidenceLevel.LOW,
            )
        except Exception as e:
            logger.error("Error assessing research completeness", exc_info=True)
            raise

    def validate_source(self, source_url: str, content: str) -> Dict[str, Any]:
        """Validate the credibility and relevance of a source"""
        if not self.config.enable_source_validation:
            return {"overall_quality": 7, "recommendation": "include"}

        try:
            prompt = self.prompts.source_validation_prompt(source_url, content)
            validation = self.source_validation_model.invoke(prompt)
            return validation.dict()
        except (ValidationError, json.JSONDecodeError) as e:
            logger.warning(f"Source validation parsing failed: {e}", exc_info=True)
            return {"overall_quality": 7, "recommendation": "include", "reasoning": "Default due to parsing error"}
        except Exception as e:
            logger.error("Error validating source", exc_info=True)
            return {"overall_quality": 5, "recommendation": "review", "reasoning": str(e)}

    def generate_final_answer(self, context: ResearchContext) -> str:
        """Generate a final, polished answer based on all research"""
        try:
            prompt = self.prompts.final_answer_prompt(context)
            final_obj = self.final_answer_model.invoke(prompt)
            return final_obj.final_answer.strip() if hasattr(final_obj, 'final_answer') else context.current_summary or ""
        except (ValidationError, json.JSONDecodeError) as e:
            logger.warning(f"Final answer parsing failed: {e}", exc_info=True)
            return context.current_summary or ""
        except Exception as e:
            logger.error("Error generating final answer", exc_info=True)
            return context.current_summary or ""

    def handle_error_recovery(self, error_context: str, user_question: str) -> Dict[str, Any]:
        """Generate alternative approaches when errors occur"""
        try:
            prompt = self.prompts.error_recovery_prompt(error_context, user_question)
            recovery = self.error_recovery_model.invoke(prompt)
            return recovery.dict()
        except (ValidationError, json.JSONDecodeError) as e:
            logger.warning(f"Error recovery parsing failed: {e}", exc_info=True)
            fallback = ErrorRecoveryResponse(
                alternatives=[ErrorRecoveryStrategy(strategy="Basic keyword search", queries=[user_question])],
                explanation="Fallback due to error recovery failure",
            )
            return fallback.dict()
        except Exception as e:
            logger.error("Error in error recovery", exc_info=True)
            fallback = ErrorRecoveryResponse(
                alternatives=[ErrorRecoveryStrategy(strategy="Basic keyword search", queries=[user_question])],
                explanation="Fallback due to error recovery failure",
            )
            return fallback.dict()

    def refine_query(self, original_query: str, previous_results: List[str]) -> List[str]:
        """Refine search queries based on previous results"""
        try:
            prompt = self.prompts.query_refinement_prompt(original_query, previous_results)
            refined = self.query_refinement_model.invoke(prompt)
            return refined.refined_queries if hasattr(refined, 'refined_queries') else [original_query]
        except (ValidationError, json.JSONDecodeError) as e:
            logger.warning(f"Query refinement parsing failed: {e}", exc_info=True)
            return [original_query]
        except Exception as e:
            logger.warning(f"Query refinement failed: {e}", exc_info=True)
            return [original_query]
