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
logging.basicConfig(level=logging.INFO)


class LLMService:
    """Service for managing all interactions with language models"""

    def __init__(self, model: BaseChatModel, config: ResearchConfig):
        logger.info("Initializing LLMService...")
        self.config = config
        self.llm = model

        self.search_strategy_model = self.llm.with_structured_output(SearchStrategyResponse)
        self.key_concepts_model = self.llm.with_structured_output(ConceptExtractionResponse)
        self.research_completeness_model = self.llm.with_structured_output(ResearchCompletenessResponse)
        self.source_validation_model = self.llm.with_structured_output(SourceValidationResponse)
        self.error_recovery_model = self.llm.with_structured_output(ErrorRecoveryResponse)
        self.query_refinement_model = self.llm.with_structured_output(QueryRefinementResponse)
        self.comprehensive_summary_model = self.llm.with_structured_output(ComprehensiveSummaryResponse)

        try:
            logger.debug("Testing LLM connectivity...")
            self.llm.invoke("Simple test")
            logger.info("LLM connection established successfully.")
        except Exception as e:
            logger.error("Failed to connect to LLM", exc_info=True)
            raise RuntimeError("LLM connection failed, check your API key and model configuration.") from e

        self.prompts = PromptTemplates()

    def invoke_json(self, prompt: str) -> Dict[str, Any]:
        logger.debug("Invoking LLM with JSON prompt.")
        try:
            logger.debug(f"Prompt: {prompt}")
            response = self.llm.invoke(prompt)
            content = response.content.replace("```json", "").replace("```", "").strip()
            logger.debug(f"LLM response content: {content}")
            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.error("Failed to parse JSON response", exc_info=True)
            raise ValueError("LLM response is not valid JSON") from e
        except Exception as e:
            logger.error("Error invoking LLM", exc_info=True)
            raise RuntimeError("LLM invocation failed") from e

    def generate_search_strategy(self, context: ResearchContext) -> Dict[str, Any]:
        logger.info("Generating search strategy...")
        try:
            prompt = self.prompts.search_strategy_prompt(context, self.config.max_iterations)
            logger.debug(f"Search strategy prompt: {prompt}")
            strategy = self.search_strategy_model.invoke(prompt)
            logger.debug(f"Search strategy model output: {strategy}")

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
        logger.info("Extracting key concepts...")
        if not self.config.enable_concept_extraction:
            logger.info("Concept extraction disabled in config.")
            return []

        try:
            prompt = self.prompts.concept_extraction_prompt(content)
            logger.debug(f"Concept extraction prompt: {prompt}")
            concepts = self.key_concepts_model.invoke(prompt)
            logger.debug(f"Extracted concepts: {concepts}")
            return concepts.key_concepts if hasattr(concepts, 'key_concepts') else []
        except (json.JSONDecodeError, ValidationError) as e:
            logger.warning(f"Concept extraction parsing failed: {e}", exc_info=True)
            return []
        except Exception as e:
            logger.error("Error extracting concepts", exc_info=True)
            return []

    def update_summary(self, context: ResearchContext, new_information: str) -> str:
        logger.info("Updating summary...")
        if not new_information.strip():
            logger.debug("No new information provided. Returning current summary.")
            return context.current_summary or ""

        try:
            prompt = self.prompts.comprehensive_summary_prompt(context, new_information)
            logger.debug(f"Comprehensive summary prompt: {prompt}")
            updated_obj = self.comprehensive_summary_model.invoke(prompt)
            logger.debug(f"Updated summary object: {updated_obj}")
            return updated_obj.main_answer.strip() if hasattr(updated_obj, 'main_answer') else context.current_summary or ""
        except (ValidationError, json.JSONDecodeError) as e:
            logger.warning(f"Summary update parsing failed: {e}", exc_info=True)
            return context.current_summary or ""
        except Exception as e:
            logger.error("Error updating summary", exc_info=True)
            return context.current_summary or ""

    def assess_research_completeness(self, context: ResearchContext) -> ResearchAssessment:
        logger.info("Assessing research completeness...")
        try:
            prompt = self.prompts.research_completeness_prompt(context)
            logger.debug(f"Research completeness prompt: {prompt}")
            data = self.research_completeness_model.invoke(prompt)
            logger.debug(f"Completeness data: {data}")
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
            logger.debug(f"Assessment result: {assessment}")
            return assessment
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
        logger.info(f"Validating source: {source_url}")
        if not self.config.enable_source_validation:
            logger.debug("Source validation disabled in config.")
            return {"overall_quality": 7, "recommendation": "include"}

        try:
            prompt = self.prompts.source_validation_prompt(source_url, content)
            logger.debug(f"Source validation prompt: {prompt}")
            validation = self.source_validation_model.invoke(prompt)
            logger.debug(f"Validation result: {validation}")
            return validation.dict()
        except (ValidationError, json.JSONDecodeError) as e:
            logger.warning(f"Source validation parsing failed: {e}", exc_info=True)
            return {"overall_quality": 7, "recommendation": "include", "reasoning": "Default due to parsing error"}
        except Exception as e:
            logger.error("Error validating source", exc_info=True)
            return {"overall_quality": 5, "recommendation": "review", "reasoning": str(e)}

    def generate_final_answer(self, context: ResearchContext) -> str:
        logger.info("Generating final answer...")
        try:
            prompt = self.prompts.final_answer_prompt(context)
            logger.debug(f"Final answer prompt: {prompt}")
            final_answer = self.llm.invoke(prompt)
            logger.debug(f"Final answer: {final_answer}")
            return final_answer.strip()
        except (ValidationError, json.JSONDecodeError) as e:
            logger.warning(f"Final answer parsing failed: {e}", exc_info=True)
            return context.current_summary or ""
        except Exception as e:
            logger.error("Error generating final answer", exc_info=True)
            return context.current_summary or ""

    def handle_error_recovery(self, error_context: str, user_question: str) -> Dict[str, Any]:
        logger.info("Handling error recovery...")
        try:
            prompt = self.prompts.error_recovery_prompt(error_context, user_question)
            logger.debug(f"Error recovery prompt: {prompt}")
            recovery = self.error_recovery_model.invoke(prompt)
            logger.debug(f"Recovery strategy: {recovery}")
            return recovery.dict()
        except (ValidationError, json.JSONDecodeError) as e:
            logger.warning(f"Error recovery parsing failed: {e}", exc_info=True)
        except Exception as e:
            logger.error("Error in error recovery", exc_info=True)

        fallback = ErrorRecoveryResponse(
            alternatives=[ErrorRecoveryStrategy(strategy="Basic keyword search", queries=[user_question])],
            explanation="Fallback due to error recovery failure",
        )
        logger.debug(f"Returning fallback recovery strategy: {fallback}")
        return fallback.dict()

    def refine_query(self, original_query: str, previous_results: List[str]) -> List[str]:
        logger.info("Refining query...")
        try:
            prompt = self.prompts.query_refinement_prompt(original_query, previous_results)
            logger.debug(f"Query refinement prompt: {prompt}")
            refined = self.query_refinement_model.invoke(prompt)
            logger.debug(f"Refined queries: {refined}")
            return refined.refined_queries if hasattr(refined, 'refined_queries') else [original_query]
        except (ValidationError, json.JSONDecodeError) as e:
            logger.warning(f"Query refinement parsing failed: {e}", exc_info=True)
            return [original_query]
        except Exception as e:
            logger.warning(f"Query refinement failed: {e}", exc_info=True)
            return [original_query]
