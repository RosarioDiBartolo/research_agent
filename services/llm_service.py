"""
Language Model service for handling all LLM interactions.
"""

import json
import logging
from typing import Dict, List, Optional, Any
from langchain.chat_models import ChatOpenAI
from langchain.schema import BaseMessage, HumanMessage, SystemMessage

from core.models import ResearchConfig, ResearchAssessment, ConfidenceLevel
from utils.prompts import PromptTemplates


logger = logging.getLogger(__name__)


class LLMService:
    """Service for managing all interactions with language models"""
    
    def __init__(self, config: ResearchConfig):
        self.config = config
        self.llm = ChatOpenAI(
            model_name=config.model_name,
            temperature=config.temperature
        )
        self.prompts = PromptTemplates()
        
    def generate_search_strategy(self, context) -> Dict[str, Any]:
        """Generate strategic search queries based on research context"""
        try:
            prompt = self.prompts.search_strategy_prompt(context, self.config.max_iterations)
            response = self.llm.predict(prompt)
            
            strategy = json.loads(response)
            
            # Validate required fields
            required_fields = ["search_queries", "research_rationale", "expected_findings"]
            if not all(field in strategy for field in required_fields):
                raise ValueError("Missing required fields in strategy response")
                
            return strategy
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse search strategy JSON: {e}")
            # Fallback strategy
            return {
                "search_queries": [context.user_question],
                "research_rationale": "Fallback to basic search due to parsing error",
                "expected_findings": "Basic information about the topic"
            }
        except Exception as e:
            logger.error(f"Error generating search strategy: {e}")
            raise
    
    def extract_key_concepts(self, content: str) -> List[str]:
        """Extract key concepts from research content"""
        if not self.config.enable_concept_extraction:
            return []
            
        try:
            prompt = self.prompts.concept_extraction_prompt(content)
            response = self.llm.predict(prompt)
            
            concepts = json.loads(response)
            
            if isinstance(concepts, list):
                return [str(concept) for concept in concepts]
            else:
                logger.warning("Concept extraction didn't return a list")
                return []
                
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse concept extraction JSON: {e}")
            return []
        except Exception as e:
            logger.error(f"Error extracting concepts: {e}")
            return []
    
    def update_summary(self, context, new_information: str) -> str:
        """Update the comprehensive summary with new research findings"""
        if not new_information.strip():
            return context.current_summary
            
        try:
            prompt = self.prompts.comprehensive_summary_prompt(context, new_information)
            updated_summary = self.llm.predict(prompt)
            return updated_summary.strip()
            
        except Exception as e:
            logger.error(f"Error updating summary: {e}")
            # Return existing summary if update fails
            return context.current_summary
    
    def assess_research_completeness(self, context) -> ResearchAssessment:
        """Assess whether research is complete enough to answer the question"""
        try:
            prompt = self.prompts.research_completeness_prompt(context)
            response = self.llm.predict(prompt)
            
            assessment_data = json.loads(response)
            
            # Convert to ResearchAssessment object
            confidence_map = {
                "low": ConfidenceLevel.LOW,
                "medium": ConfidenceLevel.MEDIUM,
                "high": ConfidenceLevel.HIGH
            }
            
            # Determine confidence level based on completeness score
            score = assessment_data.get("completeness_score", 0)
            if score >= 80:
                confidence = ConfidenceLevel.HIGH
            elif score >= 60:
                confidence = ConfidenceLevel.MEDIUM
            else:
                confidence = ConfidenceLevel.LOW
            
            return ResearchAssessment(
                should_continue=assessment_data.get("should_continue", False),
                completeness_score=score,
                reasoning=assessment_data.get("reasoning", ""),
                missing_aspects=assessment_data.get("missing_aspects", []),
                recommended_searches=assessment_data.get("recommended_next_searches", []),
                confidence_level=confidence
            )
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse assessment JSON: {e}")
            # Fallback assessment
            return ResearchAssessment(
                should_continue=context.iteration_count < 3,
                completeness_score=50.0,
                reasoning="Assessment parsing failed, using fallback logic",
                confidence_level=ConfidenceLevel.LOW
            )
        except Exception as e:
            logger.error(f"Error assessing research completeness: {e}")
            raise
    
    def validate_source(self, source_url: str, content: str) -> Dict[str, Any]:
        """Validate the credibility and relevance of a source"""
        if not self.config.enable_source_validation:
            return {
                "overall_quality": 7,  # Default acceptable quality
                "recommendation": "include"
            }
            
        try:
            prompt = self.prompts.source_validation_prompt(source_url, content)
            response = self.llm.predict(prompt)
            
            validation = json.loads(response)
            return validation
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse source validation JSON: {e}")
            return {
                "overall_quality": 7,
                "recommendation": "include",
                "reasoning": "Default acceptance due to parsing error"
            }
        except Exception as e:
            logger.error(f"Error validating source: {e}")
            return {
                "overall_quality": 5,
                "recommendation": "review",
                "reasoning": f"Validation error: {str(e)}"
            }
    
    def generate_final_answer(self, context) -> str:
        """Generate a final, polished answer based on all research"""
        try:
            prompt = self.prompts.final_answer_prompt(context)
            final_answer = self.llm.predict(prompt)
            return final_answer.strip()
            
        except Exception as e:
            logger.error(f"Error generating final answer: {e}")
            # Fallback to current summary
            return context.current_summary
    
    def handle_error_recovery(self, error_context: str, user_question: str) -> Dict[str, Any]:
        """Generate alternative approaches when errors occur"""
        try:
            prompt = self.prompts.error_recovery_prompt(error_context, user_question)
            response = self.llm.predict(prompt)
            
            recovery_plan = json.loads(response)
            return recovery_plan
            
        except Exception as e:
            logger.error(f"Error in error recovery: {e}")
            return {
                "alternatives": [
                    {
                        "strategy": "Basic keyword search",
                        "queries": [user_question]
                    }
                ],
                "explanation": "Fallback to simple search due to error recovery failure"
            }
    
    def refine_query(self, original_query: str, previous_results: List[str]) -> List[str]:
        """Refine search queries based on previous results"""
        try:
            prompt = self.prompts.query_refinement_prompt(original_query, previous_results)
            response = self.llm.predict(prompt)
            
            refined_queries = json.loads(response)
            
            if isinstance(refined_queries, list):
                return refined_queries
            else:
                return [original_query]
                
        except Exception as e:
            logger.warning(f"Query refinement failed: {e}")
            return [original_query]