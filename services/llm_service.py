"""
Language Model service for handling all LLM interactions.
"""

import json
import logging
import os
from typing import Dict, List, Optional, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import BaseMessage, HumanMessage, SystemMessage

from constant import GeminiKey
from core.models import ResearchConfig, ResearchAssessment, ConfidenceLevel
from utils.prompts import PromptTemplates
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)


class LLMService:
    """Service for managing all interactions with language models"""
    
    def __init__(self, config: ResearchConfig):
        self.config = config
        self.llm =   ChatGoogleGenerativeAI(
          model=config.model_name ,  # Adjust if model name differs per provider
          google_api_key=GeminiKey,  # Use the Gemini API key
     
            temperature=config.temperature
        )
        try:
            self.llm.invoke("Simple test")
        except Exception as e:
            logger.error(f"Failed to connect to LLM: {e}")
            raise RuntimeError("LLM connection failed, check your API key and model configuration.")
        
        self.prompts = PromptTemplates()
    
    def invoke_json(self, prompt: str) -> Dict[str, Any]:
        """Invoke the LLM with a JSON prompt and return parsed response"""
        try:
            response = self.llm.invoke(prompt)
            content = response.content.replace("```json", "").replace("```", "").strip()
            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            raise ValueError("LLM response is not valid JSON")
        except Exception as e:
            logger.error(f"Error invoking LLM: {e}")
            raise RuntimeError("LLM invocation failed")
        
    def generate_search_strategy(self, context) -> Dict[str, Any]:
        """Generate strategic search queries based on research context"""
        try:
            prompt = self.prompts.search_strategy_prompt(context, self.config.max_iterations)
            strategy = self.invoke_json(prompt)
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
             
            concepts = self.invoke_json(prompt)

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
            updated_summary = self.llm.invoke(prompt).content
            return updated_summary.strip()
            
        except Exception as e:
            logger.error(f"Error updating summary: {e}")
            # Return existing summary if update fails
            return context.current_summary
    
    def assess_research_completeness(self, context) -> ResearchAssessment:
        """Assess whether research is complete enough to answer the question"""
        try:
            prompt = self.prompts.research_completeness_prompt(context)
             
            assessment_data =  self.invoke_json(prompt)
            
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
            validation = self.invoke_json(prompt)
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
            final_answer = self.llm.invoke(prompt).content
            return final_answer.strip()
            
        except Exception as e:
            logger.error(f"Error generating final answer: {e}")
            # Fallback to current summary
            return context.current_summary
    
    def handle_error_recovery(self, error_context: str, user_question: str) -> Dict[str, Any]:
        """Generate alternative approaches when errors occur"""
        try:
            prompt = self.prompts.error_recovery_prompt(error_context, user_question)
             
            recovery_plan =  self.invoke_json(prompt)
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
             
            refined_queries = self.invoke_json(prompt)
            
            if isinstance(refined_queries, list):
                return refined_queries
            else:
                return [original_query]
                
        except Exception as e:
            logger.warning(f"Query refinement failed: {e}")
            return [original_query]