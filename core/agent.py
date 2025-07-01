"""
Main research agent class that orchestrates the entire research process.
"""

import logging
from typing import List, Dict, Any
from datetime import datetime

from research_agent.core.models import (
    ResearchContext, ResearchResult, ResearchConfig, 
    ResearchStatus, IterationResult, SearchQuery
)
from research_agent.services.llm_service import LLMService
from research_agent.services.search_service import SearchService
from research_agent.utils.helpers import format_search_results, validate_input


logger = logging.getLogger(__name__)


class ResearchAgent:
    """
    Advanced iterative research agent that performs deep research on any topic
    by continuously building context and exploring new angles.
    """
    
    def __init__(self, config: ResearchConfig = None):
        self.config = config or ResearchConfig()
        self.llm_service = LLMService(self.config)
        self.search_service = SearchService(self.config)
        
        # Configure logging
        if self.config.verbose:
            logging.basicConfig(level=logging.INFO)
        
    def conduct_research(self, user_question: str) -> ResearchResult:
        """
        Main research method that conducts iterative deep research
        
        Args:
            user_question: The original question to research
            
        Returns:
            ResearchResult containing final summary, sources, and metadata
        """
        
        
        if self.config.verbose:
            print(f"🚀 Starting iterative research for: '{user_question}'")
            print("=" * 80)
        
        # Initialize research context
        context = ResearchContext(
            user_question=user_question,
            status=ResearchStatus.IN_PROGRESS
        )
        
        try:
            # Main research loop
            while self._should_continue_research(context):
                context.iteration_count += 1
                
                if self.config.verbose:
                    print(f"\n🔄 ITERATION {context.iteration_count}/{self.config.max_iterations}")
                    print("-" * 50)
                
                # Perform research iteration
                iteration_result = self._perform_research_iteration(context)
                context.research_history.append(iteration_result)
                
                # Check if we found new information
                if iteration_result.new_sources_count == 0:
                    if self.config.verbose:
                        print("⚠️ No new sources found. Ending research.")
                    break
                
                # Update context with new findings
                self._update_research_context(context, iteration_result)
                
                # Assess research completeness
                assessment = self.llm_service.assess_research_completeness(context)
                context.last_assessment = assessment
                
                if self.config.verbose:
                    print(f"📊 Completeness: {assessment.completeness_score:.1f}% - {assessment.reasoning}")
                
                # Check stopping conditions
                if not assessment.should_continue or assessment.completeness_score >= self.config.min_completeness_score:
                    if self.config.verbose:
                        print(f"✅ Research complete: {assessment.reasoning}")
                    break
            
            # Finalize research
            context.status = ResearchStatus.COMPLETED
            context.end_time = datetime.now()
            
            if self.config.verbose:
                print("\n" + "=" * 80)
                print("🎉 RESEARCH COMPLETED")
                print("=" * 80)
            
            return ResearchResult.from_context(context)
            
        except Exception as e:
            logger.error(f"Research failed: {e}")
            context.status = ResearchStatus.FAILED
            context.end_time = datetime.now()
            
            # Return partial results even on failure
            result = ResearchResult.from_context(context)
            result.metadata["error"] = str(e)
            return result
    
    def _should_continue_research(self, context: ResearchContext) -> bool:
        """Determine if research should continue"""
        # Check maximum iterations
        if context.iteration_count >= self.config.max_iterations:
            return False
        
        # Always do at least one iteration
        if context.iteration_count == 0:
            return True
        
        # Check if we have assessment results
        if context.last_assessment:
            return context.last_assessment.should_continue
        
        # Default: continue if we haven't done much research yet
        return context.iteration_count < 3
    
    def _perform_research_iteration(self, context: ResearchContext) -> IterationResult:
        """Perform a single research iteration"""
        iteration_start = datetime.now()
        
        # Generate search strategy
        strategy = self.llm_service.generate_search_strategy(context)
        
        # Create search queries
        search_queries = []
        for query_text in strategy["search_queries"]:
            search_query = SearchQuery(
                query=query_text,
                rationale=strategy["research_rationale"],
                expected_results=strategy["expected_findings"],
                iteration=context.iteration_count
            )
            search_queries.append(search_query)
        
        # Execute searches
        search_results_by_query = self.search_service.execute_multiple_searches(
            [q.query for q in search_queries]
        )
        
        # Process and filter results
        new_search_results = []
        new_sources_count = 0
        
        for query, results in search_results_by_query.items():
            for result in results:
                # Check if this is a new source
                if context.add_search_result(result):
                    new_search_results.append(result)
                    new_sources_count += 1
        
        # Extract key concepts from new content
        new_concepts = []
        if new_search_results:
            combined_content = "\n".join([r.content for r in new_search_results])
            new_concepts = self.llm_service.extract_key_concepts(combined_content)
            context.key_concepts_found.extend([c for c in new_concepts if c not in context.key_concepts_found])
        
        # Create iteration result
        iteration_result = IterationResult(
            iteration_number=context.iteration_count,
            search_queries=search_queries,
            search_results=new_search_results,
            new_sources_count=new_sources_count,
            summary_length=len(context.current_summary),
            key_concepts_found=new_concepts,
            timestamp=iteration_start
        )
        
        return iteration_result
    
    def _update_research_context(self, context: ResearchContext, iteration_result: IterationResult):
        """Update research context with new iteration results"""
        if not iteration_result.search_results:
            return
        
        # Format new information for summary update
        new_information = format_search_results(iteration_result.search_results)
        
        # Update summary with new findings
        context.current_summary = self.llm_service.update_summary(context, new_information)
        
        if self.config.verbose:
            print(f"📚 Added {iteration_result.new_sources_count} new sources")
            print(f"🧠 Found {len(iteration_result.key_concepts_found)} new concepts")
    
    def get_research_statistics(self, result: ResearchResult) -> Dict[str, Any]:
        """Get detailed statistics about the research process"""
        if not result.research_history:
            return {"error": "No research history available"}
        
        return {
            "research_overview": {
                "total_iterations": result.iterations_completed,
                "total_sources": result.total_sources,
                "duration_seconds": result.duration_seconds,
                "final_status": result.status.value,
                "final_completeness_score": result.final_assessment.completeness_score if result.final_assessment else None
            },
            "iteration_breakdown": [
                {
                    "iteration": iter_result.iteration_number,
                    "new_sources": iter_result.new_sources_count,
                    "concepts_found": len(iter_result.key_concepts_found),
                    "queries_executed": len(iter_result.search_queries)
                }
                for iter_result in result.research_history
            ],
            "concept_discovery": {
                "total_concepts": len(result.key_concepts_discovered),
                "key_concepts": result.key_concepts_discovered[:10]  # Top 10
            },
            "search_efficiency": {
                "sources_per_iteration": result.total_sources / result.iterations_completed if result.iterations_completed else 0,
                "unique_sources": len(set(result.sources_used))
            }
        }
    
    def export_research_report(self, result: ResearchResult, format: str = "markdown") -> str:
        """Export a comprehensive research report"""
        if format.lower() == "markdown":
            return self._generate_markdown_report(result)
        elif format.lower() == "json":
            import json
            return json.dumps(result.__dict__, default=str, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _generate_markdown_report(self, result: ResearchResult) -> str:
        """Generate a markdown research report"""
        report = f"""# Research Report

## Original Question
{result.original_question}

## Executive Summary
{result.final_summary}

## Research Statistics
- **Total Sources Consulted**: {result.total_sources}
- **Research Iterations**: {result.iterations_completed}
- **Research Duration**: {result.duration_seconds:.1f} seconds
- **Key Concepts Discovered**: {len(result.key_concepts_discovered)}

## Key Concepts
{', '.join(result.key_concepts_discovered[:20])}

## Sources
"""
        
        for i, source in enumerate(result.sources_used, 1):
            report += f"{i}. {source}\n"
        
        if result.final_assessment:
            report += f"""
## Research Assessment
- **Completeness Score**: {result.final_assessment.completeness_score:.1f}%
- **Confidence Level**: {result.final_assessment.confidence_level.value}
- **Assessment**: {result.final_assessment.reasoning}
"""
        
        return report