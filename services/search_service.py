"""
Search service for handling web search operations.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, TimeoutError

from langchain_tavily import TavilySearch

from core.schemas import SearchResult,  ResearchConfig


logger = logging.getLogger(__name__)


class SearchService:
    """Service for managing web search operations"""
    
    def __init__(self, config: ResearchConfig):
        self.config = config
        self.search_tool = TavilySearch(
            max_results=config.max_search_results_per_query
        )
        self.executor = ThreadPoolExecutor(max_workers=3)
        
    def execute_search(self, query: str, timeout: Optional[int] = None) -> List[Dict[str, Any]]:
        """Execute a single search query with timeout"""
        timeout = timeout or self.config.search_timeout
        
        try:
            
            # Use ThreadPoolExecutor for timeout control
            future = self.executor.submit(self.search_tool.run, query)
            results = future.result(timeout=timeout)["results"]
            if isinstance(results, list):
                return results
            elif isinstance(results, str):
                # Handle case where tool returns string instead of list
                return [{"content": results, "url": "", "title": "Search Result"}]
            else:
                return []
                
        except TimeoutError:
            logger.warning(f"Search timeout for query: {query}")
            return []
        except Exception as e:
            logger.error(f"Search error for query '{query}': {e}")
            return []
    
    def execute_multiple_searches(self, queries: List[str]) -> Dict[str, List[SearchResult]]:
        """Execute multiple search queries and return organized results"""
        all_results = {}
        
        for query in queries:
            if self.config.verbose:
                print(f"🔍 Searching: {query}")
            

            raw_results = self.execute_search(query)

            
            processed_results = self._process_search_results(raw_results, query)

           
            

            all_results[query] = processed_results
            
        return all_results
    
    def _process_search_results(self, raw_results: List[Dict[str, Any]], query: str) -> List[SearchResult]:
        """Process raw search results into SearchResult objects"""
        processed_results = []
        
        for result in raw_results:
            try:
                search_result = self._create_search_result(result, query)
                if search_result:
                    processed_results.append(search_result)
            except Exception as e:
                logger.warning(f"Error processing search result: {e}")
                continue
                
        return processed_results
    
    def _create_search_result(self, raw_result: Dict[str, Any], query: str) -> Optional[SearchResult]:
        """Create a SearchResult object from raw search data"""
        # Handle different possible field names from different search tools
        url = (raw_result.get('url') or 
               raw_result.get('link') or 
               raw_result.get('href') or 
               "")
        
        title = (raw_result.get('title') or 
                raw_result.get('name') or 
                "Untitled")
        
        content = (raw_result.get('content') or 
                  raw_result.get('body') or 
                  raw_result.get('text') or 
                  "")
        
        snippet = (raw_result.get('snippet') or 
                  raw_result.get('summary') or 
                  content[:200] + "..." if len(content) > 200 else content)
        
        # Skip results without URL or content
        if not url or not (content or snippet):
            return None
            
        # Calculate relevance score based on query match
        relevance_score = self._calculate_relevance_score(query, title, content, snippet)
        
        return SearchResult(
            url=url,
            title=title,
            content=content,
            snippet=snippet,
            relevance_score=relevance_score,
            timestamp=datetime.now()
        )
    
    def _calculate_relevance_score(self, query: str, title: str, content: str, snippet: str) -> float:
        """Calculate a simple relevance score based on keyword matching"""
        query_words = set(query.lower().split())
        
        # Combine all text for analysis
        all_text = f"{title} {content} {snippet}".lower()
        
        # Count query word matches
        matches = sum(1 for word in query_words if word in all_text)
        
        # Calculate score as percentage of query words found
        if query_words:
            score = (matches / len(query_words)) * 100
        else:
            score = 0
            
        return min(score, 100.0)
    
    def filter_results_by_quality(self, results: List[SearchResult], min_score: float = 0.0) -> List[SearchResult]:
        """Filter search results based on relevance score"""
        return [result for result in results if result.relevance_score >= min_score]
    
    def deduplicate_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Remove duplicate results based on URL"""
        seen_urls = set()
        unique_results = []
        
        for result in results:
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                unique_results.append(result)
                
        return unique_results
    
    def get_search_statistics(self, results: Dict[str, List[SearchResult]]) -> Dict[str, Any]:
        """Get statistics about search results"""
        total_results = sum(len(result_list) for result_list in results.values())
        
        if total_results == 0:
            return {
                "total_results": 0,
                "average_relevance": 0.0,
                "queries_executed": len(results),
                "results_per_query": 0.0
            }
        
        all_results = []
        for result_list in results.values():
            all_results.extend(result_list)
            
        average_relevance = sum(r.relevance_score for r in all_results) / len(all_results)
        
        return {
            "total_results": total_results,
            "average_relevance": average_relevance,
            "queries_executed": len(results),
            "results_per_query": total_results / len(results) if results else 0,
            "unique_sources": len(set(r.url for r in all_results if r.url))
        }