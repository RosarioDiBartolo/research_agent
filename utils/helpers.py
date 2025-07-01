"""
Utility functions and helpers for the research agent.
"""

import re
import logging
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse
from datetime import datetime

from core.models import SearchResult


logger = logging.getLogger(__name__)


def format_search_results(results: List[SearchResult]) -> str:
    """Format search results into a readable string for LLM processing"""
    if not results:
        return ""
    
    formatted_results = []
    
    for result in results:
        formatted_content = f"""
**{result.title}**
{result.content}
(Source: {result.url})
Relevance: {result.relevance_score:.1f}%
"""
        formatted_results.append(formatted_content.strip())
    
    return "\n\n---\n\n".join(formatted_results)


def extract_domain_from_url(url: str) -> Optional[str]:
    """Extract domain name from URL"""
    try:
        parsed = urlparse(url)
        return parsed.netloc.lower()
    except Exception:
        return None


def is_academic_source(url: str) -> bool:
    """Check if a URL appears to be from an academic source"""
    academic_domains = [
        '.edu', '.ac.', 'scholar.google', 'pubmed', 'arxiv', 
        'jstor', 'springer', 'wiley', 'elsevier', 'nature.com',
        'science.org', 'pnas.org', 'cell.com'
    ]
    
    url_lower = url.lower()
    return any(domain in url_lower for domain in academic_domains)


def is_government_source(url: str) -> bool:
    """Check if a URL appears to be from a government source"""
    gov_domains = [
        '.gov', '.mil', 'europa.eu', 'un.org', 'who.int', 
        'worldbank.org', 'imf.org', 'oecd.org'
    ]
    
    url_lower = url.lower()
    return any(domain in url_lower for domain in gov_domains)


def is_news_source(url: str) -> bool:
    """Check if a URL appears to be from a news source"""
    news_domains = [
        'reuters.com', 'bbc.com', 'cnn.com', 'nytimes.com', 
        'washingtonpost.com', 'wsj.com', 'ft.com', 'bloomberg.com',
        'apnews.com', 'npr.org', 'theguardian.com'
    ]
    
    url_lower = url.lower()
    return any(domain in url_lower for domain in news_domains)


def categorize_source(url: str) -> str:
    """Categorize a source based on its URL"""
    if is_academic_source(url):
        return "academic"
    elif is_government_source(url):
        return "government"
    elif is_news_source(url):
        return "news"
    else:
        return "other"


def clean_text(text: str) -> str:
    """Clean and normalize text content"""
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove HTML tags if any
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove URLs from text content
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    return text.strip()


def extract_key_phrases(text: str, max_phrases: int = 10) -> List[str]:
    """Extract key phrases from text using simple heuristics"""
    if not text:
        return []
    
    # Clean text
    clean = clean_text(text)
    
    # Find phrases in quotes
    quoted_phrases = re.findall(r'"([^"]+)"', clean)
    
    # Find capitalized phrases (potential proper nouns)
    capitalized_phrases = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', clean)
    
    # Combine and deduplicate
    phrases = list(set(quoted_phrases + capitalized_phrases))
    
    # Filter out very short or very long phrases
    phrases = [p for p in phrases if 2 <= len(p.split()) <= 5]
    
    return phrases[:max_phrases]


def validate_input(user_input: str) -> bool:
    """Validate user input for basic safety and format checks"""
    if not user_input or not isinstance(user_input, str):
        return False
    
    # Check minimum length
    if len(user_input.strip()) < 3:
        return False
    
    # Check for suspicious patterns
    suspicious_patterns = [
        r'<script',
        r'javascript:',
        r'eval\(',
        r'exec\(',
    ]
    
    for pattern in suspicious_patterns:
        if re.search(pattern, user_input, re.IGNORECASE):
            return False
    
    return True


def estimate_reading_time(text: str, words_per_minute: int = 200) -> float:
    """Estimate reading time for text in minutes"""
    if not text:
        return 0.0
    
    word_count = len(text.split())
    return word_count / words_per_minute


def truncate_text(text: str, max_length: int = 1000, preserve_words: bool = True) -> str:
    """Truncate text to specified length"""
    if not text or len(text) <= max_length:
        return text
    
    if preserve_words:
        # Find the last space before max_length
        truncate_point = text.rfind(' ', 0, max_length)
        if truncate_point == -1:
            truncate_point = max_length
        return text[:truncate_point] + "..."
    else:
        return text[:max_length] + "..."


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable format"""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.1f} hours"


def merge_overlapping_concepts(concepts: List[str]) -> List[str]:
    """Merge overlapping or very similar concepts"""
    if not concepts:
        return []
    
    # Simple deduplication by lowercase comparison
    unique_concepts = []
    seen_lower = set()
    
    for concept in concepts:
        concept_lower = concept.lower()
        if concept_lower not in seen_lower:
            seen_lower.add(concept_lower)
            unique_concepts.append(concept)
    
    return unique_concepts


def calculate_text_similarity(text1: str, text2: str) -> float:
    """Calculate simple text similarity based on word overlap"""
    if not text1 or not text2:
        return 0.0
    
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    intersection = words1 & words2
    union = words1 | words2
    
    if not union:
        return 0.0
    
    return len(intersection) / len(union)


def group_sources_by_domain(sources: List[str]) -> Dict[str, List[str]]:
    """Group sources by their domain"""
    domain_groups = {}
    
    for source in sources:
        domain = extract_domain_from_url(source)
        if domain:
            if domain not in domain_groups:
                domain_groups[domain] = []
            domain_groups[domain].append(source)
    
    return domain_groups


def generate_search_suggestions(question: str, existing_concepts: List[str]) -> List[str]:
    """Generate search suggestions based on question and existing concepts"""
    suggestions = []
    
    # Add basic question variants
    suggestions.append(f"{question} explanation")
    suggestions.append(f"{question} definition")
    suggestions.append(f"{question} examples")
    
    # Add concept-based searches
    for concept in existing_concepts[:3]:  # Limit to top 3 concepts
        suggestions.append(f"{concept} {question}")
        suggestions.append(f"{concept} research")
    
    return suggestions[:5]  # Return top 5 suggestions


def create_research_timeline(research_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Create a timeline of research activities"""
    timeline = []
    
    for iteration in research_history:
        timeline.append({
            "timestamp": iteration.get("timestamp", datetime.now()),
            "iteration": iteration.get("iteration", 0),
            "activity": f"Research iteration {iteration.get('iteration', 0)}",
            "sources_found": iteration.get("new_sources_count", 0),
            "concepts_discovered": len(iteration.get("key_concepts_found", []))
        })
    
    return sorted(timeline, key=lambda x: x["timestamp"])