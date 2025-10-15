"""
Cypher-specific tokenization utilities for better model performance.
Handles database terminology, node labels, relationship types, and query patterns.
"""

import re
from typing import List, Dict, Any
from loguru import logger


# Common Cypher/Neo4j terminology that should be single tokens
CYPHER_TERMS = [
    # Cypher keywords
    "MATCH", "RETURN", "WHERE", "CREATE", "DELETE", "SET", "REMOVE",
    "WITH", "UNWIND", "ORDER BY", "LIMIT", "SKIP", "UNION", "DISTINCT",
    "CASE", "WHEN", "THEN", "ELSE", "END", "AND", "OR", "NOT", "XOR",
    "IS NULL", "IS NOT NULL", "STARTS WITH", "ENDS WITH", "CONTAINS",
    "IN", "EXISTS", "COLLECT", "COUNT", "SUM", "AVG", "MIN", "MAX",
    
    # Common node labels
    "Person", "Movie", "Actor", "Director", "User", "Product", "Company",
    "Location", "Country", "City", "State", "Address", "Phone", "Email",
    "Category", "Tag", "Review", "Rating", "Order", "Customer", "Supplier",
    
    # Common relationship types
    "ACTED_IN", "DIRECTED", "PRODUCED", "KNOWS", "FRIEND_OF", "FOLLOWS",
    "LIKES", "REVIEWED", "BOUGHT", "SOLD", "WORKS_FOR", "LIVES_IN",
    "BORN_IN", "MARRIED_TO", "PARENT_OF", "CHILD_OF", "SIBLING_OF",
    "OWNS", "MANAGES", "REPORTS_TO", "MEMBER_OF", "PART_OF", "CONTAINS",
    
    # Property names
    "name", "title", "age", "born", "died", "released", "rating", "votes",
    "price", "description", "email", "phone", "address", "city", "state",
    "country", "zip", "created", "updated", "active", "status", "type",
    
    # Functions and procedures
    "APOC", "algo", "gds", "timestamp", "datetime", "date", "time",
    "rand", "round", "floor", "ceil", "abs", "sign", "sqrt", "log",
    "size", "length", "substring", "replace", "split", "trim", "upper", "lower",
    
    # Data types and formats
    "INTEGER", "FLOAT", "STRING", "BOOLEAN", "LIST", "MAP", "NODE", "RELATIONSHIP",
    "PATH", "POINT", "DATE", "TIME", "DATETIME", "DURATION",
    
    # Common patterns
    "()-[]->()", "(n)-[r]->(m)", "(a:Person)", "[r:KNOWS]", "WHERE n.name",
    "RETURN n", "MATCH (n)", "CREATE (n)", "DELETE n", "SET n.property"
]

# Note: Pattern matching is handled directly in preprocess_text() function

def get_custom_tokens() -> List[str]:
    """Get all custom tokens that should be added to the tokenizer."""
    return CYPHER_TERMS

def preprocess_text(text: str) -> str:
    """
    Preprocess text to make it more tokenizer-friendly for Cypher generation.
    
    Args:
        text: Input text to preprocess
        
    Returns:
        Preprocessed text
    """
    logger.debug(f"Preprocessing text: {text[:100]}...")
    
    # Convert common patterns to be more readable
    # Convert node patterns for better tokenization
    text = re.sub(r'\((\w+):(\w+)\)', r'(\1 labeled \2)', text)
    text = re.sub(r'\((\w+)\)', r'(node \1)', text)
    
    # Convert relationship patterns
    text = re.sub(r'\[(\w+):(\w+)\]', r'[relationship \1 of type \2]', text)
    text = re.sub(r'\[(\w+)\]', r'[relationship \1]', text)
    
    # Convert property access patterns
    text = re.sub(r'(\w+)\.(\w+)', r'\1 property \2', text)
    
    # Convert comparison operators to words
    text = re.sub(r'>=', ' greater than or equal to ', text)
    text = re.sub(r'<=', ' less than or equal to ', text)
    text = re.sub(r'!=', ' not equal to ', text)
    text = re.sub(r'<>', ' not equal to ', text)
    text = re.sub(r'>', ' greater than ', text)
    text = re.sub(r'<', ' less than ', text)
    text = re.sub(r'=', ' equals ', text)
    
    # Convert arrows to words
    text = re.sub(r'-->', ' directed to ', text)
    text = re.sub(r'<--', ' directed from ', text)
    text = re.sub(r'--', ' connected to ', text)
    
    logger.debug(f"Preprocessed text: {text[:100]}...")
    return text

def postprocess_text(text: str) -> str:
    """
    Postprocess generated text to restore proper Cypher formatting.
    
    Args:
        text: Generated text to postprocess
        
    Returns:
        Postprocessed text with proper Cypher syntax
    """
    logger.debug(f"Postprocessing text: {text[:100]}...")
    
    # Restore node patterns
    text = re.sub(r'\(node (\w+)\)', r'(\1)', text)
    text = re.sub(r'\((\w+) labeled (\w+)\)', r'(\1:\2)', text)
    
    # Restore relationship patterns
    text = re.sub(r'\[relationship (\w+)\]', r'[\1]', text)
    text = re.sub(r'\[relationship (\w+) of type (\w+)\]', r'[\1:\2]', text)
    
    # Restore property access
    text = re.sub(r'(\w+) property (\w+)', r'\1.\2', text)
    
    # Restore comparison operators
    text = re.sub(r' greater than or equal to ', ' >= ', text)
    text = re.sub(r' less than or equal to ', ' <= ', text)
    text = re.sub(r' not equal to ', ' != ', text)
    text = re.sub(r' greater than ', ' > ', text)
    text = re.sub(r' less than ', ' < ', text)
    text = re.sub(r' equals ', ' = ', text)
    
    # Restore arrows
    text = re.sub(r' directed to ', '-->', text)
    text = re.sub(r' directed from ', '<--', text)
    text = re.sub(r' connected to ', '--', text)
    
    # Clean up extra spaces
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    logger.debug(f"Postprocessed text: {text[:100]}...")
    return text

def add_custom_tokens_to_tokenizer(tokenizer: Any) -> Any:
    """
    Add custom tokens to a tokenizer.
    
    Args:
        tokenizer: Hugging Face tokenizer
        
    Returns:
        Updated tokenizer with custom tokens
    """
    custom_tokens = get_custom_tokens()
    
    # Add tokens that don't exist in the vocabulary
    new_tokens = []
    for token in custom_tokens:
        if token not in tokenizer.get_vocab():
            new_tokens.append(token)
    
    if new_tokens:
        logger.info(f"Adding {len(new_tokens)} custom tokens to tokenizer")
        tokenizer.add_tokens(new_tokens)
        logger.debug(f"Added tokens: {new_tokens[:10]}...")  # Log first 10
    else:
        logger.info("No new tokens to add to tokenizer")
    
    return tokenizer

def normalize_cypher_query(query: str) -> str:
    """
    Normalize a Cypher query for consistency.
    
    Args:
        query: Raw Cypher query
        
    Returns:
        Normalized Cypher query
    """
    # Convert to consistent case for keywords
    cypher_keywords = [
        'MATCH', 'RETURN', 'WHERE', 'CREATE', 'DELETE', 'SET', 'REMOVE',
        'WITH', 'UNWIND', 'ORDER BY', 'LIMIT', 'SKIP', 'UNION', 'DISTINCT',
        'CASE', 'WHEN', 'THEN', 'ELSE', 'END', 'AND', 'OR', 'NOT'
    ]
    
    normalized_query = query
    for keyword in cypher_keywords:
        # Replace case-insensitive occurrences with uppercase
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)
        normalized_query = pattern.sub(keyword.upper(), normalized_query)
    
    # Clean up whitespace
    normalized_query = re.sub(r'\s+', ' ', normalized_query)
    normalized_query = normalized_query.strip()
    
    return normalized_query

def extract_cypher_components(query: str) -> Dict[str, List[str]]:
    """
    Extract components from a Cypher query for analysis.
    
    Args:
        query: Cypher query to analyze
        
    Returns:
        Dictionary with extracted components
    """
    components = {
        'nodes': [],
        'relationships': [],
        'properties': [],
        'keywords': []
    }
    
    # Extract node patterns
    node_pattern = r'\((\w*):?(\w*)\)'
    nodes = re.findall(node_pattern, query)
    components['nodes'] = [f"({':'.join(filter(None, node))})" for node in nodes]
    
    # Extract relationship patterns
    rel_pattern = r'\[(\w*):?(\w*)\]'
    relationships = re.findall(rel_pattern, query)
    components['relationships'] = [f"[{':'.join(filter(None, rel))}]" for rel in relationships]
    
    # Extract property references
    prop_pattern = r'(\w+)\.(\w+)'
    properties = re.findall(prop_pattern, query)
    components['properties'] = [f"{prop[0]}.{prop[1]}" for prop in properties]
    
    # Extract Cypher keywords
    keywords = []
    for term in CYPHER_TERMS:
        if term.upper() in query.upper():
            keywords.append(term)
    components['keywords'] = list(set(keywords))
    
    return components

