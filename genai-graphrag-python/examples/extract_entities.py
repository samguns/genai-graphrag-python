import os
from dotenv import load_dotenv
load_dotenv()

import asyncio
from neo4j_graphrag.experimental.components.entity_relation_extractor import (
    LLMEntityRelationExtractor,
)
from neo4j_graphrag.experimental.components.types import (
    TextChunks,
    TextChunk
)
from neo4j_graphrag.llm import OpenAILLM

class VLLMOpenAILLM(OpenAILLM):
    supports_structured_output = False

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "EMPTY")

text = """
Neo4j is a graph database that stores data in a graph structure.
Data is stored as nodes and relationships instead of tables or documents.
Graph databases are particularly useful when _the connections between data are as important as the data itself_.
A graph shows how objects are related to each other.
The objects are referred to as *nodes* (vertices) connected by *relationships* (edges).
Neo4j uses the graph structure to store data and is known as a *labeled property graph*.
"""

extractor = LLMEntityRelationExtractor(
    llm = VLLMOpenAILLM(
        model_name="google/gemma-4-E4B-it",
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        model_params={
        "max_tokens": 16384,
        "temperature": 0,
        "reasoning_effort": "minimal",
        "response_format": {"type": "json_object"},
    }
)
)

entities = asyncio.run(
    extractor.run(
        chunks=TextChunks(chunks=[TextChunk(text=text, index=0)])
    )
)

print(entities)
