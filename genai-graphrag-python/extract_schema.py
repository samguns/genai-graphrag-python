import os
from dotenv import load_dotenv
load_dotenv()

from neo4j_graphrag.experimental.components.schema import SchemaFromTextExtractor
from neo4j_graphrag.llm import OpenAILLM
from rich import print
import asyncio

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "EMPTY")

schema_extractor = SchemaFromTextExtractor(
    llm = OpenAILLM(
        model_name="google/gemma-4-E4B-it",
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        model_params={"temperature": 0.7, "max_tokens": 16384},
    ),
    use_structured_output=True,
)

# text = """
# Neo4j is a graph database management system (GDBMS) developed by Neo4j Inc.
# """
text = """
Large Language Models (LLMs) are a type of artificial intelligence model designed to generate human-like text.
"""

# Extract the schema from the text
extracted_schema = asyncio.run(schema_extractor.run(text=text))

print(extracted_schema)
