import os
from dotenv import load_dotenv
load_dotenv()

import asyncio
from typing import Dict, Optional, Union

from neo4j import GraphDatabase
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline

# tag::import_loader[]
# You will need to install the wikipedia package: pip install wikipedia
from neo4j_graphrag.experimental.components.pdf_loader import DataLoader, PdfDocument, DocumentInfo
from pathlib import Path
import wikipedia
from urllib.parse import quote
from fsspec import AbstractFileSystem
# end::import_loader[]

class VLLMOpenAILLM(OpenAILLM):
    supports_structured_output = False

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "EMPTY")

neo4j_driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
)
neo4j_driver.verify_connectivity()

llm = VLLMOpenAILLM(
    model_name="google/gemma-4-E4B-it",
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
    model_params={
        "max_tokens": 16384,
        "temperature": 0,
        "reasoning_effort": "minimal"
    }
)

embedder = OpenAIEmbeddings(
    model="google/embeddinggemma-300m",
    api_key=OPENAI_API_KEY,
    base_url="http://192.168.31.100:8889/v1",
)

# tag::loader[]
class WikipediaLoader(DataLoader):
    async def run(
        self,
        filepath: Union[str, Path],
        metadata: Optional[Dict[str, str]] = None,
        fs: Optional[Union[AbstractFileSystem, str]] = None,
    ) -> PdfDocument:

        # Load the Wikipedia page
        page = wikipedia.page(filepath)

        # Return a PdfDocument
        return PdfDocument(
            text=page.content,
            document_info=DocumentInfo(
                path=str(filepath),
                metadata={
                    **(metadata or {}),
                    "url": f"https://en.wikipedia.org/w/index.php?title={quote(page.title)}",
                }
            )
        )

data_loader = WikipediaLoader()
# end::loader[]

# tag::kg_builder[]
kg_builder = SimpleKGPipeline(
    llm=llm,
    driver=neo4j_driver, 
    neo4j_database=os.getenv("NEO4J_DATABASE"), 
    embedder=embedder, 
    from_pdf=True,
    pdf_loader=data_loader,
    schema="FREE",
)
# end::kg_builder[]

# tag::run_loader[]
wikipedia_page = "Knowledge Graph"
doc = asyncio.run(data_loader.run(wikipedia_page))
print(doc.text)
# end::run_loader[]

print(f"Processing {wikipedia_page}")
result = asyncio.run(kg_builder.run_async(file_path=wikipedia_page))
print(result.result)
