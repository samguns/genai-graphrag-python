import os
from dotenv import load_dotenv
load_dotenv()

import asyncio

from neo4j import GraphDatabase
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline


class VLLMOpenAILLM(OpenAILLM):
    supports_structured_output = False


# tag::neo4j_driver[]
neo4j_driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
)
neo4j_driver.verify_connectivity()
# end::neo4j_driver[]

# tag::llm[]
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "EMPTY")

llm = VLLMOpenAILLM(
    model_name="google/gemma-4-E4B-it",
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
    model_params={
        "max_tokens": 16384,
        "temperature": 0,
    }
)

embedder = OpenAIEmbeddings(
    model="google/embeddinggemma-300m",
    api_key=OPENAI_API_KEY,
    base_url="http://192.168.31.100:8889/v1",
)
# end::embedder[]

# tag::kg_builder[]
kg_builder = SimpleKGPipeline(
    llm=llm,
    driver=neo4j_driver, 
    neo4j_database=os.getenv("NEO4J_DATABASE"), 
    embedder=embedder, 
    from_pdf=True,
    schema="FREE",
)
# end::kg_builder[]

# tag::run_one_doc[]
pdf_file = "./genai-graphrag-python/data/genai-fundamentals_1-generative-ai_1-what-is-genai.pdf"
result = asyncio.run(kg_builder.run_async(file_path=pdf_file))
print(result.result)
# end::run_one_doc[]

# tag::run_multiple_docs[]
# data_path = "./genai-graphrag-python/data/"
# pdf_files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.pdf')]

# for pdf_file in pdf_files:

#     print(f"Processing {pdf_file}")
#     result = asyncio.run(kg_builder.run_async(file_path=pdf_file))
#     print(result.result)
# end::run_multiple_docs[]