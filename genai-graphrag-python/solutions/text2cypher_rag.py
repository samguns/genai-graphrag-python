import os
from dotenv import load_dotenv
load_dotenv()

from neo4j import GraphDatabase
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.retrievers import Text2CypherRetriever


class VLLMOpenAILLM(OpenAILLM):
    supports_structured_output = False


# Connect to Neo4j database
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"), 
    auth=(
        os.getenv("NEO4J_USERNAME"), 
        os.getenv("NEO4J_PASSWORD")
    )
)

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

# tag::examples[]
# Cypher examples as input/query pairs
examples = [
    "USER INPUT: 'Find a node with the name $name?' QUERY: MATCH (node) WHERE toLower(node.name) CONTAINS toLower($name) RETURN node.name AS name, labels(node) AS labels",
]
# end::examples[]

# tag::retriever[]
# Build the retriever
retriever = Text2CypherRetriever(
    driver=driver,
    neo4j_database=os.getenv("NEO4J_DATABASE"),
    llm=llm,
    examples=examples,
)
# end::retriever[]

rag = GraphRAG(
    retriever=retriever, 
    llm=llm
)

# query_text = "How many technologies are mentioned in the knowledge graph?"
# query_text = "What entities exist in the knowledge graph?"
query_text = "What resource urls are referenced?"

response = rag.search(
    query_text=query_text,
    return_context=True
    )

# tag::print_response[]
print(response.answer)
print("CYPHER :", response.retriever_result.metadata["cypher"])
print("CONTEXT:", response.retriever_result.items)
# end::print_response[]

driver.close()


"""
# tag::example_queries[]
query_text = "How many technologies are mentioned in the knowledge graph?"
query_text = "How does Neo4j relate to other technologies?"
query_text = "What entities exist in the knowledge graph?" 
query_text = "What resource urls are referenced?"
# end::example_queries[]
"""
