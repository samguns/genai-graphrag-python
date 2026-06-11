import os

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()


DROP_VECTOR_INDEX_QUERY = """
DROP INDEX chunkEmbedding IF EXISTS
"""


VECTOR_INDEX_QUERY = """
CREATE VECTOR INDEX chunkEmbedding IF NOT EXISTS
FOR (n:Chunk)
ON n.embedding
OPTIONS {indexConfig: {
  `vector.dimensions`: 768,
  `vector.similarity_function`: 'cosine'
}}
"""


driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
)

try:
    driver.verify_connectivity()
    database = os.getenv("NEO4J_DATABASE")

    with driver.session(database=database) as session:
        session.run(DROP_VECTOR_INDEX_QUERY).consume()
        session.run(VECTOR_INDEX_QUERY).consume()

    print("Vector index 'chunkEmbedding' is ready with 768 dimensions.")
finally:
    driver.close()
