"""
Script to setup Neo4j indexes and Qdrant collections.
Run this once after creating your cloud databases or local instances.
"""

import sys
from graphrag.utils.logger import logger
from graphrag.utils.config import reload_env, get_config
from graphrag.connectors.neo4j_connection import get_connection as get_neo4j
from graphrag.connectors.qdrant_connection import get_connection as get_qdrant

def setup_neo4j():
    """Setup Neo4j indexes (including fulltext and vector if supported)"""
    logger.info("Setting up Neo4j indexes...")
    neo4j = get_neo4j()
    neo4j.setup_indexes()
    logger.info("Neo4j indexes setup complete.")

def setup_qdrant():
    """Setup Qdrant collection"""
    logger.info("Setting up Qdrant collection...")
    qdrant = get_qdrant()
    vector_size = int(get_config("VECTOR_SIZE", 768))
    collection_name = get_config("QDRANT_COLLECTION_NAME", "tokens")
    success = qdrant.setup_collections(collection_name=collection_name, vector_size=vector_size)
    if success:
        logger.info(f"Qdrant collection '{collection_name}' ready.")
    else:
        logger.error("Failed to setup Qdrant collection.")
        sys.exit(1)

def main():
    reload_env()
    logger.info("Starting database setup...")
    try:
        setup_neo4j()
        setup_qdrant()
        logger.info("Database setup completed successfully.")
    except Exception as e:
        logger.error(f"Setup failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()