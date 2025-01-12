import os
from typing import Dict, Any
from pinecone import Pinecone, ServerlessSpec, Index
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Setup logging with more detailed configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pinecone_operations.log')
    ]
)

logger = logging.getLogger(__name__)

class PineconeIndexManager:
    def __init__(self):
        """Initialize PineconeIndexManager with configuration from environment variables."""
        # Required environment variables
        self.api_key = os.getenv("PINECONE_API_KEY")
        if not self.api_key:
            raise ValueError("PINECONE_API_KEY environment variable is required")

        # Optional environment variables with defaults
        self.environment = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
        self.base_index_name = os.getenv("PINECONE_BASE_INDEX_NAME", "rag-chatbot-index")
        self.num_indexes = int(os.getenv("NUM_PINECONE_INDEXES", "5"))
        self.embedding_dimension = int(os.getenv("EMBEDDING_DIMENSION", "768"))
        
        # Initialize Pinecone client
        self.pc = Pinecone(api_key=self.api_key)
        
    def _get_index_name(self, index_number: int) -> str:
        """Generate index name based on base name and number."""
        return f"{self.base_index_name}-{index_number}"

    def _create_index(self, index_name: str) -> None:
        """Create a new Pinecone index with specified configuration."""
        try:
            self.pc.create_index(
                name=index_name,
                dimension=self.embedding_dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region=self.environment
                )
            )
            logger.info(f"Successfully created index: {index_name}")
        except Exception as e:
            logger.error(f"Failed to create index {index_name}: {e}")
            raise

    def create_or_connect_to_indexes(self) -> Dict[int, Index]:
        """
        Create or connect to multiple Pinecone indexes.
        
        Returns:
            Dict[int, Index]: Dictionary mapping index numbers to Pinecone index objects
        """
        indexes: Dict[int, Index] = {}
        
        for i in range(1, self.num_indexes + 1):
            try:
                index_name = self._get_index_name(i)
                existing_indexes = self.pc.list_indexes().names()
                
                if index_name not in existing_indexes:
                    logger.info(f"Creating new index: {index_name}")
                    self._create_index(index_name)
                else:
                    logger.info(f"Connecting to existing index: {index_name}")
                
                indexes[i] = self.pc.Index(index_name)
                
            except Exception as e:
                logger.error(f"Error processing index {i}: {e}")
                raise
        
        return indexes

    def get_index_statistics(self, indexes: Dict[int, Index]) -> None:
        """Print statistics for all indexes."""
        for index_id, index in indexes.items():
            try:
                stats = index.describe_index_stats()
                logger.info(f"Index {index_id} ({self._get_index_name(index_id)}) stats: {stats}")
            except Exception as e:
                logger.error(f"Failed to get stats for index {index_id}: {e}")

def main():
    """Main execution function."""
    try:
        # Initialize manager
        manager = PineconeIndexManager()
        
        # Create or connect to indexes
        indexes = manager.create_or_connect_to_indexes()
        logger.info("Successfully initialized all Pinecone indexes")
        
        # Get and display index statistics
        manager.get_index_statistics(indexes)
        
    except Exception as e:
        logger.error(f"Application failed: {e}")
        raise

if __name__ == "__main__":
    main()