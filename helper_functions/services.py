import os
import openai
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.search.documents import SearchClient
from azure.storage.blob import BlobServiceClient, ContainerClient # Import ContainerClient
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Azure AI Search Client ---
search_endpoint = os.environ["AI_SEARCH_ENDPOINT"]
search_key = AzureKeyCredential(os.environ["AI_SEARCH_KEY"])
search_index_name = os.environ["AI_SEARCH_INDEX_NAME"]

# Client for managing documents within the index (upload, query)
search_client = SearchClient(
    endpoint=search_endpoint, 
    index_name=search_index_name, 
    credential=search_key
)

# --- Azure Document Intelligence Client ---
doc_intel_endpoint = os.environ["DOC_INTEL_ENDPOINT"]
doc_intel_key = AzureKeyCredential(os.environ["DOC_INTEL_KEY"])

document_intelligence_client = DocumentIntelligenceClient(
    endpoint=doc_intel_endpoint, 
    credential=doc_intel_key
)

# --- Azure OpenAI Client ---
openai_client = openai.AzureOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_KEY"]
)

openai_chat_deployment = os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"]
openai_embedding_deployment = os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"]

# --- NEW: Token Cache Client (uses AzureWebJobsStorage) ---
# This client connects to the Function App's own storage for caching the auth token.
TOKEN_CACHE_CONTAINER_NAME = "function-token-cache"
token_cache_container_client: Optional[ContainerClient] = None
try:
    # AzureWebJobsStorage is the standard connection string for the Function App itself
    storage_conn_str = os.environ["AzureWebJobsStorage"]
    if not storage_conn_str:
        logger.critical("FATAL: AzureWebJobsStorage is not set. Token caching will fail.")
        raise SystemExit("Configuration Error: AzureWebJobsStorage is not set.")
        
    blob_service_client = BlobServiceClient.from_connection_string(storage_conn_str)
    
    # Get the container client
    token_cache_container_client = blob_service_client.get_container_client(TOKEN_CACHE_CONTAINER_NAME)
    if not token_cache_container_client.exists():
        logger.info(f"Creating token cache container: {TOKEN_CACHE_CONTAINER_NAME}")
        token_cache_container_client.create_container()
    logger.info("Token cache blob client initialized successfully.")
    
except KeyError:
    logger.critical("FATAL: AzureWebJobsStorage environment variable not found. Durable Functions and token caching will fail.")
    # This will cause the app to fail to start, which is correct
    raise SystemExit("Configuration Error: AzureWebJobsStorage is not set.")
except Exception as e:
    logger.critical(f"FATAL: Failed to initialize token cache client: {e}")
    raise SystemExit(f"Configuration Error: Failed to initialize token cache client: {e}")