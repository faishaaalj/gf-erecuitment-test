import os
import openai
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.search.documents import SearchClient


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
