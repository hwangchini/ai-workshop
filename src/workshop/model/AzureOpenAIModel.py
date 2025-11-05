"""Configuration class responsible for loading environment variables and initializing LangChain clients for Azure OpenAI."""

import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

class AzureOpenAIModel:
    """Loads credentials and creates LangChain-compatible clients for LLM and embeddings."""

    def __init__(self):
        """Initializes the model by loading environment variables and creating the clients."""
        load_dotenv()
        
        # Create the LangChain Chat Model instance for conversational tasks.
        self.llm = AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            model=os.getenv("AZURE_OPENAI_MODEL_NAME")
        )

        # Create the LangChain Embedding Model instance for text embedding tasks.
        self.embedding_model = AzureOpenAIEmbeddings(
            azure_endpoint=os.getenv("AZURE_EMBEDDING_ENDPOINT"),
            api_key=os.getenv("AZURE_EMBEDDING_API_KEY"),
            azure_deployment=os.getenv("AZURE_EMBEDDING_DEPLOYMENT_NAME"),
            api_version=os.getenv("AZURE_EMBEDDING_API_VERSION"),
            model=os.getenv("AZURE_EMBEDDING_DEPLOYMENT_NAME")
        )
