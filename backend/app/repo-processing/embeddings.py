from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from models import CodeSectionModel
import os
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

class EmbeddingStoreFAISS:
    def __init__(self):
        # Initialize OpenAI embeddings
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=openai_api_key)
        self.store = None

    def add_sections(self, sections: list[CodeSectionModel]):
        """
        Add a list of code sections to the FAISS vector store with metadata
        """
        texts = [sec.page_content for sec in sections]
        # Extract metadata from sections
        metadatas = [{"source": sec.metadata.get("source", "unknown"), 
                      "language": sec.metadata.get("language", "unknown")} 
                     for sec in sections]

        if self.store is None:
            # Create a new FAISS store with metadata
            self.store = FAISS.from_texts(texts=texts, embedding=self.embeddings, metadatas=metadatas)
        else:
            # Add more sections to existing store with metadata
            self.store.add_texts(texts=texts, metadatas=metadatas)

    def save(self, path="faiss_index"):
        """
        Persist the FAISS index locally
        """
        if self.store:
            print("SAVING FAISS AT:", path)
            self.store.save_local(path)

    def load(self, path="faiss_index"):
        """
        Load an existing FAISS index
        """
        self.store = FAISS.load_local(path, self.embeddings, allow_dangerous_deserialization=True)
        return self.store