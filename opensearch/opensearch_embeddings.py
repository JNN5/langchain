from typing import List
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

class OpenSearchEmpeddings(Embeddings):
    model: SentenceTransformer

    def __init__(self, model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v1')) -> None:
        super().__init__()
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts)
    
    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text])[0]