import faiss
import ollama

class VectorStore:
    def __init__(self, model_name="nomic-embed-text:latest"):
        self.model_name = model_name
        self.index = None
        self.chunks = []

    def build(self, chunks: list[str]):
        if not chunks:
            return

        embeddings = []
        for chunk in chunks:
            response = ollama.embeddings(model=self.model_name, prompt=chunk)
            embeddings.append(response['embedding'])

        import numpy as np
        embeddings = np.array(embeddings, dtype='float32')

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)

        self.chunks = chunks

    def search(self, query: str, k: int = 3):
        if self.index is None:
            return []

        response = ollama.embeddings(model=self.model_name, prompt=query)
        query_vec = [response['embedding']]
        
        import numpy as np
        query_vec = np.array(query_vec, dtype='float32')
        
        distances, indices = self.index.search(query_vec, k)

        results = []
        for idx in indices[0]:
            if idx < len(self.chunks):
                results.append(self.chunks[idx])

        return results