from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from fastembed import TextEmbedding
from config import Config

class QdrantRetriever:
  """Embedded Vector Search using Qdrant and FastEmbed."""
  
  def __init__(self, config: Config):
    self.client = QdrantClient(path=config.QDRANT_PATH)
    self.embed_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    self.collection_name = "internal_prior_art"
    
    if not self.client.collection_exists(self.collection_name):
      self.client.create_collection(
        collection_name=self.collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
      )

  def index_documents(self, chunks: list[str]):
    """Embeds and indexes document chunks."""
    if not chunks:
      return
      
    embeddings = list(self.embed_model.embed(chunks))
    points = [
      PointStruct(id=i, vector=emb.tolist(), payload={"text": chunk})
      for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
    ]
    self.client.upsert(collection_name=self.collection_name, points=points)

  def search(self, query: str, top_k: int = 5) -> list[str]:
    """Retrieves top-k relevant chunks based on vector cosine similarity."""
    if not self.client.collection_exists(self.collection_name):
      return []
      
    query_emb = list(self.embed_model.embed([query]))[0].tolist()
    hits = self.client.search(
      collection_name=self.collection_name,
      query_vector=query_emb,
      limit=top_k
    )
    return [hit.payload.get("text", "") for hit in hits]