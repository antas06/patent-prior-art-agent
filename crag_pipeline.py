from duckduckgo_search import DDGS
from config import Config
from system1 import CodivSystem1Judge
from retriever import QdrantRetriever

class CRAGPipeline:
  """Corrective RAG Pipeline for evaluating and routing context."""
  
  def __init__(self, config: Config, sys1: CodivSystem1Judge, retriever: QdrantRetriever):
    self.config = config
    self.sys1 = sys1
    self.retriever = retriever

  def process_feature(self, feature: str, parent_span=None) -> dict:
    """Runs the CRAG decision loop for a single feature."""
    internal_chunks = self.retriever.search(feature, top_k=3)
    
    graded_chunks = []
    rel_count = 0

    for chunk in internal_chunks:
      scores = self.sys1.grade_chunk(feature, chunk, parent_span=parent_span)
      graded_chunks.append({"text": chunk, "scores": scores, "source": "internal"})
      if scores["relevant"] >= self.config.THRESH_RELEVANT:
        rel_count += 1

    if rel_count >= self.config.MIN_RELEVANT_CHUNKS:
      return {"route": "RELEVANT", "evidence": [c for c in graded_chunks if c["scores"]["relevant"] >= self.config.THRESH_RELEVANT]}

    web_results = self._web_search(feature)
    for w_text in web_results:
      scores = self.sys1.grade_chunk(feature, w_text, parent_span=parent_span)
      graded_chunks.append({"text": w_text, "scores": scores, "source": "web"})

    final_evidence = [c for c in graded_chunks if c["scores"]["relevant"] >= self.config.WEB_FALLBACK_THRESHOLD]
    
    return {"route": "WEB_FALLBACK", "evidence": final_evidence}

  def _web_search(self, query: str) -> list[str]:
    """Performs a web search via DuckDuckGo."""
    try:
      with DDGS() as ddgs:
        results = list(ddgs.text(f"patent prior art {query}", max_results=3))
        return [r["body"] for r in results]
    except Exception as e:
      print(f"Web search error: {e}")
      return []