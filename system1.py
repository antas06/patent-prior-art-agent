import requests
from config import Config

class CodivSystem1Judge:
  """System 1 Relevance Judge using Codiv API for Laya/OpenJev."""
  
  def __init__(self, config: Config):
    self.api_key = config.CODIV_API_KEY
    self.model = config.SYSTEM1_MODEL
    self.endpoint = "https://api.codiv.ai/v1/systemone"

  def grade_chunk(self, feature: str, chunk_text: str, parent_span=None) -> dict:
    headers = {
      "Authorization": f"Bearer {self.api_key}",
      "Content-Type": "application/json"
    }
    
    prompt = f"Is the following text relevant to the technical feature: '{feature}'?\nText: {chunk_text}"
    payload = {
      "model": self.model,
      "prompt": prompt,
      "choices": ["Relevant", "Partial", "Irrelevant"]
    }
    
    span = None
    if parent_span and hasattr(parent_span, "span"):
      span = parent_span.span(name="system1_grade", input={"feature": feature, "chunk": chunk_text[:100]})

    try:
      response = requests.post(self.endpoint, headers=headers, json=payload)
      response.raise_for_status()
      data = response.json()
      probs = data.get("probabilities", {})
      result = {
        "relevant": probs.get("Relevant", 0.0),
        "partial": probs.get("Partial", 0.0),
        "irrelevant": probs.get("Irrelevant", 0.0)
      }
    except Exception as e:
      result = {"relevant": 0.0, "partial": 0.0, "irrelevant": 1.0}

    if span and hasattr(span, "end"):
      span.end(output=result)
      
    return result