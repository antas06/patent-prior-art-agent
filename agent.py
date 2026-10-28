from groq import Groq
from config import Config
from crag_pipeline import CRAGPipeline
from obs import Tracer

class PriorArtAgent:
  """The main orchestrator loop: Plan -> Act -> Synthesize."""
  
  def __init__(self, config: Config, crag: CRAGPipeline, tracer: Tracer):
    self.config = config
    self.crag = crag
    self.tracer = tracer
    self.client = Groq(api_key=config.GROQ_API_KEY)
    self.model = config.LLM_MODEL

  def run(self, disclosure_text: str) -> dict:
    trace = self.tracer.create_trace(
      name="prior_art_screening",
      input_data={"disclosure_snippet": disclosure_text[:200]}
    )
    
    features = self._plan_features(disclosure_text, trace)
    
    evidence_cards = {}
    for feature in features:
      f_span = trace.span(name="crag_feature_route", input={"feature": feature})
      result = self.crag.process_feature(feature, parent_span=f_span)
      f_span.end(output={"route": result["route"], "count": len(result["evidence"])})
      evidence_cards[feature] = result
        
    report = self._synthesize_report(features, evidence_cards, trace)
    
    if hasattr(trace, "end"):
      trace.end(output={"report_length": len(report)})
    
    self.tracer.flush()
    
    return {
      "features": features,
      "evidence_cards": evidence_cards,
      "report": report
    }

  def _plan_features(self, text: str, trace) -> list[str]:
    gen = trace.generation(name="plan_features", model=self.model, input={"text": text[:300]})
    
    prompt = f"""You are a patent attorney. Extract up to {self.config.MAX_FEATURES} core distinct technical features from this invention disclosure. 
Return ONLY a bulleted list of features, one per line. No introductory text.

Disclosure:
{text}
"""
    response = self.client.chat.completions.create(
      messages=[{"role": "user", "content": prompt}],
      model=self.model,
      temperature=0.1
    )
    content = response.choices[0].message.content
    features = [line.strip("-*1234567890. ").strip() for line in content.split('\n') if line.strip()][:self.config.MAX_FEATURES]
    
    gen.end(output={"features": features})
    return features

  def _synthesize_report(self, features: list[str], evidence: dict, trace) -> str:
    gen = trace.generation(name="synthesize_report", model=self.model)
    
    context_block = ""
    for feature, data in evidence.items():
      context_block += f"\nFeature: {feature}\nRoute Taken: {data['route']}\nEvidence:\n"
      for card in data['evidence']:
        context_block += f"- [{card['source'].upper()}] {card['text']}\n"

    prompt = f"""You are a prior-art screening assistant. Write a triage report for a patent attorney based ONLY on the evidence provided below. 
Address each feature, stating whether prior art was found internally, externally, or not at all. Include snippets as citations.

EVIDENCE:
{context_block}
"""
    response = self.client.chat.completions.create(
      messages=[{"role": "user", "content": prompt}],
      model=self.model,
      temperature=0.2
    )
    content = response.choices[0].message.content
    gen.end(output={"report_snippet": content[:200]})
    return content