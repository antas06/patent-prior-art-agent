import streamlit as st
import os
from config import Config
from system1 import CodivSystem1Judge
from retriever import QdrantRetriever
from crag_pipeline import CRAGPipeline
from agent import PriorArtAgent
from obs import Tracer

st.set_page_config(page_title="Agentic Prior-Art Assistant", layout="wide")
st.title("🛡️ Agentic Prior-Art Assistant")

@st.cache_resource
def load_system():
  cfg = Config()
  tracer = Tracer(cfg)
  sys1 = CodivSystem1Judge(cfg)
  retriever = QdrantRetriever(cfg)
  crag = CRAGPipeline(cfg, sys1, retriever)
  agent = PriorArtAgent(cfg, crag, tracer)
  return cfg, retriever, agent, tracer

cfg, retriever, agent, tracer = load_system()

# Sidebar Setup
with st.sidebar:
  st.header("1. Settings & Observability")
  st.write(f"**Langfuse Tracing:** `{'ON' if tracer.enabled else 'OFF'}`")
  
  if st.button("Index Sample Disclosure"):
    sample_path = "data/sample_disclosure.txt"
    if os.path.exists(sample_path):
      with open(sample_path, "r") as f:
        text = f.read()
      chunks = [c.strip() for c in text.split("\n\n") if len(c.strip()) > 20]
      retriever.index_documents(chunks)
      st.success(f"Indexed {len(chunks)} chunks into Qdrant!")

# Main Interface
disclosure_input = st.text_area("2. Paste Invention Disclosure here:", height=200)

if st.button("Run Prior-Art Screening", type="primary"):
  if not disclosure_input.strip():
    st.warning("Please provide an invention disclosure.")
  elif not cfg.CODIV_API_KEY or not cfg.GROQ_API_KEY:
    st.error("API Keys missing. Please check your .env file.")
  else:
    with st.spinner("Executing Agentic Loop..."):
      results = agent.run(disclosure_input)
      
    st.subheader("Step 1: Extracted Features")
    for f in results["features"]:
      st.markdown(f"- {f}")
      
    st.subheader("Step 2: CRAG Evidence Sourcing")
    for feature, data in results["evidence_cards"].items():
      with st.expander(f"Feature: {feature} ({data['route']})"):
        for idx, card in enumerate(data["evidence"]):
          st.markdown(f"**Source:** `{card['source']}` | **P(rel):** `{card['scores']['relevant']:.2f}`")
          st.write(card["text"])
          st.divider()

    st.subheader("Step 3: Final Screening Report")
    st.write(results["report"])