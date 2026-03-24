Here's your complete README with the image swapped in:

markdown
## ✨ Financial Multimodal RAG System (System-1 vs System-2 Hybrid)

A production-style Retrieval Augmented Generation (RAG) system designed to handle large-scale financial documents with intelligent routing, multimodal ingestion, table preservation, metadata filtering and analytics capabilities.

This system supports both offline bulk-ingested corpora and real-time user document uploads, enabling scalable document question answering across multiple formats.

---

## 🚀 Key Highlights

- Hybrid System-1 (Fast Cached Retrieval) vs System-2 (Deep Reasoning Retrieval + Reranking)
- Handles PDF, DOCX, PPT, CSV, TXT, Markdown, HTML
- Multimodal-ready (scanned PDFs / images supported via Docling / Unstructured pipeline)
- Financial report reasoning with citations + page grounding
- Production-oriented modular architecture
- Metadata filtering ready vector search (Qdrant)
- Intelligent routing between offline corpus and uploaded documents
- CSV analytics + visualization capability
- Redis-style caching for latency optimization
- MCP tool-calling ready pipeline design

---

## 🧠 Architecture Diagram

![RAG Pipeline Architecture](rag-pipeline.png)

---

## 🔄 System Flow

1. Offline ingestion pipeline converts financial PDFs → structured JSON knowledge.
2. Tables are serialized and preserved for reasoning accuracy.
3. Vector embeddings + metadata stored in Qdrant vector database.
4. User asks a question through Streamlit interface.
5. Query routing detects company keyword → routes to offline corpus OR upload processor.
6. Uploaded documents are parsed using Unstructured pipeline dynamically.
7. Hybrid retrieval fetches relevant chunks (vector + keyword).
8. LLM reranking improves context precision.
9. Final reasoning performed using compact context window.
10. Answer returned with citations and page references.

---

## 📂 Supported Document Formats

- PDF (text + scanned)
- DOCX
- PPT / PPTX
- CSV
- Markdown
- TXT
- HTML
- Future ready → SQL / DB connectors

---

## 📊 CSV Analytics Capability

- Automated exploratory data analysis
- Statistical summaries
- Pattern detection
- Visualization support via Matplotlib and Seaborn
- LLM assisted tabular reasoning

---

## ⚡ Performance Optimizations

- Cache-based fast retrieval layer (System-1 behavior)
- LLM reranking only when needed (System-2 deep reasoning)
- Metadata filtering reduces vector search latency
- Parent page retrieval reduces hallucination risk
- Modular ingestion enables scalable corpus growth

---

## 📚 Citation Grounding

Every answer is grounded with:

- Page index
- Document SHA reference
- Relevant chunk evidence

This ensures auditability and explainability for financial reasoning tasks.

---

## 🧩 MCP / Tool Calling Ready

Architecture designed to integrate:

- Claude MCP tools
- External data connectors
- Autonomous retrieval workflows
- Multi-agent extensions

---

## 🧱 Production Style Design

- Modular ingestion / retrieval / reasoning layers
- Configurable pipelines
- Parallel retrieval support
- Evaluation friendly output structure
- Extensible vector DB layer

---

## 🙏 Acknowledgement

Parts of document parsing inspiration and retrieval structuring concepts were explored from community RAG challenge implementations.

Core architecture enhancements, routing logic, multimodal ingestion strategy, caching design, analytics integration and system optimization were independently designed and implemented in this project.

---

## 🎯 Future Improvements

- Full multimodal VLM reasoning
- Auto metadata extraction during ingestion
- Adaptive chunking strategies
- Latency aware retrieval planner
- Agentic workflow orchestration
- Cloud deployment with distributed vector search

---

## 🖥 Demo

- Working demo video available
- Streamlit interactive QA interface

---

⭐ If you like this project, consider starring the repo!


Just make sure rag-pipeline.png is in the *root of your repo* (same level as README.md) and it'll render automatically on GitHub.
