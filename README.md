# AI Construction Knowledge Assistant (Mini RAG System)

## Live Application
https://ai-construction-rag-gdhijzvl4gmscelzm6ummy.streamlit.app/

## Project Overview

This project implements a Retrieval-Augmented Generation (RAG) based AI assistant designed for a construction marketplace.

The assistant answers user queries strictly using internal construction policy documents instead of relying on general model knowledge. The system retrieves relevant document chunks using semantic similarity search and generates grounded responses using a Large Language Model.

This approach improves answer reliability, reduces hallucination, and ensures transparency by displaying the retrieved context used for answer generation.

---

## System Architecture

User Query → Embedding Model → FAISS Retrieval → Context Injection → LLM Response

The system consists of three main components:

• Document processing & chunking  
• Semantic retrieval using vector similarity  
• Grounded answer generation using an LLM  

---

## Model Choices

### Embedding Model
**Model:** all-MiniLM-L6-v2 (Sentence Transformers)

Chosen because:

• Lightweight and fast  
• Strong semantic similarity performance  
• Suitable for small-to-medium document corpora  
• Efficient for real-time retrieval  

---

### Language Model (LLM)

**Primary Model:** DeepSeek Chat via OpenRouter  
**Fallback Model:** LLaMA-3-8B Instruct via OpenRouter  

Reasoning:

• Instruction-following capability  
• Good structured answer generation  
• Easy API integration  
• No need for local GPU resources  

Fallback logic ensures robustness when API endpoints are rate-limited.

---

## Document Processing

Documents are split using a sliding-window chunking strategy:

- Chunk size: ~400 characters  
- Overlap: ~80 characters  

This preserves semantic continuity and improves retrieval accuracy.

## Retrieval Strategy

- Embeddings are normalized  
- Inner-product similarity approximates cosine similarity  
- Top-k relevant chunks are retrieved per query  

## Grounded Answer Generation

The LLM is explicitly instructed to:

- Use only retrieved context  
- Avoid external knowledge  
- Return structured bullet-point responses  
- Indicate when information is unavailable  

This reduces hallucination and improves explainability.

---

## Frontend Interface

The system uses Streamlit to provide:

• Interactive query input  
• Display of retrieved context snippets  
• Generated grounded responses  

This transparency allows users to verify answer sources.

---

## How to Run Locally

1. Clone repository:
2.git clone <repo_link>
3.cd project

### 2. Install dependencies
```bash
pip install -r requirements.txt
```
### 3. Set your API key
Windows:
```bash
set OPENROUTER_API_KEY=your_api_key
```
Mac/Linux:
```bash
export OPENROUTER_API_KEY=your_api_key
```
### 4. Run the app
```bash
streamlit run app.py
 ```
5. Open in Browser

After running the command, the app will open automatically.
If not, open this in your browser:

http://localhost:8501
## Example Queries

• How are construction delays managed?  
• What is stage-based contractor payment?  
• How does quality assurance work?  
• What is escrow payment model?  

---
## Evaluation Framework

To assess the effectiveness of the Retrieval-Augmented Generation (RAG) system, a structured domain-specific evaluation was conducted. The evaluation focused on retrieval relevance, groundedness of generated responses, hallucination behaviour, and response clarity.

### Evaluation Question Set

#### Delivery Reliability & Delay Governance
1. In what ways does the platform proactively detect and handle construction schedule deviations?  
2. What operational systems are implemented to minimize the risk of delayed project completion?  
3. How does the construction management framework enforce accountability among execution teams?  
4. What mechanisms ensure that project timelines remain predictable and controlled?  

#### Financial Trust & Payment Structuring
5. How does the staged disbursement model improve financial transparency in construction projects?  
6. What role does escrow play in safeguarding customer investments during project execution?  
7. How are payment approvals linked to verified construction milestones?  
8. What systemic safeguards reduce financial uncertainty for homeowners?  

#### Quality Governance & Monitoring
9. How does the multi-checkpoint quality assurance system influence construction reliability?  
10. Which construction attributes are continuously evaluated to maintain structural standards?  
11. How does the platform translate quality metrics into actionable project oversight?  

#### Visibility & Process Transparency
12. What digital mechanisms enable real-time visibility into construction progress?  
13. How does the platform communicate execution performance to customers?  
14. What transparency principles differentiate this system from conventional contractors?  

#### Lifecycle Support & Customer Experience
15. How does the structured customer journey improve project decision-making?  
16. What forms of technical guidance are provided during architectural planning stages?  
17. How does post-construction support contribute to long-term asset reliability?  

#### Strategic Differentiation & Market Positioning
18. What systemic advantages distinguish this construction platform from traditional service providers?  
19. How does integrated project governance influence customer confidence?  
20. What value propositions reinforce trust throughout the construction lifecycle?  

---

### Evaluation Observations

1. The system demonstrated strong retrieval relevance due to semantic embedding-based search.  
2. Generated responses remained largely grounded in retrieved context, with minimal hallucination observed.  
3. Overlapping chunking improved contextual continuity, resulting in clearer and more structured answers.  
4. Response completeness varied when document coverage was limited. Future improvements may include adaptive chunking strategies and hybrid retrieval approaches.
## Bonus Experiment: Local vs API-based LLM Comparison

To evaluate the trade-offs between API-hosted instruction-tuned models and locally deployed open-source models in Retrieval-Augmented Generation (RAG), a controlled experiment was conducted using identical retrieval pipelines.

### Experimental Configuration

The following components were kept constant across both experimental settings:

- Document corpus: internal construction policy documents  
- Chunking strategy: sliding window (400 characters with 80 overlap)  
- Embedding model: all-MiniLM-L6-v2  
- Vector database: FAISS inner-product index  
- Retrieval strategy: top-k semantic similarity  

Only the answer generation model was varied.

Two configurations were evaluated:

1. API-based LLM via OpenRouter (DeepSeek Chat with LLaMA fallback)  
2. Locally deployed TinyLlama (1.1B parameters) using HuggingFace Transformers  

### Evaluation Metrics

The comparison focused on:

- Response groundedness to retrieved context  
- Instruction-following capability  
- Answer structure and clarity  
- Hallucination behaviour  
- Inference latency  

### Experimental Results

| Question | API Latency (s) | Local Latency (s) | API Quality | Local Quality |
|--------|----------------|-----------------|------------|--------------|
| Delay Management | ~6.3 | ~108 | Structured & grounded | Context repetition |
| Escrow Model | ~6.0 | ~36–50 | Clear explanation | Partial summarization |
| Quality Assurance | ~4–13 | ~68 | Accurate & concise | Generalized / hallucinated |

### Observations

The API-based model consistently generated well-structured, concise, and context-grounded answers. It demonstrated strong instruction-following behaviour and effective summarization of retrieved document segments.

The locally deployed TinyLlama model showed significantly higher inference latency on CPU and produced responses that were less structured. In several cases, the local model partially copied retrieved text or generated generalized domain statements not fully supported by context.

### Groundedness Analysis

API-generated responses maintained strong alignment with retrieved document chunks, whereas the local model exhibited weaker contextual reasoning and occasional hallucination tendencies.

### Practical Trade-offs

API-based models provide superior reasoning quality, structured outputs, and ease of deployment but depend on network connectivity and external services. Local models offer autonomy, data control, and offline deployment capability, but require substantial computational resources and prompt optimization to achieve comparable performance.

### How to Reproduce Experiment

To reproduce the local vs API-based LLM comparison experiment, follow the steps below:

1. Ensure the `data/` directory contains the same document corpus used in the primary RAG system.

2. Install required dependencies:

```bash
pip install sentence-transformers faiss-cpu transformers torch openai numpy
```

3. Configure the OpenRouter API key.

For Windows:

```bash
setx OPENROUTER_API_KEY "your_key"
```

For macOS / Linux:

```bash
export OPENROUTER_API_KEY="your_key"
```

4. Run the experiment script:

```bash
python experiment.py
```

This experiment performs semantic retrieval using FAISS, followed by answer generation using both API-hosted and locally deployed language models. It reports comparative latency metrics and qualitative differences in response grounding and instruction adherence.

### Experiment Conclusion
The experiment demonstrates that instruction-tuned API models currently provide stronger grounded reasoning performance for domain-specific RAG tasks compared to lightweight locally deployed models. However, local models remain viable for offline deployments and scenarios requiring full system control.

## Overall Project Conclusion

This project demonstrates the effectiveness of Retrieval-Augmented Generation in building reliable domain-specific AI assistants. By combining semantic retrieval with grounded language generation, the system produces accurate, explainable, and context-aware responses for construction knowledge queries.

The experimental comparison further highlights that while lightweight local models offer deployment flexibility, instruction-tuned API-based models currently provide superior reasoning performance for grounded question answering tasks.

