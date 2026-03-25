import streamlit as st
import os
import numpy as np
import faiss
import time
from sentence_transformers import SentenceTransformer
from openai import OpenAI

st.set_page_config(page_title="AI Construction Assistant")

st.title("AI Construction Knowledge Assistant")

# ----------- LOAD DATA + INDEX ----------- #
@st.cache_resource
def initialize_rag():

    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    corpus = []
    for fname in os.listdir("data"):
        with open(f"data/{fname}", "r", encoding="utf-8") as f:
            corpus.append(f.read())

    def split_chunks(text, size=400, overlap=80):
        chunks = []
        start = 0
        while start < len(text):
            end = start + size
            chunks.append(text[start:end])
            start += size - overlap
        return chunks

    knowledge_base = []
    for doc in corpus:
        knowledge_base.extend(split_chunks(doc))

    vectors = embedder.encode(knowledge_base, normalize_embeddings=True)

    idx = faiss.IndexFlatIP(vectors.shape[1])
    idx.add(np.array(vectors))

    return embedder, knowledge_base, idx

encoder, kb_chunks, faiss_index = initialize_rag()

# ----------- RETRIEVAL ----------- #
def semantic_search(question, topk=3):
    q_vec = encoder.encode([question], normalize_embeddings=True)
    scores, ids = faiss_index.search(q_vec, topk)
    return [kb_chunks[i] for i in ids[0]]

# ----------- LLM ----------- #
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

def generate_response(context_list, user_q):

    joined_context = "\n\n".join(context_list)

    system_prompt = f"""
You are a domain assistant for construction services.

Use ONLY the provided reference text.
If unsure → say information unavailable.
Provide structured bullet point answers.

Reference Text:
{joined_context}

User Question:
{user_q}
"""

    llm_choices = [
        "deepseek/deepseek-chat",
        "meta-llama/llama-3-8b-instruct"
    ]

    for model_name in llm_choices:
        try:
            reply = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": system_prompt}],
                temperature=0.1
            )
            return reply.choices[0].message.content
        except:
            time.sleep(1)

    return "Model currently unavailable."

# ----------- UI ----------- #
question = st.text_input("Ask something about construction policies")

if question:
    context_hits = semantic_search(question)
    answer_text = generate_response(context_hits, question)

    st.markdown("### Supporting Context")
    for i, txt in enumerate(context_hits):
        st.write(f"Snippet {i+1}:")
        st.info(txt)

    st.markdown("### AI Answer")
    st.success(answer_text)