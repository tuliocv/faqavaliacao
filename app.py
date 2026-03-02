import os
import re
import numpy as np
import streamlit as st
import faiss
from docx import Document
from openai import OpenAI

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Chat - Guia E2A", page_icon="💬", layout="wide")
st.title("💬 Chat de Dúvidas (baseado no Guia E2A)")

DOC_PATH = "docs/Guia de Respostas para o Chat - Avaliação E2A.docx"
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"  # pode trocar

# =========================
# OpenAI
# =========================
api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("Defina OPENAI_API_KEY em Secrets (Streamlit Cloud) ou variável de ambiente.")
    st.stop()

client = OpenAI(api_key=api_key)

# =========================
# Leitura DOCX
# =========================
def read_docx_text(path: str) -> str:
    doc = Document(path)
    parts = []
    for p in doc.paragraphs:
        t = (p.text or "").strip()
        if t:
            # normaliza espaços
            t = re.sub(r"[ \t]+", " ", t)
            parts.append(t)
    return "\n".join(parts).strip()

def chunk_text(text: str, chunk_chars=900, overlap=150) -> list[dict]:
    # chunks simples por caracteres (bom o suficiente p/ 1 doc curto)
    chunks = []
    start = 0
    n = len(text)
    i = 0
    while start < n:
        end = min(n, start + chunk_chars)
        chunk = text[start:end].strip()
        if chunk:
            i += 1
            chunks.append({"chunk_id": i, "text": chunk})
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks

def embed_texts(texts: list[str]) -> np.ndarray:
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    vecs = np.array([d.embedding for d in resp.data], dtype="float32")
    faiss.normalize_L2(vecs)  # para cosine similarity
    return vecs

def build_index(vectors: np.ndarray) -> faiss.Index:
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    return index

def retrieve(query: str, chunks: list[dict], index: faiss.Index, k=5) -> list[dict]:
    qvec = embed_texts([query])
    scores, ids = index.search(qvec, k)
    out = []
    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:
            continue
        c = chunks[int(idx)]
        out.append({**c, "score": float(score)})
    return out

def answer(question: str, retrieved: list[dict]) -> str:
    # Monta contexto com “IDs” de trecho para citar
    context = "\n\n".join([f"[Trecho {r['chunk_id']}] {r['text']}" for r in retrieved])

    system = (
        "Você é um atendente virtual e deve responder APENAS usando o GUIA fornecido.\n"
        "Regras:\n"
        "1) Se a resposta não estiver no guia, diga: 'Não encontrei essa informação no guia.'\n"
        "2) Sempre cite a fonte no formato (Trecho X).\n"
        "3) Não invente e não use conhecimento externo.\n"
        "4) Seja objetivo e prático.\n"
    )

    resp = client.responses.create(
        model=CHAT_MODEL,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"GUIA:\n{context}\n\nPERGUNTA:\n{question}"}
        ],
    )
    return resp.output_text

# =========================
# Indexação 1x (cache)
# =========================
@st.cache_resource(show_spinner=True)
def load_knowledge_base():
    if not os.path.exists(DOC_PATH):
        raise FileNotFoundError(f"Arquivo não encontrado: {DOC_PATH}")

    full_text = read_docx_text(DOC_PATH)
    if not full_text:
        raise ValueError("O DOCX está vazio ou não foi possível extrair o texto.")

    chunks = chunk_text(full_text)
    vectors = embed_texts([c["text"] for c in chunks])
    index = build_index(vectors)
    return full_text, chunks, index

try:
    full_text, chunks, index = load_knowledge_base()
except Exception as e:
    st.error(f"Erro ao carregar a base: {e}")
    st.stop()

# =========================
# UI
# =========================
with st.sidebar:
    st.subheader("📚 Base carregada")
    st.write(f"**Arquivo:** `{DOC_PATH}`")
    st.write(f"**Trechos:** {len(chunks)}")
    k = st.slider("Trechos usados (k)", 3, 10, 5)
    show_sources = st.toggle("Mostrar trechos usados", value=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

q = st.chat_input("Digite sua dúvida (A1, A2, A3, prazos, 2ª oportunidade...)")
if q:
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)

    with st.spinner("Buscando no guia..."):
        retrieved = retrieve(q, chunks, index, k=k)

    if show_sources:
        with st.expander("📌 Trechos usados (fontes)"):
            for r in retrieved:
                st.markdown(f"**Trecho {r['chunk_id']}** (score {r['score']:.3f})")
                st.write(r["text"])
                st.divider()

    with st.spinner("Gerando resposta..."):
        ans = answer(q, retrieved)

    st.session_state.messages.append({"role": "assistant", "content": ans})
    with st.chat_message("assistant"):
        st.markdown(ans)
