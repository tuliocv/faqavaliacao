import os
import re
import csv
import hmac
from datetime import datetime, timezone
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import faiss
from filelock import FileLock
from docx import Document
from openai import OpenAI


# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Chat - Guia E2A", page_icon="💬", layout="wide")
st.title("💬 Chat de Dúvidas — Avaliação E2A")

DOC_PATH = "docs/Guia de Respostas para o Chat - Avaliação E2A.docx"

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

DATA_DIR = "data"
LOG_CSV = os.path.join(DATA_DIR, "faq_log.csv")
LOG_LOCK = os.path.join(DATA_DIR, "faq_log.lock")

os.makedirs(DATA_DIR, exist_ok=True)


# =========================
# Secrets / Credenciais
# =========================
api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("Defina OPENAI_API_KEY em Secrets (Streamlit Cloud) ou variável de ambiente.")
    st.stop()

ADMIN_USER = st.secrets.get("ADMIN_USER") or os.getenv("ADMIN_USER") or "admin"
ADMIN_PASS = st.secrets.get("ADMIN_PASS") or os.getenv("ADMIN_PASS") or ""

client = OpenAI(api_key=api_key)


# =========================
# Helpers: Admin auth
# =========================
def is_admin() -> bool:
    return bool(st.session_state.get("admin_logged_in", False))

def admin_login_panel():
    st.markdown("### 🔒 Área administrativa")

    if is_admin():
        st.success("Admin autenticado.")
        if st.button("Sair (logout)", use_container_width=True):
            st.session_state.admin_logged_in = False
            st.rerun()
        return

    if not ADMIN_PASS:
        st.warning("ADMIN_PASS não foi definido em Secrets. Configure para ativar o login.")
        return

    with st.form("admin_login", clear_on_submit=False):
        u = st.text_input("Usuário", value="", placeholder="admin")
        p = st.text_input("Senha", value="", type="password")
        ok = st.form_submit_button("Entrar")

    if ok:
        user_ok = hmac.compare_digest(u.strip(), str(ADMIN_USER).strip())
        pass_ok = hmac.compare_digest(p, str(ADMIN_PASS))
        if user_ok and pass_ok:
            st.session_state.admin_logged_in = True
            st.success("Login realizado.")
            st.rerun()
        else:
            st.error("Usuário ou senha inválidos.")


# =========================
# DOCX -> texto
# =========================
def read_docx_text(path: str) -> str:
    doc = Document(path)
    parts = []
    for p in doc.paragraphs:
        t = (p.text or "").strip()
        if t:
            t = re.sub(r"[ \t]+", " ", t)
            parts.append(t)
    return "\n".join(parts).strip()

def chunk_text(text: str, chunk_chars: int = 900, overlap: int = 150) -> List[Dict]:
    chunks = []
    start = 0
    n = len(text)
    chunk_id = 0
    while start < n:
        end = min(n, start + chunk_chars)
        chunk = text[start:end].strip()
        if chunk:
            chunk_id += 1
            chunks.append({"chunk_id": chunk_id, "text": chunk})
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks

def embed_texts(texts: List[str]) -> np.ndarray:
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    vecs = np.array([d.embedding for d in resp.data], dtype="float32")
    faiss.normalize_L2(vecs)
    return vecs

def build_index(vectors: np.ndarray) -> faiss.Index:
    dim = vectors.shape[1]
    idx = faiss.IndexFlatIP(dim)
    idx.add(vectors)
    return idx

def retrieve(query: str, chunks: List[Dict], index: faiss.Index, k: int = 5) -> List[Dict]:
    qvec = embed_texts([query])
    scores, ids = index.search(qvec, k)
    out = []
    for score, i in zip(scores[0], ids[0]):
        if i == -1:
            continue
        c = chunks[int(i)]
        out.append({**c, "score": float(score)})
    return out


# =========================
# Resposta padronizada (curta, institucional)
# =========================
def answer_institutional(question: str, retrieved: List[Dict]) -> str:
    context = "\n\n".join([f"[Trecho {r['chunk_id']}] {r['text']}" for r in retrieved])

    system = (
        "Você é um atendente institucional. Responda APENAS com base no GUIA fornecido.\n"
        "Formato obrigatório:\n"
        "1) Resposta curta e objetiva (máx. 3 frases).\n"
        "2) Se houver ação para o aluno, inclua uma linha '✅ O que fazer:' (uma única orientação).\n"
        "3) Sempre inclua fonte no final: 'Fonte: (Trecho X, Trecho Y)'.\n"
        "4) Se não estiver no guia, diga: 'Não encontrei essa informação no guia.' e finalize com 'Fonte: -'.\n"
        "Regras:\n"
        "- Não invente, não use conhecimento externo.\n"
        "- Tom: cordial, formal e direto.\n"
    )

    resp = client.responses.create(
        model=CHAT_MODEL,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"GUIA:\n{context}\n\nPERGUNTA:\n{question}"}
        ],
    )
    return resp.output_text.strip()


# =========================
# Logging CSV
# =========================
def ensure_log_header():
    if not os.path.exists(LOG_CSV):
        with FileLock(LOG_LOCK):
            if not os.path.exists(LOG_CSV):
                with open(LOG_CSV, "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(["timestamp_utc", "category", "question", "top_chunks", "top_scores"])

def append_log(category: str, question: str, retrieved: List[Dict]):
    ensure_log_header()
    ts = datetime.now(timezone.utc).isoformat()
    top_chunks = ",".join([str(r["chunk_id"]) for r in retrieved[:5]])
    top_scores = ",".join([f"{r['score']:.4f}" for r in retrieved[:5]])
    with FileLock(LOG_LOCK):
        with open(LOG_CSV, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([ts, category, question, top_chunks, top_scores])

def load_log_df() -> pd.DataFrame:
    if not os.path.exists(LOG_CSV):
        return pd.DataFrame(columns=["timestamp_utc", "category", "question", "top_chunks", "top_scores"])
    return pd.read_csv(LOG_CSV)

def normalize_question(q: str) -> str:
    q = (q or "").strip().lower()
    q = re.sub(r"\s+", " ", q)
    return q


# =========================
# Indexação 1x (cache)
# =========================
@st.cache_resource(show_spinner=True)
def load_knowledge_base() -> Tuple[str, List[Dict], faiss.Index]:
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
# Quick Buttons
# =========================
QUICK = [
    ("A1", "Quais são as regras e orientações para a avaliação A1?"),
    ("A2", "Como funciona a avaliação A2 e quais são as orientações para realizar?"),
    ("A3", "Como funciona a avaliação A3 e quais são as orientações?"),
    ("2ª oportunidade", "Como solicitar a segunda oportunidade e quais prazos e regras se aplicam?"),
    ("A3 online", "A A3 é online ou presencial? Em quais casos a A3 pode ser online?"),
    ("nota não lançada", "Minha nota ainda não foi lançada. Qual o prazo e o que devo fazer?"),
    ("Lista de presença", "Qual o endereço para acessar a lista de presença das UCs assíncronas ?"), 
    ("Modelo de Ata", "Qual o link para acesso do Modelo de Ata Lista? ")
]


# =========================
# Sidebar
# =========================
with st.sidebar:
    st.subheader("📚 Base carregada")
    st.write(f"**Arquivo:** `{DOC_PATH}`")
    st.write(f"**Trechos indexados:** {len(chunks)}")

    st.divider()
    st.subheader("⚙️ Ajustes")
    k = st.slider("Trechos usados (k)", 3, 10, 5)
    show_sources = st.toggle("Mostrar trechos usados", value=False)

    st.divider()
    st.subheader("📌 Atalhos")
    st.caption("Clique para disparar uma pergunta padrão:")

    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None
        st.session_state.pending_category = "Manual"

    cols = st.columns(2)
    for i, (label, q) in enumerate(QUICK):
        with cols[i % 2]:
            if st.button(label, use_container_width=True):
                st.session_state.pending_question = q
                st.session_state.pending_category = label

    st.divider()

    # ---------- Admin area ----------
    admin_login_panel()

    if is_admin():
        st.divider()
        st.subheader("📤 Logs (Admin)")

        df_log = load_log_df()
        if df_log.empty:
            st.caption("Ainda sem registros.")
        else:
            # Top FAQ rápido (visual)
            df_tmp = df_log.copy()
            df_tmp["q_norm"] = df_tmp["question"].astype(str).apply(normalize_question)
            top = df_tmp.groupby("q_norm").size().sort_values(ascending=False).head(10)
            st.write(top.reset_index(name="qtd").rename(columns={"q_norm": "pergunta"}))

            # Exportar CSV (download)
            csv_bytes = df_log.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Exportar log (CSV)",
                data=csv_bytes,
                file_name="faq_log.csv",
                mime="text/csv",
                use_container_width=True
            )

            # Zerar log (opcional)
            if st.button("🗑️ Zerar log", type="secondary", use_container_width=True):
                with FileLock(LOG_LOCK):
                    if os.path.exists(LOG_CSV):
                        os.remove(LOG_CSV)
                st.success("Log zerado.")
                st.rerun()


# =========================
# Chat state
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])


# =========================
# Entrada de pergunta (manual ou botão)
# =========================
user_typed = st.chat_input("Digite sua dúvida (A1, A2, A3, prazos, 2ª oportunidade...)")

question = None
category = "Manual"

if st.session_state.get("pending_question"):
    question = st.session_state.pending_question
    category = st.session_state.pending_category
    st.session_state.pending_question = None
    st.session_state.pending_category = "Manual"
elif user_typed:
    question = user_typed
    category = "Manual"

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.spinner("Buscando no guia..."):
        retrieved = retrieve(question, chunks, index, k=k)

    if show_sources:
        with st.expander("📌 Trechos usados (fontes)"):
            for r in retrieved:
                st.markdown(f"**Trecho {r['chunk_id']}** (score {r['score']:.3f})")
                st.write(r["text"])
                st.divider()

    with st.spinner("Gerando resposta..."):
        ans = answer_institutional(question, retrieved)

    st.session_state.messages.append({"role": "assistant", "content": ans})
    with st.chat_message("assistant"):
        st.markdown(ans)

    # Log (não exige admin; admin só exporta)
    try:
        append_log(category=category, question=question, retrieved=retrieved)
    except Exception:
        pass
