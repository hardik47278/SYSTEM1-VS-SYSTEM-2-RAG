import hashlib
from pathlib import Path
import streamlit as st
from pyprojroot import here
from uuid import uuid4  # ✅ session id
import sys
from io import BytesIO
import pandas as pd

from data_vis import (
    init_gemini_sync,
    run_csv_eda_sync,
    cleanup_files,
)

# ✅ ensure project root is on sys.path (so `src.*` imports inside pipeline.py work)
ROOT = Path(__file__).resolve().parents[1]  # D:\ragg\RAG-Challenge-2
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline import Pipeline, max_nst_o3m_config
from question_processing_copy import QuestionsProcessor
from upload import (
    create_file_index,
    stream_answer_from_uploaded_file,
    answer_from_uploaded_file,
    record_user_input,
    extract_text,
    generate_audio,
    
)

st.set_page_config(page_title="Financial RAG QA", layout="wide")
st.title("📊 Annual Report Question Answering (Auto Routing + Voice)")

# -----------------------------
# OFFLINE PROCESSOR (CSV routing)
# -----------------------------
@st.cache_resource
def load_offline_processor():
    root_path = here() / "data" / "test_set"
    pipeline = Pipeline(root_path, run_config=max_nst_o3m_config)

    processor = QuestionsProcessor(
        vector_db_dir=pipeline.paths.vector_db_dir,
        documents_dir=pipeline.paths.documents_dir,
        subset_path=pipeline.paths.subset_path,
        new_challenge_pipeline=True,
        parent_document_retrieval=True,
        llm_reranking=True,
        answering_model="o3-mini-2025-01-31",
    )
    return processor


processor = load_offline_processor()

# -----------------------------
# SESSION STATE
# -----------------------------
if "upload_index_key" not in st.session_state:
    st.session_state.upload_index_key = None
if "upload_filename" not in st.session_state:
    st.session_state.upload_filename = None
if "upload_file_hash" not in st.session_state:
    st.session_state.upload_file_hash = None

# ✅ session id for LangGraph memory
if "chat_session_id" not in st.session_state:
    st.session_state.chat_session_id = str(uuid4())

# ✅ chat UI history
if "ui_chat" not in st.session_state:
    st.session_state.ui_chat = []  # [{"role","content","audio","meta"}]

# ✅ CSV EDA state (NEW)
if "csv_eda_result" not in st.session_state:
    st.session_state.csv_eda_result = None

# -----------------------------
# UPLOAD INDEX (auto build + cached)
# -----------------------------
@st.cache_resource
def build_upload_index_cached(file_hash: str, filename: str, file_bytes: bytes) -> str:
    tmp_dir = Path("tmp_uploads")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    safe_name = filename.replace(" ", "_")
    file_path = tmp_dir / f"{file_hash}_{safe_name}"
    file_path.write_bytes(file_bytes)

    idx_or_key = create_file_index(str(file_path), filename, base_dir="file_indexes")
    return idx_or_key


st.sidebar.header("📄 Optional: Upload fallback")

uploaded_file = st.sidebar.file_uploader(
    "Upload PDF/DOCX/PPTX/Excel/CSV",
    type=["pdf", "docx", "pptx", "xlsx", "xls", "csv"],
)

file_bytes = None  # ✅ keep defined even if no upload (NEW)

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.sha256(file_bytes).hexdigest()[:16]

    if st.session_state.upload_file_hash != file_hash:
        with st.sidebar.spinner("Indexing uploaded file (one-time)..."):
            idx_or_key = build_upload_index_cached(file_hash, uploaded_file.name, file_bytes)

        st.session_state.upload_index_key = idx_or_key
        st.session_state.upload_filename = uploaded_file.name
        st.session_state.upload_file_hash = file_hash
        st.sidebar.success("File indexed ✅")

if st.session_state.upload_index_key:
    st.sidebar.info(f"Active fallback:\n`{st.session_state.upload_filename}`")

# -----------------------------
# CSV EDA (NEW) — ONLY when CSV uploaded
# -----------------------------
if uploaded_file is not None and uploaded_file.name.lower().endswith(".csv") and file_bytes is not None:
    st.sidebar.markdown("---")
    st.sidebar.subheader("📈 CSV Analyzer (EDA + Gemini Vision)")

    gem_ok, gem_msg = init_gemini_sync()
    st.sidebar.caption(gem_msg)

    run_eda = st.sidebar.button("Run CSV Analysis", use_container_width=True)
    clear_eda = st.sidebar.button("Clear CSV Report", use_container_width=True)

    if clear_eda:
        try:
            if st.session_state.csv_eda_result:
                cleanup_files(st.session_state.csv_eda_result.get("tmp_files", []))
        except Exception:
            pass
        st.session_state.csv_eda_result = None
        st.sidebar.success("Cleared ✅")
        st.rerun()

    if run_eda:
        with st.sidebar.spinner("Running EDA + Gemini insights..."):
            df = pd.read_csv(BytesIO(file_bytes))
            eda_result = run_csv_eda_sync(df, gemini_available=gem_ok)
        st.session_state.csv_eda_result = eda_result
        st.sidebar.success("CSV analysis ready ✅")

show_debug = st.toggle("Show debug", value=False)

# -----------------------------
# MAIN UI: show CSV EDA result (NEW)
# -----------------------------
if st.session_state.get("csv_eda_result"):
    r = st.session_state["csv_eda_result"]

    with st.expander("📈 CSV Analysis Report (EDA + Gemini)", expanded=True):
        st.markdown("### Dataset Context")
        st.markdown(r.get("context_md", ""))

        st.markdown("### Visuals")
        for title, path in r.get("visuals", []):
            st.image(path, caption=title, use_container_width=True)

        st.markdown("### AI Plan (Gemini)")
        st.markdown(r.get("plan", ""))

        st.markdown("### AI Summary (Gemini)")
        st.markdown(r.get("summary", ""))

        st.markdown("### AI Chart Explanations (Gemini Vision)")
        for title, explanation in r.get("vision_notes", []):
            st.markdown(f"**{title}:** {explanation}")

        with st.expander("CSV EDA Raw JSON", expanded=False):
            st.json(r)

# -----------------------------
# UI: metadata renderer (citations/cache always shown)
# -----------------------------
def _pick_final_answer(d: dict) -> str:
    return (d.get("final_answer") or d.get("value") or d.get("answer") or "").strip()


def render_meta(meta: dict | None, always_json_title: str | None = None):
    if not meta or not isinstance(meta, dict):
        return

    # ✅ cache label always (if present)
    if "cached" in meta:
        st.caption(f"Cache: {'HIT ✅' if bool(meta.get('cached')) else 'MISS'}")

    # ✅ precision if present
    if meta.get("context_precision") is not None:
        try:
            st.caption(f"RAGAS Context Precision: {float(meta['context_precision']):.4f}")
        except Exception:
            pass

    # ✅ applied filters if present
    if meta.get("applied_filters"):
        with st.expander("Applied Filters", expanded=False):
            st.json(meta.get("applied_filters"))

    # ✅ citations/sources ALWAYS visible (even if empty)
    with st.expander("Citations / Retrieved Chunks", expanded=False):
        st.json(meta.get("sources", []))

    # ✅ raw JSON (always for offline/online if you want)
    if always_json_title:
        with st.expander(always_json_title, expanded=False):
            st.json(meta)

    if show_debug:
        with st.expander("Debug: raw meta", expanded=False):
            st.json(meta)


# -----------------------------
# RENDER CHAT HISTORY
# -----------------------------
for msg in st.session_state.ui_chat:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("audio") is not None:
            st.audio(msg["audio"], format="audio/wav")
        render_meta(msg.get("meta"))

# -----------------------------
# INPUTS (chat + voice)
# -----------------------------
col1, col2 = st.columns([6, 1])
with col2:
    voice_btn = st.button("🎙 Voice")

user_prompt = st.chat_input("Ask a question...")

# voice -> same pipeline input
if voice_btn:
    with st.chat_message("user"):
        st.info("Recording...")
        audio_buffer = record_user_input(duration=5)
        st.audio(audio_buffer, format="audio/wav")

        st.info("Transcribing...")
        transcribed_text = extract_text(audio_buffer)
        st.markdown(f"**You said:** {transcribed_text}")

    user_prompt = transcribed_text

# -----------------------------
# AUTO ROUTING LOGIC (UNCHANGED)
# -----------------------------
def company_match_exists(q: str) -> bool:
    try:
        matches = processor._extract_companies_from_subset(q)
        return len(matches) > 0
    except Exception:
        return False


# -----------------------------
# MAIN RUN (chat UI) — ROUTING LOGIC UNCHANGED
# -----------------------------
if user_prompt and user_prompt.strip():
    question = user_prompt.strip()

    # append + show user
    st.session_state.ui_chat.append({"role": "user", "content": question, "audio": None, "meta": None})
    with st.chat_message("user"):
        st.markdown(question)

    assistant_container = st.chat_message("assistant")
    assistant_stream_box = assistant_container.empty()
    status_box = st.empty()

    with st.spinner("Routing + answering..."):
        try:
            # 1) OFFLINE ROUTE
            if company_match_exists(question):
                route_used = "OFFLINE (company matched CSV)"
                answer = processor.process_question(question=question)

                if isinstance(answer, dict):
                    final_answer = _pick_final_answer(answer)
                    assistant_stream_box.markdown(final_answer)
                else:
                    final_answer = str(answer)
                    assistant_stream_box.markdown(final_answer)
                    answer = {"final_answer": final_answer, "sources": []}  # UI-safe

                # voice
                audio_reply = None
                if final_answer:
                    status_box.info("Generating voice response...")
                    audio_reply = generate_audio(final_answer)

                # ✅ citations always + OFFLINE raw JSON always
                with assistant_container:
                    render_meta(answer, always_json_title="OFFLINE Raw JSON (always)")

                # persist
                st.session_state.ui_chat.append(
                    {"role": "assistant", "content": final_answer, "audio": audio_reply, "meta": answer}
                )

                status_box.success(f"Done ✅ | Route: {route_used}")

            # 2) UPLOAD STREAM ROUTE
            elif st.session_state.upload_index_key:
                route_used = "UPLOAD fallback (LangGraph STREAM)"
                final_payload = None
                live_text = ""
                last_filters = {}

                for upd in stream_answer_from_uploaded_file(
                    question=question,
                    file_index=st.session_state.upload_index_key,
                    session_id=st.session_state.chat_session_id,
                ):
                    evt = upd.get("event")

                    if evt == "cached":
                        live_text = upd.get("final_answer", "") or ""
                        assistant_stream_box.markdown(live_text)
                        final_payload = upd

                    elif evt == "retrieved":
                        status_box.info("Retrieved context… generating answer…")
                        st.sidebar.subheader("applied filters")
                        st.sidebar.json(upd.get("applied_filters", {}))

                    elif evt == "answer":
                        live_text = upd.get("final_answer", "") or ""
                        assistant_stream_box.markdown(live_text)

                    elif evt == "done":
                        final_payload = upd
                        last_filters = upd.get("applied_filters", {}) or {}

                # voice
                audio_reply = None
                if live_text:
                    status_box.info("Generating voice response...")
                    audio_reply = generate_audio(live_text)
                    with assistant_container:
                        st.audio(audio_reply,format="audio/mp3")

                meta = {
                    "final_answer": live_text,
                    "sources": (final_payload or {}).get("sources", []),
                    "context_precision": (final_payload or {}).get("context_precision", 0.0),
                    "cached": bool((final_payload or {}).get("cached", False)),
                    "applied_filters": last_filters,
                }

                # ✅ citations always
                with assistant_container:
                    render_meta(meta, always_json_title="UPLOAD Raw JSON (always)")

                st.session_state.ui_chat.append(
                    {"role": "assistant", "content": live_text, "audio": audio_reply, "meta": meta}
                )

                # ✅ cache label in status too
                if meta.get("cached"):
                    status_box.success(f"Done ✅ | Route: {route_used} | Cache: HIT ✅")
                else:
                    status_box.success(f"Done ✅ | Route: {route_used} | Cache: MISS")

            # 3) ONLINE ROUTE
            else:
                route_used = "ONLINE fallback"
                answer = processor.openai_processor.get_answer_online(
                    question=question,
                    model=processor.answering_model,
                )

                if isinstance(answer, dict):
                    final_answer = _pick_final_answer(answer)
                    assistant_stream_box.markdown(final_answer)
                else:
                    final_answer = str(answer)
                    assistant_stream_box.markdown(final_answer)
                    answer = {"final_answer": final_answer, "sources": []}  # UI-safe

                # voice
                audio_reply = None
                if final_answer:
                    status_box.info("Generating voice response...")
                    audio_reply = generate_audio(final_answer)

                # ✅ citations always + ONLINE raw JSON always
                with assistant_container:
                    render_meta(answer, always_json_title="ONLINE Raw JSON (always)")

                st.session_state.ui_chat.append(
                    {"role": "assistant", "content": final_answer, "audio": audio_reply, "meta": answer}
                )

                status_box.success(f"Done ✅ | Route: {route_used}")

        except Exception as e:
            status_box.error(str(e))
            if show_debug:
                st.exception(e)
