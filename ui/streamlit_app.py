# ui/streamlit_app.py
import streamlit as st
from app.orchestrate import answer

st.set_page_config(page_title="⚡ Ask the Grid (Local)", layout="centered")
st.title("⚡ Grid for the noob — Local Ollama model")
st.caption("Hybrid router: TOOL (numbers) • RAG (docs) • CHITCHAT")

if "history" not in st.session_state:
    st.session_state.history = []

q = st.chat_input("Ask about GB carbon intensity, forecasts, or energy concepts…")
if q:
    out = answer(q, model="llama3:instruct")  # change model if you pulled a different one
    st.session_state.history.append((q, out))

for q, out in st.session_state.history:
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        st.write(out["text"])
        st.caption(f"Route: {out['route']}")
        if out["citations"]:
            st.caption("Sources: " + " · ".join(out["citations"]))