"""
ui/chat.py
----------
# * Query history panel — shows previous questions and allows reload.
"""

import html

import streamlit as st

from utils.logger import get_logger

log = get_logger(__name__)


def render_chat_history(session):
    """
    # * Render the query history expander.
    """
    history = session.get_chat_history()
    if not history:
        return

    user_messages = [msg for msg in history if msg.get("role") == "user"]
    if not user_messages:
        return

    with st.expander(f"Query History ({len(user_messages)} queries)", expanded=False):
        for i, msg in enumerate(reversed(user_messages[-10:])):
            question = msg.get("content", "")
            timestamp = msg.get("timestamp", "")

            c1, c2 = st.columns([5, 1])
            with c1:
                st.markdown(
                    f"""
                    <div class="ld-history-item">
                        <strong>Query {len(user_messages) - i}</strong>
                        <span>{html.escape(question)}</span><br>
                        <small>{html.escape(timestamp)}</small>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with c2:
                st.write("")
                if st.button("Reload", key=f"reload_{i}", use_container_width=True):
                    st.session_state["_q"] = question
                    session.set("pipeline_phase", None)
                    session.set("pending_intent", None)
                    session.set("pending_clarifier", None)
                    session.set("pending_slicers", {})
                    st.rerun()
