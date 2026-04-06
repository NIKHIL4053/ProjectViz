"""
ui/chat.py
----------
# * Query history panel — shows previous questions and their metrics.
# * Lets user click a previous question to reload it.
# * Shown in an expander at the bottom of the dashboard.
"""

import streamlit as st

from utils.logger import get_logger

log = get_logger(__name__)


def render_chat_history(session):
    """
    # * Render the query history expander.
    # * Shows last N questions with their metric and timestamp.
    # * Clicking a previous question reloads it into the input.

    Args:
        session : SessionManager instance from get_session()
    """
    history = session.get_chat_history()
    if not history:
        return

    # * Only show user messages (skip assistant replies)
    user_messages = [
        msg for msg in history
        if msg.get("role") == "user"
    ]

    if not user_messages:
        return

    with st.expander(
        f"🕐 Query History ({len(user_messages)} queries)",
        expanded=False,
    ):
        for i, msg in enumerate(reversed(user_messages[-10:])):
            question  = msg.get("content", "")
            timestamp = msg.get("timestamp", "")

            c1, c2 = st.columns([5, 1])
            with c1:
                st.markdown(
                    f"**{len(user_messages) - i}.** {question}  \n"
                    f"<small style='color:#6c7086'>{timestamp}</small>",
                    unsafe_allow_html=True,
                )
            with c2:
                if st.button("↩ Reload", key=f"reload_{i}", use_container_width=True):
                    st.session_state["_q"] = question
                    # * Reset pipeline so this question runs fresh
                    session.set("pipeline_phase",    None)
                    session.set("pending_intent",    None)
                    session.set("pending_clarifier", None)
                    session.set("pending_slicers",   {})
                    st.rerun()

            if i < len(user_messages) - 1:
                st.divider()