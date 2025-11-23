import streamlit as st
from groq import Groq
from src.config.groq_key import GROQ_API_KEY


def loan_chatbot_ui(prediction_result=None, user_inputs=None):

    # ------------------ GLOBAL CHAT CSS ------------------
    st.markdown("""
    <style>

    .chat-box {
        padding: 10px;
        max-height: 450px;
        overflow-y: auto;
    }

    /* USER MESSAGE */
    .user-msg {
        background-color: #4CAF50;
        color: white;
        padding: 12px 16px;
        border-radius: 16px;
        max-width: 65%;
        float: right;
        clear: both;
        margin: 10px 0;
        font-size: 16px;
        line-height: 1.4;
    }

    /* BOT MESSAGE */
    .bot-msg {
        background-color: #2f2f2f;
        color: white;
        padding: 12px 16px;
        border-radius: 16px;
        max-width: 70%;
        float: left;
        clear: both;
        margin: 10px 0;
        font-size: 16px;
        line-height: 1.4;
    }

    /* Fix chat input */
    .stChatInput input {
        border: 2px solid #555 !important;
        background-color: #1c1c1c !important;
        color: white !important;
    }

    </style>
    """, unsafe_allow_html=True)
    # -----------------------------------------------------

    st.write("")  
    st.markdown("""
        <div style='text-align:center; margin-top: 0px; margin-bottom: 0px;'>
            <h3>💬 Loan Help Chatbot (Powered by Groq)</h3>
        </div>
    """, unsafe_allow_html=True)

    client = Groq(api_key=GROQ_API_KEY)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Context for chatbot
    if prediction_result is not None and user_inputs is not None:
        context = f"""
        The user received this result:
        Loan Status: {"APPROVED" if prediction_result == 1 else "REJECTED"}
        Applicant Details: {user_inputs}
        """
    else:
        context = "No loan prediction yet. Provide general loan guidance."

    # ---------------- CHAT INPUT (Below Title) ----------------
    user_message = st.chat_input("Ask anything about your loan or approval tips...")

    if user_message:
        st.session_state.chat_history.append(("user", user_message))

        try:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are a helpful AI loan assistant."},
                    {"role": "user", "content": context},
                    {"role": "user", "content": user_message},
                ],
            )
            bot_reply = response.choices[0].message.content

        except Exception as e:
            bot_reply = f"⚠ Error fetching response: {str(e)}"

        st.session_state.chat_history.append(("bot", bot_reply))

    # ---------------- MESSAGE DISPLAY AREA ----------------
    st.markdown("<div class='chat-box'>", unsafe_allow_html=True)

    for role, text in st.session_state.chat_history:
        if role == "user":
            st.markdown(f"<div class='user-msg'>You: {text}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='bot-msg'>Bot: {text}</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)