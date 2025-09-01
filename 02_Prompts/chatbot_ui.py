import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv
import os
import json

load_dotenv()

model = ChatGroq(model="llama-3.1-8b-instant", temperature=0.9)

CHAT_FILE = "chatbot_history.json"

def load_history():
    if os.path.exists(CHAT_FILE) and os.path.getsize(CHAT_FILE) > 0:
        with open(CHAT_FILE, 'r', encoding="utf-8") as f:
            data = json.load(f)
        history = []
        for msg in data:
            if msg["role"] == "user":
                history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistance":
                history.append(AIMessage(content=msg["content"]))
        return history
    else:
        return []


def save_history(history):
    data = []
    for msg in history:
        if isinstance(msg, HumanMessage):
            data.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            data.append({"role": "assistance", "content": msg.content})
    with open(CHAT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

st.set_page_config(page_title="Chatbot", page_icon="🤖")
st.title("💬 AI Chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_history()

chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistance"),
    MessagesPlaceholder(variable_name='history'),
    ("user", "{query}")
])

for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.write(msg.content)

user_input = st.chat_input("Type your message...")

if user_input:
    st.session_state.chat_history.append(HumanMessage(content=user_input))

    prompt = chat_template.invoke({"history": st.session_state.chat_history, "query": user_input})

    result = model.invoke(prompt)

    st.session_state.chat_history.append(AIMessage(content=result.content))

    with st.chat_message("user"):
        st.write(user_input)
    with st.chat_message("assistant"):
        st.write(result.content)
    
    save_history(st.session_state.chat_history)
