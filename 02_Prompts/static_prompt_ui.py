from langchain_groq import ChatGroq
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

model = ChatGroq(model="llama3-8b-8192", temperature=0.5)

st.header("Research Tool")

user_input = st.text_input("Enter your prompt")

if st.button("Summarize"):
      if user_input.strip() != "":
        result = model.invoke(user_input)
        st.write(result.content)
      else:
        st.warning("Please enter a prompt first.")