from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

chat_template = ChatPromptTemplate([
      ("system", "You are a helpful {domain} expert"),
      ("human", "Explain in simple terms, what is {topic}")
])

prompt = chat_template.invoke({"domain": "cricket", "topic": "run out"})

model = ChatGroq(model="llama-3.1-8b-instant", temperature=0.5)

result = model.invoke(prompt)

print(result.content)