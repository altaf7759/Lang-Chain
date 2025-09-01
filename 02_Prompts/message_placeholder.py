from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
import json

load_dotenv()

model = ChatGroq(model="llama-3.1-8b-instant", temperature=0.9)

chat_template = ChatPromptTemplate([
      ("system", "you are a helpful customer support agent"),
      MessagesPlaceholder(variable_name="chat_history"),
      ("human", "{query}")
])

chat_history = []

with open("chat_history.txt", 'r') as f:
      raw_history = json.load(f)
      for msg in raw_history:
            if msg["type"] == "human":
                  chat_history.append(HumanMessage(content=msg["content"]))
            elif msg["type"] == "ai":
                  chat_history.append(AIMessage(content=msg["content"]))

while True:
      query = input("You: ")
      if query.lower() in ["quit", "exit"]:
            break
      prompt = chat_template.invoke({"chat_history": chat_history, "query": query})
      response = model.invoke(prompt)
      print(f"AI: {response.content}")

      chat_history.append(HumanMessage(content=query))
      chat_history.append(AIMessage(content=response.content))


all_history =[]

for m in chat_history:
      if isinstance(m, HumanMessage):
            all_history.append({"type": "human", "content": m.content})
      elif isinstance(m, AIMessage):
            all_history.append({"type": "ai", "content": m.content})

with open("chat_history.txt", "w") as f:
      json.dump(all_history, f, indent=2)