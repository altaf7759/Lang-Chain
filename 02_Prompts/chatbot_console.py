from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model="llama3-8b-8192", temperature=0.5)

chat_history = [
      SystemMessage(content="you are helpful AI assistance"),
]

while True:
      user_input = input("You: ")
      chat_history.append(HumanMessage(content=user_input))
      if user_input == "exit":
            break
      result = model.invoke(chat_history)
      print("AI: ", result.content)
      chat_history.append(AIMessage(content=result.content))