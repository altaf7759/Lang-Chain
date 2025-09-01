from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

model = ChatAnthropic("cloud-3-5-sonnet-20241022", temperature=0.9)

result = model.invoke("prime minister of india")

print(result.content)