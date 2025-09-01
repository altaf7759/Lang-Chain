from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")

text = "Delhi is the capital of india"

vector = embedding.embed_query(text)

print(str(vector))

