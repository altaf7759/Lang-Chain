from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from dotenv import load_dotenv

load_dotenv()

documents = [
    "Virat Kohli is one of India’s greatest batsmen, known for his aggressive batting style and consistency across all formats. He has scored over 70 international centuries and led India to many memorable victories.",
    
    "Rohit Sharma, the current captain of the Indian cricket team, is famous for his elegant stroke play and ability to score big hundreds. He is the only cricketer to score three double centuries in ODIs.",
    
    "Sachin Tendulkar, often called the 'God of Cricket', is the highest run-scorer in international cricket. With 100 international centuries, his career has inspired millions of cricket fans worldwide.",
    
    "MS Dhoni is regarded as one of the best captains in cricket history. Known for his calmness under pressure, he led India to the 2007 T20 World Cup, 2011 ODI World Cup, and 2013 Champions Trophy victories.",
    
    "Jasprit Bumrah is India’s leading fast bowler, renowned for his unorthodox action and deadly yorkers. He has played a crucial role in India’s success in Test and limited-overs cricket in recent years."
]

query = "greatest captain"

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")

document_embeddings = embeddings.embed_documents(documents)
query_embedding = embeddings.embed_query(query)

scores = cosine_similarity([query_embedding], document_embeddings)[0]

index, score = sorted(list(enumerate(scores)), key=lambda x:x[1])[-1]

print("*" * 50)
print(f"Query: {query} \n")
print(f"Response: {documents[index]} \n")
print(f"Similarity score is: {score}")
print("*" * 50)


