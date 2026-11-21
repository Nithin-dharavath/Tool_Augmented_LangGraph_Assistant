from langgraph.store.memory import InMemoryStore
from langchain_huggingface import HuggingFaceEmbeddings

from dotenv import load_dotenv

load_dotenv()

embeddings = HuggingFaceEmbeddings(
        model = "sentence-transformers/all-MiniLM-L6-v2"
    )

store = InMemoryStore(index={"embed" : embeddings, "dims" : 1200})
space = ("user", "uu1")

store.put(space, "11", {"data" : "user like burger"})
store.put(space, "22", {"data" : "user like Tech"})
store.put(space, "12", {"data": "user like fried rice"})
store.put(space, "12", {"data": "user rice is uncooked"})
store.put(space, "12", {"data": "user like langgraph best"})


items = store.search(space, query="what does user like?", limit=4)
for i in items:
    print("all the data",  i.value)