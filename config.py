import os
from pymongo import MongoClient
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_mongodb.index import create_fulltext_search_index
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
if not MONGODB_URI:
    raise ValueError("MONGODB_URI environment variable is required")

embedding_model = OpenAIEmbeddings(model="text-embeddings-small")
llm = ChatOpenAI(model="gpt-5.1")

mongo_client = MongoClient(MONGODB_URI)
collection = mongo_client["sample_mflix"]["embedded_movies"]

vector_store = MongoDBAtlasVectorSearch.from_connection_string(
    connection_string=MONGODB_URI,
    namespace="sample_mflix.embedded_movies",
    embedding=embedding_model,
    text_key="plot",
    # embedding_key="plot_embedding_voyage_3_large",
    relevance_score_fn="dotProduct",
)

print("Setting up vector store and indexes...")
existing_indexes = []
try:
    existing_indexes = list(collection.list_search_indexes())
    vector_index_exists = any(
        idx.get("name") == "vector_index" for idx in existing_indexes
    )
    if vector_index_exists:
        print("Vector search index already exists, skipping creation...")
    else:
        print("Creating vector search index...")
        vector_store.create_vector_search_index(
            dimensions=2048,
            wait_until_complete=60,
        )
        print("Vector search index created successfully!")
except Exception as e:
    print(f"Error creating vector search index: {e}")
try:
    fulltext_index_exists = any(
        idx.get("name") == "search_index" for idx in existing_indexes
    )

    if fulltext_index_exists:
        print("Search index already exists, skipping creation...")
    else:
        print("Creating search index...")
        create_fulltext_search_index(
            collection=collection,
            field="title",
            index_name="search_index",
            wait_until_complete=60,
        )
        print("Search index created successfully!")
except Exception as e:
    print(f"Error creating search index: {e}")
