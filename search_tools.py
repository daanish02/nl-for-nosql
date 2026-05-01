from bson import ObjectId
from langchain_core.tools import tool
from langchain_mongodb.retrievers.full_text_search import (
    MongoDBAtlasFullTextSearchRetriever,
)
from config import vector_store, collection


@tool
def plot_search(user_query: str) -> str:
    """
    Retrieve information on the movie's plot to answer a user query by using vector search.
    """

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5},
    )

    results = retriever.invoke(user_query)

    context = "\n\n".join(
        [
            f"[Movie ID: {doc.metadata.get('_id', 'Unknown')}] {doc.metadata.get('title', 'Unknown Title')}: {doc.page_content}"
            for doc in results
        ]
    )
    return context


@tool
def title_search(user_query: str) -> str:
    """
    Retrieve movie plot content based on the provided title by using full-text search.
    """

    retriever = MongoDBAtlasFullTextSearchRetriever(
        collection=collection,
        search_field="title",
        search_index_name="search_index",
        top_k=3,
    )
    results = retriever.invoke(user_query)

    for doc in results:
        if doc:
            movie_id = doc.metadata.get("_id", "Unknown")
            return f"[Movie ID: {movie_id}]\nPlot: {doc.metadata.get('fullplot', 'No additional plot details available.')}"
        else:
            return "Movie not found"

    return "Movie not found"


@tool
def get_movie_details(movie_id: str) -> str:
    """
    Retrieve all available detailed information (cast, directors, genres, ratings, etc.) for a specific movie using its exact ID. \nUse this tool when you need more details about a movie after finding its ID from title_search or plot_search.
    """
    try:
        if len(movie_id) == 24:
            query_id = ObjectId(movie_id)
        else:
            query_id = movie_id

        movie = collection.find_one({"_id": query_id})

        if movie:
            movie["_id"] = str(movie["_id"])
            return str(movie)
        return "Movie details not found for the given exact ID."
    except Exception as e:
        return f"Error retrieving movie details: {str(e)}"


SEARCH_TOOLS = [plot_search, title_search, get_movie_details]
