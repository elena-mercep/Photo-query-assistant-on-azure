import numpy as np
from dotenv import load_dotenv

import azure_utils
import model_utils
import logging_utils
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv(dotenv_path='photo_assistant.env')
logger = logging_utils.setup_custom_logger('photo-assistant')


def find_closest_match_by_vector_search(cosmos_container_client, query_embedding):
    # Define the query parameters.
    query = ('SELECT TOP 1 c.id, VectorDistance(c.embedding, @query_embedding) AS SimilarityScore FROM c ORDER BY '
             'VectorDistance(c.embedding,@query_embedding)')

    # Prepare the query parameters.
    parameters = [
        {'name': '@query_embedding', 'value': query_embedding.tolist()}
    ]

    # Execute the query.
    items = list(cosmos_container_client.query_items(
        query=query,
        parameters=parameters,
        enable_cross_partition_query=True
    ))

    # Return the closest photo ID.
    if items:
        closest_photo_id = items[0]['id']
        return closest_photo_id
    else:
        return None, -1  # Return default values if no items found


# Fetch all embeddings from Cosmos DB.
def fetch_all_embeddings(cosmos_container_client):
    query = "SELECT c.id, c.embedding FROM c"
    items = list(cosmos_container_client.query_items(query=query, enable_cross_partition_query=True))
    return items


# Find the closest match to the query embedding.
def find_closest_match(cosmos_container_client, query_embedding):
    embeddings = fetch_all_embeddings(cosmos_container_client)
    max_similarity = -1
    closest_photo_id = None

    for item in embeddings:
        photo_id = item['id']
        photo_embedding = np.array(item['embedding'])
        similarity = cosine_similarity([query_embedding], [photo_embedding])[0][0]
        if similarity > max_similarity:
            max_similarity = similarity
            closest_photo_id = photo_id

    return closest_photo_id, max_similarity


def main():
    # Configure Azure Blob Storage.
    blob_container_client = azure_utils.get_blob_container_client()

    # Configure Azure Cosmos DB.
    cosmos_container_client = azure_utils.get_cosmos_container_client()

    # Load the CLIP model and processor.
    clip_model = model_utils.get_clip_model()
    clip_processor = model_utils.get_clip_processor()

    # Generate the embedding for the query text.
    query_text = "photo with city"
    query_embedding = model_utils.generate_clip_embedding_from_text(query_text, clip_model, clip_processor)

    # Find the closest match.
    logger.info("Searching for the closest match between text query and photo embeddings...")
    closest_photo_id = find_closest_match_by_vector_search(cosmos_container_client, query_embedding)
    print(f"Closest photo ID: {closest_photo_id} for the query: {query_text}")


if __name__ == "__main__":
    main()
