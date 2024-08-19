import os
import tempfile
import uuid
from datetime import datetime

from dotenv import load_dotenv

import azure_utils
import model_utils
import processing_utils
import logging_utils

load_dotenv(dotenv_path='photo_assistant.env')
logger = logging_utils.setup_custom_logger('photo-assistant')

LOCAL_PHOTO_FOLDER = "/home/elena/PycharmProjects/photoAssistant/source_photos"
RESIZE_FACTOR = 0.5


def get_photo_creation_date(file_path):
    try:
        return datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat() + 'Z'
    except Exception as e:
        print(f"Error reading creation date for {file_path}: {e}")
        return None


def upload_file_to_blob(blob_container_client, file_path, photo_id):
    blob_client = blob_container_client.get_blob_client(f"{photo_id}.jpg")
    with open(file_path, "rb") as data:
        blob_client.upload_blob(data, overwrite=True)
    return f"https://{blob_container_client.account_name}.blob.core.windows.net/{blob_container_client.container_name}/{photo_id}.jpg"


def insert_metadata_to_cosmos(cosmos_container_client, photo_id, url, filename, tags, upload_date, create_date,
                              embedding):
    metadata = {
        "id": photo_id,
        "filename": filename,
        "url": url,
        "tags": tags,
        "uploadDate": upload_date,
        "create_date": create_date,
        "embedding": embedding
    }
    cosmos_container_client.create_item(body=metadata)


def main():
    # Configure Azure Blob Storage.
    blob_container_client = azure_utils.get_blob_container_client()

    # Configure Azure Cosmos DB.
    cosmos_container_client = azure_utils.get_cosmos_container_client()

    # Load the CLIP model and processor.
    clip_model = model_utils.get_clip_model()
    clip_processor = model_utils.get_clip_processor()

    for root, _, files in os.walk(LOCAL_PHOTO_FOLDER):
        for file in files:
            file_path = os.path.join(root, file)
            upload_date = datetime.utcnow().isoformat() + 'Z'
            create_date = get_photo_creation_date(file_path)
            photo_id = str(uuid.uuid4())  # Generate a unique UUID for the photo.
            url = upload_file_to_blob(blob_container_client, file_path, photo_id)

            # Generate embeddings.
            logger.info(f"Generating embedding for file {file}...")
            try:
                # Resize image and create image embedding.
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                    resized_file_path = temp_file.name
                    processing_utils.resize_image(file_path, resized_file_path, RESIZE_FACTOR)
                    if os.path.exists(resized_file_path):
                        embedding = model_utils.generate_clip_embedding_from_image(resized_file_path, clip_model,
                                                                                   clip_processor)
                        logger.info(f"Embedding dimension: {len(embedding)}.")
                    else:
                        raise FileNotFoundError(f"Resized file {resized_file_path} not found.")
                # Ensure the temporary file is deleted after the embedding is computed.
                os.remove(resized_file_path)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue

            tags = ["iphone"]  # Example tags.
            insert_metadata_to_cosmos(cosmos_container_client, photo_id, url, file, tags, upload_date, create_date,
                                      embedding)
            logger.info(f"Uploaded {file} and inserted metadata with ID {photo_id}.")


if __name__ == "__main__":
    main()
