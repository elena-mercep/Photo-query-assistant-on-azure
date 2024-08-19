# Photo Query Assistant on Azure

The notebook `photo_assistant.ipynb` describes solution to implement a sample solution to query photos stored on Azure.

You'll learn how to create and manage Azure resources such as Azure Storage and Cosmos DB, upload photos and metadata using Python scripts, retrieve photos based on text queries using embeddings, and deploy an Azure Function to query photos via HTTP requests.

Repository structure:

*  `.py` files contain relevant scripts for upload or retrieve photos.
*  `_utils.py` files contain extendable helper methods.
*  `.env` file contains relevant environment variables.

## Prerequisites

1. **Azure Subscription**: Obtain an Azure subscription to create and manage cloud services.
2. **Install Azure CLI**: Install Azure CLI to manage Azure resources from the command line.
3. **Azure Storage Account**: Set up an Azure Storage account to store photos and metadata securely.
4. **Python Environment**: Install Python, create a new Conda environment, and install necessary dependencies for Azure CLI, Azure Cosmos, Azure Blob Storage, and additional required packages.
5. **PyCharm or Similar IDE**: Set up an IDE like PyCharm or other IDE of choice for Python development.
