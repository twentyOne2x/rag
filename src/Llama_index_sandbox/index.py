from llama_index.legacy import VectorStoreIndex
from llama_index.legacy.vector_stores import PineconeVectorStore
import logging
import os
import pinecone

from src.Llama_index_sandbox.custom_react_agent.tools.reranker.custom_vector_store_index import CustomVectorStoreIndex
from src.Llama_index_sandbox.utils.utils import timeit, load_vector_store_from_pinecone_database, load_vector_store_from_pinecone_database_legacy


@timeit
def initialise_vector_store(embedding_model_vector_dimension, vector_space_distance_metric='cosine') -> PineconeVectorStore:
    api_key = os.environ["PINECONE_API_KEY"]
    pinecone.init(api_key=api_key, environment=os.environ["PINECONE_API_ENVIRONMENT"])
    index_name = "icmfyi"

    # Check if the index already exists
    existing_indexes = pinecone.list_indexes()
    if index_name in existing_indexes:
        # If the index exists, delete it
        pinecone.delete_index(index_name)

    # NOTE: We do not index the metadata fields by video/paper link.
    #  https://docs.pinecone.io/docs/manage-indexes#selective-metadata-indexing
    #  https://docs.pinecone.io/docs/metadata-filtering
    # High cardinality consumes more memory: Pinecone indexes metadata to allow
    # for filtering. If the metadata contains many unique values — such as a unique
    # identifier for each vector — the index will consume significantly more
    # memory. Consider using selective metadata indexing to avoid indexing
    # high-cardinality metadata that is not needed for filtering.
    metadata_config = {
        "indexed": ["document_type", "title", "authors", "release_date"]
    }
    pinecone.create_index(name=index_name,
                          metadata_config=metadata_config,
                          dimension=embedding_model_vector_dimension,
                          metric=vector_space_distance_metric,
                          pod_type="s1.x1")
    pinecone_index = pinecone.Index(index_name=index_name)
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

    # Optionally, you might want to delete all contents in the index
    # pinecone_index.delete(deleteAll=True)
    return vector_store

@timeit
def load_nodes_into_vector_store_create_index(nodes, embedding_model_vector_dimension, vector_space_distance_metric) -> VectorStoreIndex:
    """
    We now insert these nodes into our PineconeVectorStore.

    NOTE: We skip the VectorStoreIndex abstraction, which is a higher-level
    abstraction that handles ingestion as well. We use VectorStoreIndex in the next section to fast-track retrieval/querying.
    """
    vector_store = initialise_vector_store(embedding_model_vector_dimension=embedding_model_vector_dimension, vector_space_distance_metric=vector_space_distance_metric)
    vector_store.add(nodes)
    index = VectorStoreIndex.from_vector_store(vector_store)
    return index


@timeit
def load_index_from_disk(service_context) -> CustomVectorStoreIndex:
    # load the latest directory in index_dir
    try:
        vector_store = load_vector_store_from_pinecone_database_legacy()
        index = CustomVectorStoreIndex.from_vector_store(vector_store, service_context)
        logging.info(f"Successfully loaded index from disk.")
        return index
    except Exception as e:
        logging.error(f"Error: {e}")
        # To accommodate for the case where the vector_store.json file is not persisted https://stackoverflow.com/questions/76837143/llamaindex-index-storage-context-persist-not-storing-vector-store
        if "No existing llama_index.vector_stores.simple" in str(e):
            # create a vector_store.json file with {} inside
            try:
                vector_store = load_vector_store_from_pinecone_database()
                index = VectorStoreIndex.from_vector_store(vector_store, service_context)
                return index
            except Exception as e:
                logging.error(f"load_index_from_disk ERROR: {e}")
                exit(1)
