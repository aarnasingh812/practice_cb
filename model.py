import numpy as np
from embeddings import generate_embeddings


def semantic_search(query, index, k=2):
    
    query_embedding = generate_embeddings(query)
    query_embedding = np.array(query_embedding).reshape(1, -1)
    D, I = index.search(np.array(query_embedding).astype('float32'), k)  # D: distances, I: indices
    
    return D, I


def retrieve_relevant_chunks(indices, chunks):
    relevant_chunks = [chunks[i] for i in indices[0]]
    return relevant_chunks


def perform_semantic_search(query, index, chunks, k=2):
    
    D, I = semantic_search(query, index, k)
    
    relevant_chunks = retrieve_relevant_chunks(I, chunks)
    
    return relevant_chunks, D