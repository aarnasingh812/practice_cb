from sentence_transformers import SentenceTransformer
import faiss
import numpy as np



def chunk_text_with_overlap(sentences, chunk_size=3, overlap_size=1):
    chunks = []
    current_chunk = []
    
    for i in range(len(sentences)):
        current_chunk.append(sentences[i])
        
        if len(current_chunk) == chunk_size:
            chunks.append(" ".join(current_chunk)) 
            
            current_chunk = current_chunk[-overlap_size:]  
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks


def generate_embeddings(sentences):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(sentences)
    return embeddings



def save_to_faiss(embeddings, index_path="index.faiss"):
    embeddings = np.array(embeddings).astype('float32')
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, index_path)



def load_faiss_index(index_path="index.faiss"):
    
    index = faiss.read_index(index_path)  
    return index

