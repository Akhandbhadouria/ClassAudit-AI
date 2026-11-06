import numpy as np

def match_face(known_embedding, live_embedding, threshold=0.45):
    """
    Matches two normalized face embeddings using Cosine Distance.
    
    Cosine distance = 1 - dot_product(u, v) (for unit-normalized vectors).
    Threshold <= 0.45 is standard for ArcFace 512-D cosine distance.
    
    Args:
        known_embedding: 1D numpy array
        live_embedding: 1D numpy array
        threshold: float cutoff (default 0.45)
        
    Returns:
        bool: True if faces match, False otherwise.
    """
    if known_embedding is None or live_embedding is None:
        return False

    try:
        u = np.asarray(known_embedding, dtype=np.float32)
        v = np.asarray(live_embedding, dtype=np.float32)

        norm_u = np.linalg.norm(u)
        norm_v = np.linalg.norm(v)

        if norm_u > 0:
            u = u / norm_u
        if norm_v > 0:
            v = v / norm_v

        # Cosine similarity and distance
        similarity = float(np.dot(u, v))
        distance = 1.0 - similarity

        return distance < threshold
    except Exception:
        return False
