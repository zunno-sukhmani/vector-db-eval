import numpy as np
from sklearn.datasets import make_blobs
import time

# Generate synthetic data
X, _ = make_blobs(n_samples=1000, n_features=128, centers=5, random_state=42)
X = X.astype(np.float32)

results = []

# 1. FAISS
try:
    import faiss
    d = X.shape[1]
    index = faiss.IndexFlatL2(d)
    start = time.time()
    index.add(X)
    insert_time = time.time() - start

    start = time.time()
    D, I = index.search(X[:5], k=5)
    search_time = time.time() - start

    results.append(("FAISS", insert_time, search_time, "In-memory, no metadata"))
except Exception as e:
    results.append(("FAISS", "ERROR", "ERROR", str(e)))

# 2. Chroma
try:
    import chromadb
    from chromadb.config import Settings
    client = chromadb.Client(Settings())
    collection = client.create_collection("test")
    start = time.time()
    for i, vec in enumerate(X):
        collection.add(embeddings=[vec.tolist()], ids=[str(i)])
    insert_time = time.time() - start

    start = time.time()
    res = collection.query(query_embeddings=[X[0].tolist()], n_results=5)
    search_time = time.time() - start

    results.append(("Chroma", insert_time, search_time, "Simple API, persistent"))
except Exception as e:
    results.append(("Chroma", "ERROR", "ERROR", str(e)))

# 3. Qdrant (in-memory)
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import VectorParams, Distance
    client = QdrantClient(":memory:")
    client.recreate_collection(
        collection_name="test",
        vectors_config=VectorParams(size=128, distance=Distance.COSINE)
    )
    start = time.time()
    client.upload_collection(
        collection_name="test",
        vectors=X.tolist(),
        ids=list(range(len(X)))
    )
    insert_time = time.time() - start

    start = time.time()
    result = client.search(
        collection_name="test",
        query_vector=X[0].tolist(),
        limit=5
    )
    search_time = time.time() - start

    results.append(("Qdrant", insert_time, search_time, "Filtering, metadata, in-memory"))
except Exception as e:
    results.append(("Qdrant", "ERROR", "ERROR", str(e)))

# 4. Scikit-learn baseline
try:
    from sklearn.neighbors import NearestNeighbors
    start = time.time()
    nn = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean')
    nn.fit(X)
    insert_time = time.time() - start

    start = time.time()
    distances, indices = nn.kneighbors([X[0]])
    search_time = time.time() - start

    results.append(("sklearn-NN", insert_time, search_time, "Brute-force, pure Python, no index"))
except Exception as e:
    results.append(("sklearn-NN", "ERROR", "ERROR", str(e)))
# 5. NumPy Baseline (Brute-force search)
try:
    start = time.time()
    # "Insert" is just storing the array, so it's instant
    stored_vectors = X.copy()
    insert_time = time.time() - start

    start = time.time()
    # Brute-force L2 distance to all vectors for the first query vector
    dists = np.linalg.norm(stored_vectors - X[0], axis=1)
    nearest = np.argsort(dists)[:5]
    search_time = time.time() - start

    results.append(("NumPy", insert_time, search_time, "Brute-force, baseline, no index"))
except Exception as e:
    results.append(("NumPy", "ERROR", "ERROR", str(e)))

# Print results
print("\nBenchmark Results:")
print(f"{'DB':<10} {'Insert Time (s)':<15} {'Search Time (s)':<15} {'Notes'}")
for db, ins, sea, note in results:
    print(f"{db:<10} {ins!s:<15} {sea!s:<15} {note}")