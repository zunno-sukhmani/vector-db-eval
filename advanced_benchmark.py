import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.neighbors import NearestNeighbors
import time, os, csv, psutil, matplotlib.pyplot as plt

# Settings
dataset_sizes = [1000, 5000, 10000]
dimensions = [64, 128, 256]
num_queries = 100
k = 5  # number of neighbors

# Prepare results storage
results = []

# Helper for memory usage
def get_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

# Main benchmarking loop
for d in dimensions:
    for n in dataset_sizes:
        print(f"\n--- Benchmarking: {n} vectors, {d} dimensions ---")
        X, _ = make_blobs(n_samples=n, n_features=d, centers=5, random_state=42)
        X = X.astype(np.float32)
        query_vectors = X[:num_queries]

        # 1. NumPy Baseline
        print("NumPy baseline...")
        mem_before = get_memory_mb()
        start = time.time()
        stored_vectors = X.copy()
        build_time = time.time() - start
        mem_after = get_memory_mb()
        mem_used = mem_after - mem_before

        start = time.time()
        numpy_neighbors = []
        for qv in query_vectors:
            dists = np.linalg.norm(stored_vectors - qv, axis=1)
            nearest = np.argsort(dists)[:k]
            numpy_neighbors.append(nearest)
        search_time = (time.time() - start) / num_queries

        results.append(["NumPy", n, d, build_time, search_time, mem_used, 1.0, "Brute-force, baseline"])

        # 2. scikit-learn NearestNeighbors
        print("scikit-learn NearestNeighbors...")
        mem_before = get_memory_mb()
        start = time.time()
        nn = NearestNeighbors(n_neighbors=k, algorithm='brute', metric='euclidean')
        nn.fit(X)
        build_time = time.time() - start
        mem_after = get_memory_mb()
        mem_used = mem_after - mem_before

        start = time.time()
        recall_total = 0
        for i, qv in enumerate(query_vectors):
            distances, indices = nn.kneighbors([qv])
            recall = len(set(indices[0]) & set(numpy_neighbors[i])) / k
            recall_total += recall
        search_time = (time.time() - start) / num_queries
        avg_recall = recall_total / num_queries

        results.append(["sklearn-NN", n, d, build_time, search_time, mem_used, avg_recall, "Brute-force, pure Python"])

        # 3. FAISS
        try:
            import faiss
            print("FAISS...")
            mem_before = get_memory_mb()
            start = time.time()
            index = faiss.IndexFlatL2(d)
            index.add(X)
            build_time = time.time() - start
            mem_after = get_memory_mb()
            mem_used = mem_after - mem_before

            start = time.time()
            recall_total = 0
            for i, qv in enumerate(query_vectors):
                D, I = index.search(np.expand_dims(qv, 0), k)
                recall = len(set(I[0]) & set(numpy_neighbors[i])) / k
                recall_total += recall
            search_time = (time.time() - start) / num_queries
            avg_recall = recall_total / num_queries

            results.append(["FAISS", n, d, build_time, search_time, mem_used, avg_recall, "In-memory, no metadata"])
        except Exception as e:
            results.append(["FAISS", n, d, "ERROR", "ERROR", "ERROR", "ERROR", str(e)])

        # 4. Chroma
        try:
            import chromadb
            from chromadb.config import Settings
            print("Chroma...")
            mem_before = get_memory_mb()
            client = chromadb.Client(Settings())
            collection = client.create_collection("test")
            start = time.time()
            for i, vec in enumerate(X):
                collection.add(embeddings=[vec.tolist()], ids=[str(i)])
            build_time = time.time() - start
            mem_after = get_memory_mb()
            mem_used = mem_after - mem_before

            start = time.time()
            recall_total = 0
            for i, qv in enumerate(query_vectors):
                res = collection.query(query_embeddings=[qv.tolist()], n_results=k)
                ids = [int(x) for x in res['ids'][0]]
                recall = len(set(ids) & set(numpy_neighbors[i])) / k
                recall_total += recall
            search_time = (time.time() - start) / num_queries
            avg_recall = recall_total / num_queries

            results.append(["Chroma", n, d, build_time, search_time, mem_used, avg_recall, "Simple API, persistent"])
        except Exception as e:
            results.append(["Chroma", n, d, "ERROR", "ERROR", "ERROR", "ERROR", str(e)])

        # 5. Qdrant (in-memory)
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import VectorParams, Distance
            print("Qdrant...")
            mem_before = get_memory_mb()
            client = QdrantClient(":memory:")
            client.recreate_collection(
                collection_name="test",
                vectors_config=VectorParams(size=d, distance=Distance.COSINE)
            )
            start = time.time()
            client.upload_collection(
                collection_name="test",
                vectors=X.tolist(),
                ids=list(range(len(X)))
            )
            build_time = time.time() - start
            mem_after = get_memory_mb()
            mem_used = mem_after - mem_before

            start = time.time()
            recall_total = 0
            for i, qv in enumerate(query_vectors):
                result = client.search(
                    collection_name="test",
                    query_vector=qv.tolist(),
                    limit=k
                )
                ids = [point.id for point in result]
                recall = len(set(ids) & set(numpy_neighbors[i])) / k
                recall_total += recall
            search_time = (time.time() - start) / num_queries
            avg_recall = recall_total / num_queries

            results.append(["Qdrant", n, d, build_time, search_time, mem_used, avg_recall, "Filtering, metadata, in-memory"])
        except Exception as e:
            results.append(["Qdrant", n, d, "ERROR", "ERROR", "ERROR", "ERROR", str(e)])

# Export results to CSV
os.makedirs("results", exist_ok=True)
csv_path = "results/benchmark_results.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["DB", "Dataset Size", "Dimensions", "Build Time (s)", "Search Time (s)", "Memory Used (MB)", "Recall", "Notes"])
    writer.writerows(results)
print(f"\nResults saved to {csv_path}")

# Plotting (example: search time vs. dataset size for each DB at d=128)
import matplotlib.pyplot as plt
import pandas as pd

df = pd.DataFrame(results, columns=["DB", "Dataset Size", "Dimensions", "Build Time (s)", "Search Time (s)", "Memory Used (MB)", "Recall", "Notes"])
df = df[df["Dimensions"] == 128]
plt.figure(figsize=(10,6))
for db in df["DB"].unique():
    sub = df[df["DB"] == db]
    plt.plot(sub["Dataset Size"], sub["Search Time (s)"], marker='o', label=db)
plt.xlabel("Dataset Size")
plt.ylabel("Avg Search Time (s)")
plt.title("Search Time vs. Dataset Size (128 dimensions)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/search_time_vs_size.png")
plt.show()

df = pd.DataFrame(results, columns=["DB", "Dataset Size", "Dimensions", "Build Time (s)", "Search Time (s)", "Memory Used (MB)", "Recall", "Notes"])
# Filter for dimension 128
plot_dim = 128
df = df[df["Dimensions"] == plot_dim]
# Remove rows with 'ERROR' in any metric columns
for col in ["Build Time (s)", "Search Time (s)", "Memory Used (MB)", "Recall"]:
    df = df[df[col] != "ERROR"]
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Plot Search Time
plt.figure(figsize=(10,6))
for db in df["DB"].unique():
    sub = df[df["DB"] == db]
    plt.plot(sub["Dataset Size"], sub["Search Time (s)"], marker='o', label=db)
plt.xlabel("Dataset Size")
plt.ylabel("Avg Search Time (s)")
plt.title(f"Search Time vs. Dataset Size ({plot_dim} dimensions)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/search_time_vs_size.png")
plt.show()

# Plot Recall
plt.figure(figsize=(10,6))
for db in df["DB"].unique():
    sub = df[df["DB"] == db]
    plt.plot(sub["Dataset Size"], sub["Recall"], marker='o', label=db)
plt.xlabel("Dataset Size")
plt.ylabel("Recall")
plt.title(f"Recall vs. Dataset Size ({plot_dim} dimensions)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/recall_vs_size.png")
plt.show()

# Plot Memory Used
plt.figure(figsize=(10,6))
for db in df["DB"].unique():
    sub = df[df["DB"] == db]
    plt.plot(sub["Dataset Size"], sub["Memory Used (MB)"], marker='o', label=db)
plt.xlabel("Dataset Size")
plt.ylabel("Memory Used (MB)")
plt.title(f"Memory Usage vs. Dataset Size ({plot_dim} dimensions)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/memory_vs_size.png")
plt.show()

# Plot Build Time
plt.figure(figsize=(10,6))
for db in df["DB"].unique():
    sub = df[df["DB"] == db]
    plt.plot(sub["Dataset Size"], sub["Build Time (s)"], marker='o', label=db)
plt.xlabel("Dataset Size")
plt.ylabel("Build Time (s)")
plt.title(f"Build Time vs. Dataset Size ({plot_dim} dimensions)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/build_time_vs_size.png")
plt.show() 