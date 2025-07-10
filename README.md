**Vector Database Benchmarking Project**

Project Overview

This project benchmarks and compares several popular vector databases and libraries for similarity search on synthetic datasets of varying sizes and dimensions. The goal is to evaluate their performance, accuracy, and resource usage to inform future technology choices for vector search workloads.

Databases and Methods Evaluated

NumPy: Brute-force baseline using pure NumPy (no index).

scikit-learn NearestNeighbors: Brute-force search using sklearn.

FAISS: Facebook AI Similarity Search, in-memory, L2 index.

Chroma: Simple persistent vector DB with Python API.

Qdrant: In-memory vector DB with filtering and metadata support.

Benchmarking Methodology

Synthetic datasets are generated using sklearn's make\_blobs, with sizes of 1,000, 5,000, and 10,000 vectors, and dimensions of 64, 128, and 256.

For each configuration, 100 random queries are run and average search time is measured.

Metrics collected: build/insert time, average search time, memory usage, and recall (accuracy vs. NumPy baseline).

Recall is calculated as the fraction of true nearest neighbors (from brute-force) found by each method.

Results are exported to CSV and visualized as plots.

Metrics

Build/Insert Time: Time taken to build the index or insert all vectors.

Search Time: Average time to perform a nearest neighbor search.

Memory Usage: Memory used by the index after building.

Recall: Accuracy of the search results compared to the brute-force baseline.

How to Run

1.Install dependencies:

pip install numpy scikit-learn chromadb qdrant-client faiss-cpu matplotlib psutil pandas fpdf

2.Run the benchmark:

python scripts/advanced\_benchmark.py

3.Generate a PDF report:

python scripts/generate\_full\_report.py

The PDF report will be saved as results/benchmark\_full\_report.pdf.

Result Interpretation

FAISS and NumPy are extremely fast for small to medium datasets.

scikit-learn's brute-force is slower for search, especially as dataset size increases.

Chroma and Qdrant offer persistent storage and filtering, with good performance.

Memory usage and build time scale with dataset size and dimensionality.

For large-scale or production workloads, FAISS or Qdrant are recommended.

For simple, small-scale tasks, NumPy or scikit-learn may suffice.

Plots and Reports

Plots for search time, recall, memory usage, and build time vs. dataset size are saved in the results folder.

The PDF report includes project overview, methodology, sample results, plots, key findings, and an appendix with the full results table.

Customization

You can adjust dataset sizes, dimensions, and the number of queries in scripts/advanced\_benchmark.py.

Add or remove databases as needed by editing the script.

Update the PDF report template in scripts/generate\_full\_report.py for your organization's needs.

For any questions or further customization, please contact the project maintainer.


# Initial Commit by Vanshika

\# Prompt Engineering Fundamentals 🚀



This project contains five core NLP prompt designs and comparisons using OpenAI (GPT-4) and DeepSeek models.



\## 📌 Tasks Covered

1\. Text Classification

2\. Summarization

3\. Few-Shot Reasoning

4\. Tool-Calling with JSON Output

5\. Chain-of-Thought Reasoning



\## 📁 Folder Structure

\- `prompts/`: Raw prompts

\- `comparisons/`: Model outputs

\- `results/`: Evaluation and metrics



\## ⚙️ Goal

Compare LLM behavior across tasks to understand prompt engineering impact.



\## 🔗 Model Used

\- OpenAI GPT-4 (via ChatGPT)

\- DeepSeek Chat (via web or API)



\## 📊 Summary

Evaluation includes accuracy, structure, and response logic for each task.


