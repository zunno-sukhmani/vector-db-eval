import pandas as pd
from fpdf import FPDF
import os

# Paths
csv_path = "results/benchmark_results.csv"
plot_files = [
    "results/search_time_vs_size.png",
    "results/recall_vs_size.png",
    "results/memory_vs_size.png",
    "results/build_time_vs_size.png"
]
pdf_path = "results/benchmark_report.pdf"

# Load results
df = pd.read_csv(csv_path)

# Create PDF
pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()

# Title
pdf.set_font("Arial", "B", 16)
pdf.cell(0, 10, "Vector Database Benchmark Report", ln=True, align="C")
pdf.ln(10)

# Table summary (first 10 rows)
pdf.set_font("Arial", "B", 12)
pdf.cell(0, 10, "Results Table (first 10 rows):", ln=True)
pdf.set_font("Arial", "", 10)
table_cols = ["DB", "Dataset Size", "Dimensions", "Build Time (s)", "Search Time (s)", "Memory Used (MB)", "Recall"]
for i, row in df.head(10).iterrows():
    row_str = " | ".join(str(row[col]) for col in table_cols)
    pdf.cell(0, 8, row_str, ln=True)
pdf.ln(5)

# Add plots
for plot in plot_files:
    if os.path.exists(plot):
        pdf.add_page()
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, os.path.basename(plot), ln=True)
        pdf.image(plot, w=180)
        pdf.ln(5)

# Save PDF
pdf.output(pdf_path)
print(f"PDF report saved to {pdf_path}")