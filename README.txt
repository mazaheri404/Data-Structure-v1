QDCEL-Benchmark

QDCEL-Benchmark is a MATLAB framework for benchmarking geometric data structures in Field-of-View (FoV) queries and spatial computations. It evaluates the proposed QDCEL (Quadtree + DCEL) data structure against several standard baselines, including KDTree, RTree, and Uniform Quadtree, across different scene densities.


⚙️ Features

Automatic generation of synthetic scenes with varying densities (sparse → dense).

Benchmarks 5 indexing methods:

QDCEL-Leaf

QDCEL-Clip

Uniform Quadtree (Leaf)

KDTree

RTree

Measures four key metrics:

Build Time (index construction time)

Memory Usage (peak RAM)

Query Latency (response time for FoV queries)

Accuracy (IoU against ground truth geometry)

Exports ISI-style vector plots suitable for publications.

🚀 Usage

Copy the repository or the file qdcel_benchmark_with_DCEL_RTree.m.

Run the main script in MATLAB:

qdcel_benchmark_with_DCEL_RTree


The script will automatically:

Generate triangulated scenes.

Build indexes for all methods.

Sample random FoVs and run queries.

Record metrics (BuildT, MemMB, QryT, IoU).

Outputs include:

qdcel_bench_plots_v2.pdf → Comparative plots of all metrics.

qdcel_bench_results_v2.mat → Raw results for further analysis.

📂 Output Files

<name>_bench_plots.pdf → Benchmark plots (Build Time, Memory, Query Latency, Accuracy).

<name>_bench_results.mat → Raw numerical results for MATLAB analysis.

👤 Contact

Hakimeh mazaheri
University of kashan
Hakimeh.mazaheri@gmail.com

📝 License

Copyright (C) 2025 

