# G-DBSCAN

Based on: [G-DBSCAN: A GPU Accelerated Algorithm for Density-based
Clustering](https://www.sciencedirect.com/science/article/pii/S1877050913003438)

## Build instructions:
0. For the first time: `git submodule update --init --recursive`
1. `cmake -Bbuild -H.`
2. `cmake --build build --target example`

## How to run:
0. Generate an example using the Python script, and visualize the
clustering: `python3 generate_input_and_cluster.py --visualize --n-samples 10000 --cluster-std 3.0 --eps 0.1 --min_samples 12`
  - 10,000 points with 3.0 stddiv;
  - 0.1 radius and 12 neighbour points for clustering.

1. Once satisfied with the configuration, generate the input:
`python3 generate_input_and_cluster.py --generate --n-samples 10000 --cluster-std 3.0`
  - The default test input name is `test.out`.

2. `./build/example test.out 0.1 12`.
