# G-DBSCAN

Based on: [G-DBSCAN: A GPU Accelerated Algorithm for Density-based
Clustering](https://www.sciencedirect.com/science/article/pii/S1877050913003438)

## Build instructions:
0. For the first time: `git submodule update --init --recursive`
1. `cmake -Bbuild -H.`
2. `cmake --build build --target example`

## How to run:
0. Generate an example using the Python script:
`python3 generate_dateset.py --n-samples=20000 --cluster-std=3.0`
  - 20,000 points with 3.0 stddiv;

1. Cluster the input and visualize it:
`python3 dbscan.py --input-name=test_input.txt --eps=0.1 --min-samples=12`
  - 0.1 radius and 12 neighbour points for clustering.

2. `./build/main --input=$(pwd)/test_input.txt --eps=0.1 --min-samples=12`.
