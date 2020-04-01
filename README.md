# DBSCAN

The GPU implementation is inspired by 
[this](https://www.sciencedirect.com/science/article/pii/S1877050913003438) 
paper.

## Requirements
C++14; CUDA; Thrust.

## Build instructions
0. For the first time: `git submodule update --init --recursive`
1. `cmake -Bbuild -H.`
2. `cmake --build build --target <target>`

## How to run

0. [optional] Generate an example using the Python script:
`python3 generate_dateset.py --n-samples=20000 --cluster-std=3.0`
    - 20,000 points with 3.0 stddiv;

1. [optional] Cluster the input and visualize it using Sklearn:
`python3 dbscan.py --input-name=test_input.txt --eps=0.1 --min-samples=12`
    - 0.1 radius and 12 neighbour points for clustering.

### CPU algorithm
2. `./build/bin/cpu-main --input=test_input.txt --eps=0.1 --min-samples=12`.
    - Append `--print` to see the cluster ids.
    - Append `--num-threads=K` to speed up the processing.

### GPU algorithm
2. `./build/bin/gpu-main --input=test_input.txt --eps=0.1 --min-samples=12`.
    - Append `--print` to see the cluster ids.

## Test data
test_input_20k.txt is generated using `--cluster-std=2.2 --n-samples=20000`,
with 4 clusters. Should use `--eps=0.15 --min-samples=180` to query.

test_input_50k.txt is generated using `--cluster-std=2.2 --n-samples=50000`,
with 20 clusters. Should use `--eps=0.09 --min-samples=110` to query.

test_input_100k.txt is generated using `--cluster-std=2.2 --n-samples=100000`,
with 20 clusters. Should use `--eps=0.07 --min-samples=100` to query.

test_input_200k.txt is generated using `--cluster-std=2.2 --n-samples=200000`,
with 20 clusters. Should use `--eps=0.05 --min-samples=100` to query.

test_input_300k.txt is generated using `--cluster-std=2.2 --n-samples=300000`,
with 20 clusters. Should use `--eps=0.044 --min-samples=100` to query.

test_input_400k.txt is generated using `--cluster-std=2.2 --n-samples=400000`,
with 20 clusters. Should use `--eps=0.04 --min-samples=105` to query.

test_input_500k.txt is generated using `--cluster-std=2.2 --n-samples=500000`,
with 20 clusters. Should use `--eps=0.035 --min-samples=115` to query.

test_input_600k.txt is generated using `--cluster-std=2.2 --n-samples=600000`,
with 20 clusters. Should use `--eps=0.035 --min-samples=120` to query.

test_input_700k.txt is generated using `--cluster-std=2.2 --n-samples=600000`,
with 20 clusters. Should use `--eps=0.033 --min-samples=122` to query.

test_input_800k.txt is generated using `--cluster-std=2.2 --n-samples=800000`,
with 20 clusters. Should use `--eps=0.03 --min-samples=120` to query.