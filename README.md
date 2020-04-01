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

0. Generate an example using the Python script:
`python3 generate_dateset.py --n-samples=20000 --cluster-std=3.0`
    - 20,000 points with 3.0 stddiv;

1. Cluster the input and visualize it:
`python3 dbscan.py --input-name=test_input.txt --eps=0.1 --min-samples=12`
    - 0.1 radius and 12 neighbour points for clustering.

2. `./build/cpu/main/cpu-main --input=$(pwd)/test_input.txt --eps=0.1 --min-samples=12 --print`.

## Misc
test_input_100k.txt is generated using `--cluster-std=2.2 --n-samples=100000`,
with 20 clusters. Should use `--eps=0.07 --min-samples=100` to query.

test_input_200k.txt is geenrated using `--cluster-std=2.2 --n-samples=200000`,
with 20 clusters. Should use `--eps=0.05 --min-samples=100` to query.

test_input_300k.txt is geenrated using `--cluster-std=2.2 --n-samples=300000`,
with 20 clusters. Should use `--eps=0.044 --min-samples=100` to query.

test_input_400k.txt is geenrated using `--cluster-std=2.2 --n-samples=400000`,
with 20 clusters. Should use `--eps=0.04 --min-samples=105` to query.

test_input_500k.txt is geenrated using `--cluster-std=2.2 --n-samples=500000`,
with 20 clusters. Should use `--eps=0.035 --min-samples=115` to query.

test_input_800k.txt is geenrated using `--cluster-std=2.2 --n-samples=800000`,
with 20 clusters. Should use `--eps=0.03 --min-samples=120` to query.