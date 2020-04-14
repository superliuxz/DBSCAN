# DBSCAN

## Requirements
C++17; GCC 7.5; CUDA 10.2; Thrust.

## How to build
- For the first time: `git submodule update --init --recursive`
### Build main
- `cmake -DCMAKE_BUILD_TYPE=None -Bbuild -H.`
  - For `cpu-main`
    - set environment variable `AVX=1` to enable AVX instruction set;
    - set environment variable `BIT_ADJ=1` if on average, each vertex has more than `|V|/64`
      number of neighbours.ss
  - Fpr `gpu-main`
    - Modify `gpu/CMakeLists.txt`, replace the architecture code with your corresponding
      hardware
- `cmake --build build --target cpu-main/gpu-main`
### Build tests
- `cmake -DCMAKE_BUILD_TYPE=Debug -Bbuild -H.`
  - For `cpu-test` set `AVX=1` and `BIT_ADJ=1` correspondingly.
  - For `gpu-test` modify `gpu/CMakeLists.txt` correspondingly.
- `cmake --build build --target cpu-test/gpu-test`

## How to run

- [optional] Generate an example using the Python script:
`python3 generate_dateset.py --n-samples=20000 --cluster-std=3.0`
  - 20,000 points with 3.0 stddiv;

- [optional] Cluster the input and visualize it using Sklearn:
`python3 dbscan.py --input-name=test_input.txt --eps=0.1 --min-pts=12`
  - 0.1 radius and 12 neighbour points for clustering.

### CPU algorithm
- `./build/bin/cpu-main --input=test_input.txt --eps=0.1 --min-pts=12`.
  - Append `--print` to see the cluster ids.
  - Append `--num-threads=K` to speed up the processing.

### GPU algorithm
- `./build/bin/gpu-main --input=test_input.txt --eps=0.1 --min-pts=12`.
  - Append `--print` to see the cluster ids.

## Test data
test_input_20k.txt is generated using `--cluster-std=2.2 --n-samples=20000`,
with 4 clusters. Should use `--eps=0.15 --min-pts=180` to query.

test_input_50k.txt is generated using `--cluster-std=2.2 --n-samples=50000`,
with 20 clusters. Should use `--eps=0.09 --min-pts=110` to query.

test_input_100k.txt is generated using `--cluster-std=2.2 --n-samples=100000`,
with 20 clusters. Should use `--eps=0.07 --min-pts=100` to query.

test_input_200k.txt is generated using `--cluster-std=2.2 --n-samples=200000`,
with 20 clusters. Should use `--eps=0.05 --min-pts=100` to query.

test_input_300k.txt is generated using `--cluster-std=2.2 --n-samples=300000`,
with 20 clusters. Should use `--eps=0.044 --min-pts=100` to query.

test_input_400k.txt is generated using `--cluster-std=2.2 --n-samples=400000`,
with 20 clusters. Should use `--eps=0.04 --min-pts=105` to query.

test_input_500k.txt is generated using `--cluster-std=2.2 --n-samples=500000`,
with 20 clusters. Should use `--eps=0.035 --min-pts=115` to query.

test_input_600k.txt is generated using `--cluster-std=2.2 --n-samples=600000`,
with 20 clusters. Should use `--eps=0.035 --min-pts=120` to query.

test_input_700k.txt is generated using `--cluster-std=2.2 --n-samples=600000`,
with 20 clusters. Should use `--eps=0.033 --min-pts=122` to query.

test_input_800k.txt is generated using `--cluster-std=2.2 --n-samples=800000`,
with 20 clusters. Should use `--eps=0.03 --min-pts=120` to query.