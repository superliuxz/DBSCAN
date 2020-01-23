# Pagerank

A single threaded Pagerank implementation.

## Dependency:
- Gperftools (optional)

## Build:
1. `cmake -Bbuild -H.`
2. `cmake --build build --target Pagerank`
3. `./build/Pagerank test.txt 1000` (input: test.txt; iteration: 1000)
