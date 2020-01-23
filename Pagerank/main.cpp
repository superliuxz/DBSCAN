#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <unordered_map>

#include "include/matrix.h"

int main(int argc, char *argv[]) {
  std::cout << "Hello, World!" << std::endl;

  std::ifstream ifs(argv[1]);
  std::unordered_map<int, int> degrees;
  // number of nodes
  int n;
  ifs >> n;
  for (int i = 0; i < n; ++i) {
    degrees[i] = 0;
  }
  // construct m
  pagerank::Matrix m(n, n);
  // edges, u->v
  int u, v;
  while (ifs >> u >> v) {
    m(u, v) = 1.0f;
    ++degrees[v];
  }
  // i -> j
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      m(i, j) /= degrees[j]; // each column must sum to 1
    }
  }
  std::cout << "m:" << std::endl << m << std::endl;
  // m_hat
  float damping = 0.85f;
  pagerank::Matrix
      m_hat = (m * damping + (1 - damping) / static_cast<float>(n));

  std::cout << "m_hat:" << std::endl << m_hat << std::endl;

  int iter = std::stoi(argv[2]);

  // init rank
  pagerank::Matrix rank(n, 1);

  std::random_device rd;
  std::seed_seq ssq{rd()};
  std::default_random_engine engine{ssq};
  std::uniform_int_distribution dist(0, 10000);

  float sum = 0.0f;
  for (int i = 0; i < n; ++i) {
    rank(i, 0) = static_cast<float>(dist(engine) / (10001.0));
    sum += rank(i, 0);
  }
  // normalize rank with 1-norm (so rank sums to 1)
  rank = rank / sum;

  // iterations
  for (int i = 0; i < iter; ++i) {
    rank = m_hat * rank;
  }

  std::cout << rank << std::endl;

  return 0;
}




