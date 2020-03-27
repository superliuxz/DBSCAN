import argparse

from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42

parser = argparse.ArgumentParser(
  description='Generate input and for DBSCAN.')

# random points generation
parser.add_argument('--output-name', type=str, default='test_input.txt',
                    help='the generated test file name')
parser.add_argument('--n-samples', type=int, default=500,
                    help='number of sample points')
parser.add_argument('--cluster-std', type=float, default=2.0,
                    help='stddiv for generating clustered points')

args = parser.parse_args()

centers = [[row, col] for col in range(0, 50, 10) for row in range(0, 40, 10)]
points, _ = make_blobs(n_samples=args.n_samples, centers=centers,
                       cluster_std=args.cluster_std, random_state=RANDOM_STATE)

points = StandardScaler().fit_transform(points)

N = len(points)

with open(args.output_name, 'w') as fout:
  fout.write(f'{N}\n')
  for i, p in enumerate(points):
    fout.write(f"{i} {p[0]:.6f} {p[1]:.6f}\n")
