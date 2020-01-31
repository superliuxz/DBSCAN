import argparse
import numpy as np

from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42

parser = argparse.ArgumentParser(
  description='Generate input and perform DBSCAN.')

# random points generation
parser.add_argument('--generate', action='store_true',
                    help='generate test input (overwrite the existing one). '
                         'this flag disables other feature of the script.')
parser.add_argument('--output-name', type=str, default='test.out',
                    help='the output file name')
parser.add_argument('--n-samples', type=int, default=500,
                    help='number of sample points')
parser.add_argument('--cluster-std', type=float, default=2.0,
                    help='stddiv for generating clustered points')
# DBSCAN clustering
parser.add_argument('--visualize', action='store_true',
                    help='visualize the clustering result')
parser.add_argument('--print-cluster-id', action='store_true',
                    help='print the clustered result to stdout')
parser.add_argument('--eps', type=float, default=0.3,
                    help='clustering radius')
parser.add_argument('--min_samples', type=int, default=9,
                    help='number of points to be considered as a Core')

args = parser.parse_args()

centers = [[10, 10], [-10, -10], [10, -10], [-10, 10]]
points, _ = make_blobs(n_samples=args.n_samples, centers=centers,
                       cluster_std=args.cluster_std, random_state=RANDOM_STATE)

points = StandardScaler().fit_transform(points)

N = len(points)

if args.generate:
  with open(args.output_name, 'w') as fout:
    fout.write(f'{N}\n')
    for i, p in enumerate(points):
      fout.write(f"{i} {p[0]:.6f} {p[1]:.6f}\n")
  exit(0)

db = DBSCAN(eps=args.eps, min_samples=args.min_samples + 1).fit(points)

if args.print_cluster_id:
  for l in db.labels_:
    print(l)

if args.visualize:
  core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
  core_samples_mask[db.core_sample_indices_] = True
  labels = db.labels_

  # Number of clusters in labels, ignoring noise if present.
  n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
  n_noise_ = list(labels).count(-1)

  # ############################################################################
  # Plot result
  import matplotlib.pyplot as plt

  # Black removed and is used for noise instead.
  unique_labels = set(labels)
  colors = [
    plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
  for k, col in zip(unique_labels, colors):
    if k == -1:
      # Black used for noise.
      col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = points[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14)

    xy = points[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

  plt.title('Estimated number of clusters: %d' % n_clusters_)
  plt.show()
