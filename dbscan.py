import argparse
import numpy as np
import time

from sklearn.cluster import DBSCAN

RANDOM_STATE = 42

parser = argparse.ArgumentParser(
  description='Perform DBSCAN clustering on test input')

# DBSCAN clustering
parser.add_argument('--input', type=str,
                    help='visualize the clustering result')
parser.add_argument('--print', action='store_true',
                    help='print the clustered result to stdout')
parser.add_argument('--eps', type=float, default=0.3,
                    help='clustering radius')
parser.add_argument('--min-pts', type=int, default=9,
                    help='number of points to be considered as a Core')
parser.add_argument('--algorithm', type=str, default=['kd_tree'],
                    help="algorithm used to fixed-radius neighbour query. "
                         "Choose between 'brute' and 'kd_tree'",
                    choices=['brute', 'kd_tree'], nargs=1)

args = parser.parse_args()

points = []
with open(args.input) as fin:
  for line in fin.readlines():
    line = line.split(" ")
    if len(line) != 3:
      continue
    points.append(np.array([float(line[1]), float(line[2])]))

points = np.array(points)

N = len(points)

t = time.time()
assert len(args.algorithm) == 1
db = DBSCAN(eps=args.eps, min_samples=args.min_pts + 1,
            algorithm=args.algorithm[0]).fit(points)
print(f"DBSCAN takes {time.time() - t:.4f} seconds")

if args.print:
  for l in db.labels_:
    print(l)

core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)
unique, cnt = np.unique(core_samples_mask, return_counts=True)
cnt = dict(zip(unique, cnt))
n_core_ = cnt.get(True, 0)
n_boundary_ = cnt.get(False, 0) - n_noise_


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

title = f'Num of clusters: {n_clusters_}; ' \
        f'core / boundary / noise ratio: ' \
        f'{n_core_ / N:.3f} / {n_boundary_ / N:.3f} / {n_noise_ / N:.3f}'
print(title)
plt.title(title)
plt.show()
