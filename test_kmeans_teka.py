#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 11:31:40 2025

@author: Pierre-François Marteau
"""

import numpy as np
import sys, os, random, time
import argparse
from aeon.datasets import load_classification
from aeon.transformations.collection import Normalizer
from tslearn.clustering import silhouette_score
from sklearn.metrics.cluster import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.metrics import normalized_mutual_info_score, homogeneity_score, completeness_score, v_measure_score, fowlkes_mallows_score
import KDTWKMedoids
import TekaKernelKMeans

verbose = False

def evaluate(true_labels, labels):
        print("ARI:", adjusted_rand_score(true_labels, labels))
        print("AMI:", adjusted_mutual_info_score(true_labels, labels))
        print("NMI:", normalized_mutual_info_score(true_labels, labels))
        print("Homogeneity:", homogeneity_score(true_labels, labels))
        print("Completeness:", completeness_score(true_labels, labels))
        print("V-measure:", v_measure_score(true_labels, labels))
        print("Fowlkes-Mallows:", fowlkes_mallows_score(true_labels, labels))
    
def kmedoids_kdtw(X, args: dict ={"n_clusters":3, "sigma":1., "epsilon":1e-3, "n_init":1}):
    kdtw_kmd = KDTWKMedoids.KDTWKMedoids(n_clusters=args["n_clusters"], method="pam", init='random',
        distance_params={"sigma": args["sigma"], "epsilon":args["epsilon"]}, n_init=args["n_init"], max_iter=100, verbose=verbose)
    kdtw_kmd.fit(X)
    return kdtw_kmd.predict(X) 

##pfm: 
def kmeans_teka(X, args: dict ={"n_clusters":3, "sigma":1., "epsilon":1e-300, "n_init":1}):
    print("sigma:",args["sigma"], "epsilon:", args["epsilon"])
    init_type = "kmedoids"
    teka_km = TekaKernelKMeans.TekaKernelKMeans(n_clusters=args["n_clusters"], init_type=init_type, 
            kernel_params={"sigma": args["sigma"], "epsilon":args["epsilon"]}, n_init=args["n_init"], max_iter=100, verbose=verbose)
    teka_km.fit(X)
    return teka_km.predict(X) 

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='NATOPS', help='AEON/UCR dataset name to process')
parser.add_argument('--sigma', default=1., help='sigma meta parameter')
parser.add_argument('--epsilon', default=1e-300, help='epsilon meta parameter')
parser.add_argument('--n_ts_max', default=10000, help='max number of processed time series')
try:
    args = parser.parse_args()
except SystemExit:
    os._exit(1)

print("Processing", args.dataset)
X, y = load_classification(name=args.dataset)
n_clusters = len(np.unique(y))
    
normaliser = Normalizer()
X = normaliser.fit_transform(X)
# Transpose X to match tslearn's expected shape (n_ts, sz, d)
X = X.swapaxes(1, 2)
N = min(len(y),int(args.n_ts_max))
X = X[:N,:,:]
y = y[:N]
print("#ts:", N, "#clusters", n_clusters, "length:", len(X[0]), "dim:", len(X[0,0]))

#Test KMedois_KDTW
method_name = "KMedoids_KDTW"
print("running", method_name)
start_time = time.time()
labs = kmedoids_kdtw(X, args={"n_clusters":n_clusters, "sigma":float(args.sigma), "epsilon":float(args.epsilon), "n_init": 10})
runtime = time.time() - start_time
print(f"Completed: {args.dataset}, {method_name}, Runtime: {runtime:.2f} seconds")
print("evaluate", method_name)
evaluate(y, labs)

print()

#Test KMeans_TEKA
method_name = "KMeans_TEKA"
print("running", method_name)
start_time = time.time()
labs = kmeans_teka(X, args={"n_clusters":n_clusters, "sigma":float(args.sigma), "epsilon":float(args.epsilon), "n_init": 10})
runtime = time.time() - start_time
print(f"Completed: {args.dataset}, {method_name}, Runtime: {runtime:.2f} seconds")
print("evaluate", method_name)
evaluate(y, labs)








