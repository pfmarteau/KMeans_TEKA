# KMeans-TEKA
## Implementation of the Kmeans algorithm for clustering time series using TEKA (Time Elastic Kernel Averaging of set of time series [1]). 
## Requirements
- python3.*
- matplotlib
- numpy
- aeon toolbox
- ScikitLearn
- pyTEKA (https://github.com/pfmarteau/py-TEKA)

# This code includes KMedoids_KDTW, an implementation of the kmedoids algorithm using the KDTW kernel [2] (that is used by TEKA). KMedoids_KDTW provides the initials centroids to the TEKA algorithm.

## Testing Kmeans-TEKA on UCR/AEON datasets
$ python3 test_kmeans_teka.py --dataset NATOPS --sigma 1. --epsilon 1e-300


Please cite this article if you wish to reference TEKA or Kmeans_TEKA:

[1] Marteau, P.F., Times series averaging and denoising from a probabilistic perspective on time-elastic kernels International Journal of Applied Mathematics and Computer Science, Vol 29, num 2, pp 375–392, De Gruyter editor, 2019.\
[https://arxiv.org/abs/1611.09194], [bibtex](bibtex/marteau2019.bib)
