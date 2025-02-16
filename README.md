# KMeans-TEKA
## Implementation of the Kmeans algorithm for clustering time series using TEKA (Time Elastic Kernel Averaging of set of time series [1]). 
## Requirements
- python3.*
- matplotlib
- numpy
- aeon toolbox
- ScikitLearn toolbox
- pyTEKA (https://github.com/pfmarteau/py-TEKA)

## Note
This code includes KMedoids_KDTW, an implementation of the kmedoids algorithm using the KDTW kernel [2] (that is used by TEKA). KMedoids_KDTW provides the initials centroids to the TEKA algorithm.

## Testing Kmeans-TEKA on UCR/AEON datasets
```text
$ python3 test_kmeans_teka.py --dataset NATOPS --sigma 1. --epsilon 1e-300

usage: test_kmeans_teka.py [-h] [--dataset DATASET] [--sigma SIGMA] [--epsilon EPSILON] [--n_ts_max N_TS_MAX]

options:

  -h, --help           show this help message and exit
  --dataset DATASET    AEON/UCR dataset name to process
  --sigma SIGMA        sigma meta parameter
  --epsilon EPSILON    epsilon meta parameter
  --n_ts_max N_TS_MAX  max number of processed time series
 ``` 


## References

[1] Marteau, P.F., Times series averaging and denoising from a probabilistic perspective on time-elastic kernels International Journal of Applied Mathematics and Computer Science, Vol 29, num 2, pp 375–392, De Gruyter editor, 2019.\
[https://arxiv.org/abs/1611.09194], [bibtex](bibtex/marteau2019.bib)

[2] P. F. Marteau and S. Gibet, "On Recursive Edit Distance Kernels With Application to Time Series Classification", 
in IEEE Transactions on Neural Networks and Learning Systems, vol. 26, no. 6, pp. 1121-1133, June 2015. 
[https://arxiv.org/abs/1005.5141], [bibtex](bibtex/marteau2015.bib)
