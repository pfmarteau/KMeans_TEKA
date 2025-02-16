# KMeans-TEKA
## Implementation of the Kmeans algorithm for clustering time series using TEKA (Time Elastic Kernel Averaging of set of time series [1]). 
## Requirements
- python3.*
- matplotlib
- numpy
- aeon toolbox
- ScikitLearn
- pyTEKA (https://github.com/pfmarteau/py-TEKA)


## Testing Kmeans-TEKA on the Cylinder, Bell, Funnel dataset
$ python3 testCBF.py
<p float="left">
  <img src="fig/CBF_ITEKA_15_c.jpg" width="200" height="150">
  <img src="fig/CBF_ITEKA_15_b.jpg" width="200" height="150">
  <img src="fig/CBF_ITEKA_15_f.jpg" width="200" height="150">
  <img src="fig/CBF_ITEKA_Centroids.jpg" width="200" height="150">
</p>
<p float="left">
  <img src="fig/CBF_ITEKA_15_c0.jpg" width="200" height="150">
  <img src="fig/CBF_ITEKA_15_b0.jpg" width="200" height="150">
  <img src="fig/CBF_ITEKA_15_f0.jpg" width="200" height="150">
  <img src="fig/CBF_ITEKA_15_b_t0.jpg" width="200" height="150">
</p>

Please cite this article if you wish to reference TEKA or Kmeans_TEKA:

[1] Marteau, P.F., Times series averaging and denoising from a probabilistic perspective on time-elastic kernels International Journal of Applied Mathematics and Computer Science, Vol 29, num 2, pp 375–392, De Gruyter editor, 2019.\
[https://arxiv.org/abs/1611.09194], [bibtex](bibtex/marteau2019.bib)
