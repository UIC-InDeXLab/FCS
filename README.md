# Fair Content Spread

## Maximizing Fair Content Spread via Edge Suggestion in Social Networks

## Abstract
Content spread inequity is a potential unfairness issue in online social networks, disparately impacting minority groups. In this paper, we view friendship suggestion, a common feature in social network platforms, as an opportunity to achieve an equitable spread of content. In particular, we propose to suggest a subset of potential edges (currently not existing in the network but likely to be accepted) that maximizes content spread while achieving fairness. Instead of re-engineering the existing systems, our proposal builds a fairness wrapper on top of the existing friendship suggestion components.

We prove the problem is NP-hard and inapproximable in polynomial time unless P = NP. Therefore, allowing \new{relaxation of the fairness constraint}, we propose an algorithm based on LP-relaxation and randomized rounding with fixed approximation ratios on fairness and content spread. We provide multiple optimizations, further improving the performance of our algorithm in practice. Besides, we propose a scalable algorithm that dynamically adds subsets of nodes, chosen via iterative sampling, and solves smaller problems corresponding to these nodes. Besides theoretical analysis, we conduct comprehensive experiments on real and synthetic data sets.  Across different settings, our algorithms found solutions with near-zero unfairness while significantly increasing the content spread. Our scalable algorithm could process a graph with half a million nodes on a single machine, reducing the unfairness to around 0.0004 while lifting content spread by 43\%.

## Publications to cite:
[1] Ian Swift, Sana Ebrahimi, Azade Nova, Abolfazl Asudeh. **Maximizing Fair Content Spread via Edge Suggestion in Social Networks.** VLDB, 2022, VLDB Endowment.

[2] Vineet Chaoji, Sayan Ranu, Rajeev Rastogi, and Rushi Bhatt. **Recommendations to boost content spread in social networks.** WWW, 2012, ACM (Continuous Greedy algorithm derived from this work)

[3] Wenguo Yang, Jianmin Ma, Yi Li, Ruidong Yan, Jing Yuan, Weili Wu, and Deying Li. **Marginal gains to maximize content spread in social networks.** IEEE Transactions on Computational Social Systems, 2019, IEEE (IRFA Algorithm implemented from this work)

[4] Liwang Zhu, Qi Bao, and Zhongzhi Zhang. **Minimizing Polarization and Disagreement in Social Networks via Link Recommendation.** NeurIPS, 2021 (SPGREEDY algorithm implemented from this work)

[5] Zhi Yu, Can Wang, Jiajun Bu, Xin Wang, Yue Wu, and Chun Chen. **Friend recommendation with content spread enhancement in social networks.** Information Sciences, 2015 (ACR-FoF algorithm implemented from this work)

## Getting Started

These instructions will get you a copy of the project up and running on your local machine, and allow you to generate some of the result data from the paper [1].

### Prerequisites

What things you need to install the software and how to install them

* [Python 3.10](https://www.python.org/downloads/)

* [NetworkX ](https://networkx.org/documentation/stable/install.html)

* [Numpy 1.23.1](https://numpy.org/install/)

* [Scipy 1.8.1](https://scipy.org/install/)

## Running

From the terminal, run the FCS.py script by entering: ```python3 FCS.py```. This will generate one at a time several pickle files of the form ```'medium-graph-'+algorithm+'-size-'+str(sizePos)+'-trial-'+str(trial)+'.pickle'``` for the following parameters:

* algorithm in: 'ForestFire' (LP-SCALE), 'CG' (Continuous Greedy), 'IterFCS' (LP-Advanced), 'SPGREEDY', 'ACR', and 'IRFA'.

* size in 0-3, where 0 = 500 Nodes, 1 = 1000 Nodes, 2 = 2000 Nodes and 3 = 4000 Nodes.

* trial in 0-4, where each number represents a unique problem input

When loading the pickle file, the results will be stored in a python dictionary.

## Authors

* **[Ian P Swift](https://github.com/Ian-P-Swift)**

* **[Sana Ebrahmi](https://github.com/sanaebrahimi)**

## Additional Acknowlegements

Some code was used from the following open source repositories, and accordingly, we would like to acknowledge them for openly providing their work:

* [Breaking cycles in noisy hierarches](https://github.com/zhenv5/breaking_cycles_in_noisy_hierarchies)
* [Little Ball of Fur](https://github.com/benedekrozemberczki/littleballoffur)

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE.md](LICENSE.md) file for details

<p align="center"><img width="50%" src="https://www.cs.uic.edu/~indexlab/imgs/InDeXLab2.gif"></p>
