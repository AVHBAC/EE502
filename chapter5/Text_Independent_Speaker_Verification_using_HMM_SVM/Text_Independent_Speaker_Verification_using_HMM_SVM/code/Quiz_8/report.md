## Model Parameters 
### HMM
For the HMM I experimented a bit with different hyper parameters and I found that the following gave the best metrics. It is worth mentioning that setting the `tied` covariance type trains considerably slower than the other I tried, but it was not a big deal as it was only slower 30 seconds slower.
- `n_components=5`
- `covariance_type="tied"`
- `n_iter=1000` 

### K-means clustering
For this I used all of the default values except the number of clusters which I set to 2 because I am doing binary classification. When looking through the documentation for `sklearn` K-means clustering implementation I did not see any hyper parameters that would effect what I was doing so I just left them all default.


## Comparison 
To compare the two models I created a five number summary plus mean of 5 different metrics; accuracy, precision, recall, F1-score, and ROC AUC. For a break down by test subject please see the metrics folder and the ROC plots.

| HMM       | Min    |   Q1   | Median |  Q3    | Max    | Mean    |
|:---------:|:------:|:------:|:------:|:------:|:------:|:-------:|
|  accuracy | 0.6410 | 0.7479 | 0.7692 | 0.8034 | 0.8462 |  0.7675 |
|  precision| 0.6622 | 0.6875 | 0.7088 | 0.7437 | 0.8000 |  0.7183 |
|  recall   | 0.3509 | 0.8728 | 0.9298 | 0.9518 | 1.0000 |  0.8754 |
|  F1-score | 0.4878 | 0.7700 | 0.8000 | 0.8203 | 0.8636 |  0.7782 |
|  ROC AUC  | 0.7784 | 0.8631 | 0.8892 | 0.9036 | 0.9617 |  0.8804 |

| K-Means Clustering | Min    |   Q1   | Median |  Q3    | Max    | Mean   |
|:--------:|:------:|:------:|:------:|:------:|:------:|:------:|
| accuracy | 0.5983 | 0.6966 | 0.7436 | 0.7778 | 0.8462 | 0.7350 |
| precision| 0.6389 | 0.6656 | 0.6812 | 0.7285 | 0.7600 | 0.6941 |
| recall   | 0.3333 | 0.7500 | 0.8596 | 0.9298 | 1.0000 | 0.8175 |
| F1-score | 0.4471 | 0.7084 | 0.7703 | 0.7967 | 0.8636 | 0.7422 | 
| ROC AUC  | 0.6804 | 0.7725 | 0.8370 | 0.8841 | 0.9737 | 0.8249 | 


| Percent Diff HMM and Kmean | Min    |   Q1   | Median |  Q3    | Max    | Mean    |
|:--------:|:------:|:--------:|:-------:|:-------:|:------:|:-------:|
| accuracy | -6.6615 | -6.8592  | -3.3281 | -3.1865 | 0.0000 | -4.0071 |
| precision| -3.5186 | -3.1855  | -3.8939 | -2.0438 | -5.0000| -3.5284 |
| recall   | -5.0157 | -14.0697 | -7.5500 | -2.3114 | 0.0000 | -5.7894 | 
| F1-score | -8.3436 | -8.0000  | -3.7125 | -2.8770 | 0.0000 | -4.5866 | 
| ROC AUC  | -12.5899| -10.4970 | -5.8704 | -2.1580 | 1.2478 | -5.9735 |

Both of the models did pretty well with HMM being the clear winner between the two. It is worth noting that the difference between the two becomes more apparent on the lower ends meaning that inaccurate K-means models are much more inaccurate than inaccurate HMM models.  