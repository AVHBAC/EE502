# BioMed HW5


## Models Chosen
1. K-means clustering 
2. Support Vector Machine

## Features
### MFCC
1. Mean
2. Median
3. Standard Deviation 
4. Skew
5. Kurtosis
6. Max
7. Min

#### Reasoning
I initially used these seven features as a starting point because they were given [here](https://github.com/ksrvap/Audio-classification-using-SVM-and-CNN), and they worked really well when I did not segment out key words and analyzed entire audio tracks. When I started testing on segmented key words my models did quite bit worse so I also started using the pitch features in addition to these mfcc features which has given me good results.

### Non-MFCC
1. Mean pitch
2. Median pitch

#### Reasoning
I tried pitch because it is a common feature used in speaker verification, and it improved my models accuracy quite a bit.   

## Data Set Info
1. 80-20 train-test partition
2. no normalization needed
3. no grid search 

### Cleaning
I ran my code on feature generated from cleaned and uncleaned audio files. The cleaned audio files are cleaned as such `denoised_audio_signal = nr.reduce_noise(y=audio_signal, sr=sample_rate, prop_decrease=0.8)`. The uncleaned audio files are just as they appear out of the box.

### Key Word segmentation
I used the `pocketsphinx` library to identify and segment out instances of my chosen keyword. After segmenting instances of key words I would save them in individual audio files. This code can be found in `process_data/extract_words.ipynb`. It is worth noting that the key word segmentation was marginally worse on the uncleaned data. 

### Key Word selection
For testing I chose  "that" as the keyword used for identification. I chose this because it is a distinct sounding word which appears pretty frequently.


# Results
I ran a total of 8 tests. 4 for K-means clustering and 4 for SVM. I tested how my models worked when the two speakers were of the same gender, and how it worked when the speakers were of different genders. I ran this test on my features obtained from clean and uncleaned audio signals.

- 87 and 201 (Different Gender) 
- 201 and 311 (Same Gender)

## K-Means clustering

### Cleaned 
* Total data points 79

| Metrics                 | 87 and 201 | 201 and 311|
|:-----------------------:|:----------:|:----------:|
| True Positives (TP)     | 11         | 14         |
| True Negatives (TN)     | 16         | 10         |
| False Positives (FP)    | 0          | 6          |
| False Negatives (FN)    | 5          | 2          |
| F1-Score:               | 0.8148     | 0.7778     |
| Accuracy                | 0.8438     | 0.75       |

### Uncleaned
* Total data points 76

| Metrics                 | 87 and 201 | 201 and 311|
|:-----------------------:|:----------:|:----------:|
| True Positives (TP)     | 13         | 12         |
| True Negatives (TN)     | 14         | 11         |
| False Positives (FP)    | 1          | 4          |
| False Negatives (FN)    | 2          | 3          |
| F1-Score:               | 0.8966     | 0.7742     |
| Accuracy                | 0.9        | 0.7667     |

## Support Vector Machine

### Cleaned
* Total data points 79

| Metrics                 | 87 and 201 | 201 and 311|
|:-----------------------:|:----------:|:----------:|
| True Positives (TP)     | 17         | 14         |
| True Negatives (TN)     | 15         | 11         |
| False Positives (FP)    | 0          | 4          |
| False Negatives (FN)    | 0          | 3          |
| F1-Score:               | 1          | 0.8        |
| Accuracy                | 1          | 0.7813     |

### Uncleaned
* Total data points 76

| Metrics                 | 87 and 201 | 201 and 311|
|:-----------------------:|:----------:|:----------:|
| True Positives (TP)     | 15         | 13         |
| True Negatives (TN)     | 14         | 12         |
| False Positives (FP)    | 2          | 4          |
| False Negatives (FN)    | 0          | 2          |
| F1-Score:               | 0.9375     | 0.8125     |
| Accuracy                | 0.9355     | 0.8065     |

## Analysis
The SVM is the clear winner between the two chosen models, as it was able to achieve a perfect accuracy on cleaned data when the speakers where different genders. It is also worth noting that cleaning the data does not necessarily make the model more accurate. For example the accuracy of the SVM prediction when the speakers are of different genders is only marginally effected by clean vs unclean data, while cleaning the data makes a notable different when the genders are different. This is both similar and different to the effect it has on K-means cluster. Similarly clean vs unclean data has little effect when the speakers genders are the same, and clean vs unclean has a notable effect of speakers of different genders, although this effect is opposite to the effect observed in the SVM.