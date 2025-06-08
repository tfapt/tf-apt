## From One Attack Domain to Another: Contrastive Transfer Learning with Siamese Networks for APT Detection.

## Abstract

Advanced Persistent Threats (APTs) pose a major cybersecurity challenge due to their stealth, persistence, and adaptability. Traditional machine learning models struggle with APT detection due to imbalanced datasets, complex feature spaces, and limited real-world attack traces. Additionally, APT tactics vary by target, making generalizable detection models difficult to develop. Many proposed anomaly detection methods suffer from key limitations, since they lack transferability when trained in a closed-world setting, performing well on their training dataset but failing when applied to new, unseen attack scenarios. This necessitates a robust transfer learning approach to adapt knowledge across different attack scenarios.

To address these challenges, we propose a hybrid framework integrating Transfer Learning, Explainable AI (XAI), Contrastive Learning, and Siamese Networks. Our pipeline employs an attention-based autoencoder for knowledge transfer, while SHAP-based feature selection optimizes transferability by retaining only the most informative features, reducing computational overhead. Contrastive learning enhances anomaly separability, and Siamese networks align feature spaces, mitigating feature drift.

We evaluate our approach using real-world cybersecurity data from the DARPA Transparent Computing program, augmented with synthetic attack traces to simulate diverse cyber threats. By combining transfer learning, XAI-driven feature selection, contrastive learning, and synthetic data augmentation, our framework establishes a new benchmark for scalable, explainable, and transferable APT detection.

## Architecture

![Architecture](https://github.com/tfapt/tf-apt/blob/main/figure/pipeline.png)


# APT-AutoEncoders 

Run the script:
```shell
root % cd src
src % python main.py [model type: AE, AAE, ADAE] [data file] [labels file]
EXAMPLE:
src % python main.py AE pandex/trace/ProcessAll.csv pandex/trace/trace_pandex_merged.csv
```

# Baseline anomaly detectors

## Quick start

# Dependencies

```
pip install pandas numpy scipy matplotlib scikit-learn pillow h5py pyfpgrowth tensorflow keras rpy2
```


Also, to use the R-based anomaly detection method for rare rule mining, R >= 3.3.3 must be
installed and the following libraries should be installed by running
the following command within R:
```
install.packages(c("stringr","ppls","plyr"))
```


# Running

## Input data

The different anomaly detectors presented below use as input the process centric formal context represented in CSV formats. These data files are publically available in
```
https://gitlab.com/adaptdata
```
## Algorithms

### AVF (Attribute Value Frequency) anomaly detection

As described in the original paper authored by Koufakou et al. 2007, the AVF scores are averaged probabilities of attribute values.
The script `ad.py` implements this anomaly detection method in a batch and a streaming mode.  The batch mode reads and
processes all of the input and then traverses it again to score it, whereas the streaming mode traverses the input just once
and scores each transaction when it is first seen, using the model obtained for the previous input consumed only.
 
The script's options are as follows.

```
usage: ad.py [-h] -i INPUT -o OUTPUT -s avf
             [-m {batch,stream}]
```

Batch scoring is the default mode with this algorithm.
 
### Frequent Pattern Outlier Factor (FPOF) and Outlier Degree (OD)

We have also incorporated implementations of two simple anomaly detectors from the
literature, called FPOF (Frequent Pattern Outlier Factor) and OD (Outlier
Degree):

```
pattern.py [-h] --input INPUT --output OUTPUT [--score {fpof,od}]
                  [--conf CONF] [--minsupp MINSUPP]
```

Here, the `--score` parameter chooses which algorithm to use, `conf` chooses the
confidence level to use for rules and `--minsuppo` chooses the minimum support
level to use.  Both of these take values between 0 and 1; higher tends to be
faster.  `conf` is only relevant for `--score od` while `--minsupp` is
relevant to both algorithms.  

### OC3/Krimp:  

We have also added a Linux binary based on the Krimp system, which mines the
frequent (closed) itemsets and then attempts to identify a subset of them that
"compress the data well".  The end result includes scores for the objects
indicating their "compressed size", which we interpret
as anomaly scores.  There are currently no parameters (Krimp does have some,
but we currently hard-wire them to reasonable values).

```
krimp.py [-h] --input INPUT --output OUTPUT
```

### CompreX:  

We used the Matlab implementation of CompreX available at the official web page of the author:

```
http://www.andrew.cmu.edu/user/lakoglu/tools/CompreX_12_tbox.tar.gz
```
Note that we have adapted the original Comprex code for accepting our input formal context files, and we would provide the modified version of th code we used upon request.

## Checking scores against ground truth

The script `check.py` takes a score file (produced for example by `ad.py` or `krimp.py`),
a ground truth CSV file, and some additional options and produces a
ranking file, which is a CSV file listing just those object IDs found
in the ground truth with their ranks (= how anomalous the object
was).  The checker also prints out the "normalized discounted
cumulative gain" (NDCG) and the "area under ROC curve" (AUC) as
proxies for how "good" the ranking is overall.
The options are as follows:

```
check.py [-h] -i INPUT -o OUTPUT -g GROUNDTRUTH
                [-t TY] [-r]

```

The optional argument `-t TY` allows to specify the
"type" of ground truth ID.  The ground truth CSV files consist of
objectID,type pairs, where the type is a string indicating what kind
of obejct the objectID is.  For example, in the ADM ground truth CSV
files the types are things like `AdmSubject::Node`.  It is helpful to
specify this so that the checker can filter out the ground truth
objects that we do not expect to find in a given context.  If such
objects are included then the NDCG and AUC scores are typically very
low due to the other objects being counted as "misses".

The optional `-r` argument allows to specify whether the scores should
be sorted in decreasing order (i.e. high score = more anomalous).  The
default behavior is to sort in increasing order, which is appropriate
for AVF (small score = more anomalous).



### APT-GAN (Adversarial Generative Neural Networks) anomaly detection

Here we are going to use a GAN to train a neural network for generating fake APT data, then use the generated data to test a given anomaly detection method.
The script APT-GAN/skynet.py trains a GAN using a formal context, thun runs a rule-mining anomaly detection method.
The script's options are as follows.

```GAN.py  -i ./sample/ProcessEvent.cs -g ./sample/cadets_pandex_merged.csv -t 1000```

The results are generated in APT-GAN/output folder, and show the variation of the ndcg scores with the simulated data.


## Running example

Let's run the AVF anomaly detector on the ProcessEvent context (BSD, 1st attack scenario) available in the sample folder (the original file can be downloaded from the data repository at https://gitlab.com/adaptdata/e2/blob/master/pandex/cadets/ProcessEvent.csv.gz).

The user can then call the python script:

```python3 ad.py -i ./sample/ProcessEvent.csv -o ./sample/ScoreProcessEvent.csv -s avf -m batch```

As explained above, this script produces a scoring file where an anomaly score is calculated for each process in the formal context.
Then using the check.py script and the ground truth database (./sample/cadets_pandex_merged.csv), the previously generated scoring file is ranked and the global nDCG as-well as the AUX are calculated for this context.

```python3 check.py -i  ./sample/ScoreProcessEvent.csv -o OutputProcessEvent.csv -g ./sample/cadets_pandex_merged.csv -t AdmSubject::Node```
