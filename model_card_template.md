# Model Card
This papers describes a model card:https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Model created as part of Udacity ML DevOps course by Ross Fitzgerald

## Intended Use
This model should be used to predict the salary of a car person based off a list of attributes. The categorial attributes are split into slices and their performaces are also 
measured.

## Training Data
We used 80% of data for trainig on and stratified based on salary

## Evaluation Data
We held-out 20% of data for testing

## Metrics
Metrics used: fbeta_score, precision_score and recall_score.


