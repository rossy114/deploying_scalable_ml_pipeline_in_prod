# Model Card
This papers describes a model card:https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Model created as part of Udacity ML DevOps course by Ross Fitzgerald

## Intended Use
This model should be used to predict the salary of a person based off a number of attributes. The categorial attributes are split into slices and their performaces are also measured.

## Training Data
We used 80% of data for trainig on and stratified based on salary

## Evaluation Data
We held-out 20% of data for testing

## Metrics
Metrics and Performance on overall dataset: 
											Fbeta Score:0.70098
											Precision Score:0.7772 
											Recall Score:0.6384


## Ethical Considerations
The model was never debiased and there were inbalances in certain variable slices so more work may needed to be done in this area.

## Caveats and Recommendations
Some slices performed condsiderably worse then other as shown in the slice_outut.txt. More data for some of the underepresented variable slices may lead to improvements.