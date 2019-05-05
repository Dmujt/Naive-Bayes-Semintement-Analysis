# Assignment 1 - Naive Bayes Sentiment Analysis
Dena Mujtaba

## Part 1

### Project Files
Everything is written in python (Python 3.4+). The follow are the files in this project and their use:
 - `model.py` : This contains the code for loading the ratings text data and assigning them into three folds, as well as store the code for the Naive Bayes model itself.
 - `train.py` : This file is used to train an instance of the model and export to a parameter file. Running this is described below.
 - `test.py` : This is used to import a model and test it with a set of test data to return an accuracy. Running this is described below.
 - `mutual_information.py` : This script to calculate the average accuracy across the three folds and calculates the mutual information for part 2 of the assignment.
 - `model_params.json` : This file is the export of the model and stores the model parameters (further described below).

### Training and Testing the Model
##### Training
To train the model, run the following in the command line (you can replace the fold parameters with `fold1`, `fold2`, or `fold3`, so long as two are provided):
```bash
python3.4 train.py fold1 fold2
```
This will train an instance of the Naive Bayes model and export the parameters to a JSON file (described below).

##### Testing
To test the model that was exported, run the following in the command line (you can replace the fold parameters with `fold1`, `fold2`, or `fold3`, so long as one are provided):
```bash
python3.4 test.py fold3
```
This will output the accuracy of the pre-trained model. If a model hasn't been pre-trained, and the test script is run, it will result in an error.

### Model Parameter File
This file is the `model_params.json` file that is exported after the model is trained. This is a JSON file with the following keys/format:
```json
{
    "pos_prob": 			"p(c=positive)",
    "neg_prob": 			"p(c=negative)",
    "laplace": 				"Laplace k",
    "pos_vocab_count": 		"Size of vocab in the positive rated docs",
    "neg_vocab_count": 		"Size of vocab in the negative rated docs",
    "pos_probabilities": 	"p(w,c=postive)",
    "neg_probabilities": 	"p(w,c=negative)",
}
```

### Average Accuracy
Running all variations of folds results in the following (this can be seen by running the `mutual_information.py` file, described below):
```bash
PREPARED FOLD 1 DOCS: 459
PREPARED FOLD 2 DOCS: 462
PREPARED FOLD 3 DOCS: 465
Calculating p(w,c)...
Model Trained & Exported:  462  positive docs &  459  negative docs
Calculating p(w,c)...
Model Trained & Exported:  462  positive docs &  462  negative docs
Calculating p(w,c)...
Model Trained & Exported:  464  positive docs &  463  negative docs
The average accuracy across 3-folds is  77.05378230362417 %
```
Therefore the average accuracy is `77.05%`.

## Part 2

### Running the Script
The script that holds the results for part 2 is `mutual_information.py` which can be run with the following:
```bash
python3.4 mutual_information.py
```
This outputs the results shown below and the calculated mutual information for the top ten words.

#### Results
##### Calculated Mutual Information
```bash
Mutual information for 'the':  0.0
Mutual information for 'like':  4.186007896989259e-06
Mutual information for 'good':  0.00043644718654328556
Mutual information for 'movie':  0.0008851627021837314
```
##### Top 10 Words
```bash
Top 10 Words:
0 )  bad
1 )  worst
2 )  wasted
3 )  ?
4 )  wonderfully
5 )  boring
6 )  mess
7 )  awful
8 )  dull
9 )  outstanding
```
##### Unexpected Words
The following were words with unexpected mutual information scores:
```bash
Unexpected:
?  0.02811521313539274
also  0.019388313059304232
both  0.017643657909817687
```
These are unexpected because they have a higher mutual information score than other words,
even though they are not words that would seem to correlate with a positive or negative review (for instance, also is a common word to continue a sentence, and can be used
in both a postive and negative tone).
