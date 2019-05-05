import glob
import json
import math
import operator

#Path to the token data
DATA_DIRECTORY = "/user/cse842/SentimentData/"
POS_TOKENS_DIRECTORY = DATA_DIRECTORY + "tokens/pos/*"
NEG_TOKENS_DIRECTORY = DATA_DIRECTORY + "tokens/neg/*"
FOLD1_RANGE = range(0,233)
FOLD2_RANGE = range(233,466)
FOLD3_RANGE = range(466,700)
POS_FILES = glob.glob(POS_TOKENS_DIRECTORY)
NEG_FILES = glob.glob(NEG_TOKENS_DIRECTORY)
#arrays of  data ([tokens], review 1/0)
FOLD1_DATA = []
FOLD2_DATA = []
FOLD3_DATA = []

#
# loads in an array of files and 
# assigns the data with teh corresponding
# rating value (-1 for negative, 1 for positive)
def format_data(arr_files, rating_value):
    #loop and open the files
    for filepath in arr_files:
        f = open(filepath,"r" , encoding="ISO-8859-1") 
        #read review txt
        fdata = f.read()

        #split into unigrams
        grams = fdata.split()

        #parse the filename/path to get the fold its in
        filenumber = int(filepath.replace('\\', '/').split('/')[-1][2:5])
    
        #place into folds
        if filenumber in FOLD1_RANGE:
            FOLD1_DATA.append((grams, rating_value))
        elif filenumber in FOLD2_RANGE:
            FOLD2_DATA.append((grams, rating_value))
        elif filenumber in FOLD3_RANGE:
            FOLD3_DATA.append((grams, rating_value))
        f.close()

#
# Naive Bayes Classifier to train/predict data for pos/neg reviews
#
class SentimentNaiveBayes:
    
    param_file_name = 'model_params.json'
    
    def __init__(self):
        self.pos_probabilities = {}
        self.neg_probabilities = {}
        self.pos_vocab_count = 1
        self.neg_vocab_count = 1
        self.pos_prob = 0
        self.neg_prob = 0
        self.laplace = .1
        
    # exports the model to the param file name
    # @param param_file_name JSON file containing parameters
    def export_params(self):
        with open(self.param_file_name, 'w') as outfile:
            json.dump({
                'pos_prob': self.pos_prob,
                'neg_prob': self.neg_prob,
                'laplace': self.laplace,
                'pos_vocab_count': self.pos_vocab_count,
                'neg_vocab_count': self.neg_vocab_count,
                'pos_probabilities': self.pos_probabilities,
                'neg_probabilities': self.neg_probabilities
            }, outfile)
        
    # imports the model from the param file name
    # @param param_file_name JSON file containing parameters
    def import_params(self):
        model_params = {}
        with open(self.param_file_name, 'r') as f:
            model_params = json.load(f)
        self.pos_probabilities = model_params['pos_probabilities']
        self.neg_probabilities = model_params['neg_probabilities']
        self.pos_vocab_count = model_params['pos_vocab_count']
        self.neg_vocab_count = model_params['neg_vocab_count']
        self.pos_prob = model_params['pos_prob']
        self.neg_prob = model_params['neg_prob']
        self.laplace = model_params['laplace']
        
    # constructor
    # @param training_data array of tuples ([tokens], prediction)
    def train(self, training_data):
        pos_doc_count = 0
        neg_doc_count = 0
        total_pos_count = 0 #number of tokens in pos
        total_neg_count = 0 #number of tokens in neg
        pos_counts = {} 
        neg_counts = {} 
        vocab = {} #v
        
        #loop the docs and tokens in each and assign
        for doc in training_data:
            if doc[1] == 1:
                pos_doc_count += 1
            else:
                neg_doc_count += 1
            
            #loop the tokens in this doc and count
            for tok in doc[0]:
                if tok in vocab:
                    vocab[tok] += 1
                    if doc[1] == 1:
                        total_pos_count +=1
                        if tok in pos_counts:
                            pos_counts[tok] += 1
                        else:
                            pos_counts[tok] = 1
                    else:
                        total_neg_count +=1
                        if tok in neg_counts:
                            neg_counts[tok] += 1
                        else:
                            neg_counts[tok] = 1
                else:
                    vocab[tok] = 1
                    if doc[1] == 1:
                        total_pos_count +=1
                        pos_counts[tok] = 1
                    else:
                        total_neg_count +=1
                        neg_counts[tok] = 1
                        
        # then calculate p(c) and p(w,c) for all w
        self.pos_prob = float(pos_doc_count)/float(pos_doc_count + neg_doc_count)
        self.neg_prob = float(neg_doc_count)/float(pos_doc_count + neg_doc_count)  
        self.pos_vocab_count = float(total_pos_count) + len(vocab.keys())
        self.neg_vocab_count = float(total_neg_count) + len(vocab.keys())
        print("Calculating p(w,c)...")
        
        #loop the vocab set and find p(w,c)
        for w, c in vocab.items():
            pwc_pos = self.find_prob(pos_counts.get(w), self.pos_vocab_count)
            pwc_neg = self.find_prob(neg_counts.get(w), self.neg_vocab_count)
            self.neg_probabilities[w] = pwc_neg
            self.pos_probabilities[w] = pwc_pos
        
        self.export_params()
        print("Model Trained & Exported: ",pos_doc_count, " positive docs & ", neg_doc_count, " negative docs")
    
    # calculate p(w,c)
    # @param occurances is the number of times the word occurs with the class c
    # @param vocab_count is the count(w,c) + |v|
    def find_prob(self, occurances, vocab_count):
        if occurances is None:
            return float(self.laplace)/(vocab_count)
        else:
            return float(occurances + self.laplace)/(vocab_count)
        
    #
    # evaluate the prediction for a given doc and class 
    # @param class_value 1 for positive review, -1 for negative review
    # @param doc array of tokens from the document
    def predict(self, class_value, doc):
        prob = 0
        for tok in doc:
            if class_value == 1:
                if tok in self.pos_probabilities:
                    prob = prob + math.log2(float(self.pos_probabilities[tok]))
                else:
                    prob = prob + math.log2(float(self.find_prob(None, self.pos_vocab_count)))
            else:       
                if tok in self.neg_probabilities:
                    prob = prob + math.log2(float(self.neg_probabilities[tok]))
                else:
                    prob = prob + math.log2(float(self.find_prob(None, self.neg_vocab_count)))
        if class_value == 1:
            return (prob + math.log2(float(self.pos_prob)))
        else:       
            return (prob + math.log2(float(self.neg_prob)))
        
    # test the trained classifier and output the accuracy %
    # @param testing_data array of tuples ([tokens], prediction)
    def accuracy(self, testing_data):
        #loop the docs and tokens in each
        correct_predictions = 0 
        all_predictions = 0
        for doc in testing_data:
            predict_pos = self.predict(1, doc[0])
            predict_neg = self.predict(-1, doc[0])
            if predict_pos > predict_neg:
                #more likely pos
                if doc[1] == 1:
                    correct_predictions +=1
            else:
                #more likely neg
                if doc[1] == -1:
                    correct_predictions +=1
            all_predictions += 1
           # print(predict_pos, " ", predict_neg, " actual:", doc[1])

        #make sure the model was trained beforehand
        if all_predictions < 1:
            return "Model was not trained, couldn't determine accuracy"
        else:
            return (correct_predictions / float(all_predictions))
