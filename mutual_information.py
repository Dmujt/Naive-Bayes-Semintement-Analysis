from model import SentimentNaiveBayes, format_data,  FOLD1_DATA, FOLD2_DATA, FOLD3_DATA, POS_FILES, NEG_FILES
import math
import operator

#place corresponding data the folds
format_data(POS_FILES, 1)
format_data(NEG_FILES, -1)

print("PREPARED FOLD 1 DOCS: " + str(len(FOLD1_DATA)))
print("PREPARED FOLD 2 DOCS: " + str(len(FOLD2_DATA)))
print("PREPARED FOLD 3 DOCS: " + str(len(FOLD3_DATA)))

# start a model instance, run the training and test
# and return the accuracy
# @param train_data training data folds
# @param test testing data fold
def run_trial(train_data, test):
    model = SentimentNaiveBayes()
    model.train(train_data)
    return model.accuracy(test)

#find the average accuracy
total = 0
total += run_trial((FOLD1_DATA + FOLD2_DATA), FOLD3_DATA)
total += run_trial((FOLD1_DATA + FOLD3_DATA), FOLD2_DATA)
total += run_trial((FOLD3_DATA + FOLD2_DATA), FOLD1_DATA)

print("The average accuracy across 3-folds is ", ((total/3)*100), "%")

#combine the three folds into one set
token_data = FOLD1_DATA + FOLD2_DATA + FOLD3_DATA

#dictionary with word : (w_occurs_pos, w_occurs_neg)
vocab = {}
n = 0
n_pos_count = 0
n_neg_count = 0

#count the words
for doc in token_data:
    c = doc[1]
    n +=1
    if c == 1:
        n_pos_count += 1
    else:
        n_neg_count += 1
    already_counted = []
    for tok in doc[0]:
        if tok not in already_counted:
            if tok not in vocab:
                vocab[tok] = [0,0]
            if c == 1:
                vocab[tok][0] += 1
            else:
                vocab[tok][1] += 1
            already_counted.append(tok)
                
print("Vocab Counted...")

mutual_information_calculations = {}
for w, s in vocab.items():
    #calculate the mutual information for the w
    n11 = s[0]
    n01 = s[1]
    n10 = float(n_pos_count - n11)
    n00 = float(n_neg_count - n01)
    mi = 0
    if n11 > 0:
         mi += ((n11/n)*math.log2((n*n11)/((n11+n10)*(n11+n01))))
    if n01 > 0:
        mi += ((n01/n)*math.log2((n*n01)/((n01+n00)*(n11+n01))))
    if n10 > 0:
        mi += ((n10/n)*math.log2((n*n10)/((n11+n10)*(n10+n00))))  
    if n00 > 0:
        mi += ((n00/n)*math.log2((n*n00)/((n01+n00)*(n10+n00)))) 
    mutual_information_calculations[w] = mi

print("Mutual Information Calculated...\n")

print("Mutual information for 'the': ", mutual_information_calculations["the"])
print("Mutual information for 'like': ", mutual_information_calculations["like"])
print("Mutual information for 'good': ", mutual_information_calculations["good"])
print("Mutual information for 'movie': ", mutual_information_calculations["movie"])
print("\nTop 10 Words:")

#get the top ten vocab
sorted_vocab = sorted(mutual_information_calculations.items(), key=operator.itemgetter(1))
top10words = reversed(sorted_vocab[-10:])
for idx, word in enumerate(top10words):
    print(idx,") ",word[0])

print("Unexpected:")
print("? ", mutual_information_calculations['?'])
print("also ", mutual_information_calculations['also'])
print("both ", mutual_information_calculations['both'])

#print all the mutual information calculations for all words
for idx, word in enumerate(reversed(sorted_vocab)):
    print(idx,") ",word[0], " ", word[1])