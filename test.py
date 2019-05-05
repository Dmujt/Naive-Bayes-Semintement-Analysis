from model import SentimentNaiveBayes, format_data, FOLD1_DATA, FOLD2_DATA, FOLD3_DATA, POS_FILES, NEG_FILES
import sys

#place corresponding data the folds
format_data(POS_FILES, 1)
format_data(NEG_FILES, -1)

print("PREPARED FOLD 1 DOCS: " + str(len(FOLD1_DATA)))
print("PREPARED FOLD 2 DOCS: " + str(len(FOLD2_DATA)))
print("PREPARED FOLD 3 DOCS: " + str(len(FOLD3_DATA)))

#the training data to use
testing_data = []

#make sure the correct number of params were given
if len(sys.argv) != 2:
	print("Need 1 fold to train the model (fold1, fold2, or fold3)")
else:
	if "fold1" in sys.argv:
		testing_data = FOLD1_DATA
	elif "fold2" in sys.argv:
		testing_data = FOLD2_DATA 
	elif "fold3" in sys.argv:
		testing_data = FOLD3_DATA 
	else:
		print("Invalid training fold configuration given")

	#import the model and test the accuracy
	if len(testing_data) > 0:
		model = SentimentNaiveBayes()
		model.import_params()
		print("The accuracy is ", (round(model.accuracy(testing_data), 3)*100), "%")
	else:
		print("Test data not found - cannot determine accuracy")
