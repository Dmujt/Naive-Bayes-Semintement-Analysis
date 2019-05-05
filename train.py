from model import SentimentNaiveBayes,format_data,  FOLD1_DATA, FOLD2_DATA, FOLD3_DATA, POS_FILES, NEG_FILES
import sys

#place corresponding data the folds
format_data(POS_FILES, 1)
format_data(NEG_FILES, -1)

print("PREPARED FOLD 1 DOCS: " + str(len(FOLD1_DATA)))
print("PREPARED FOLD 2 DOCS: " + str(len(FOLD2_DATA)))
print("PREPARED FOLD 3 DOCS: " + str(len(FOLD3_DATA)))

#the training data to use
training_data = []

#make sure the correct number of folds were assigned
if len(sys.argv) != 3:
	print("Need 2 folds to train the model (fold1, fold2, and/or fold3)")
else:
	if "fold1" in sys.argv and "fold2" in sys.argv:
		training_data = FOLD1_DATA + FOLD2_DATA
	elif "fold2" in sys.argv and "fold3" in sys.argv:
		training_data = FOLD2_DATA + FOLD3_DATA
	elif "fold3" in sys.argv and "fold1" in sys.argv:
		training_data = FOLD3_DATA + FOLD1_DATA
	else:
		print("Invalid training fold configuration given (only provide 2)")

	#train the model
	if len(training_data) > 0:
		model = SentimentNaiveBayes()
		model.train(training_data)
	else: 
		print("Data for folds not found - cannot form and train model")
