import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Set pandas to display all of the dataframe's columns without wrapping (in the console)
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

# Saved in the same directory as the script
path = "Youtube05-Shakira.csv"

# Load the data
shakira_df = pd.read_csv(path)

# Data exploration
print("\n----------First 3 rows of Youtube05-Shakira.csv----------\n", shakira_df.head(3))

print("\n----------Shape of the dataframe----------\n", shakira_df.shape)

print("\n----------Dataframe info----------") 
shakira_df.info(verbose=True)

print("\n----------Unique values of CLASS column----------\n", shakira_df["CLASS"].unique())

# Columns of relevance for this project are CONTENT and CLASS
# Drop the other columns
shakira_df = shakira_df.drop(columns=["COMMENT_ID", "AUTHOR", "DATE"])

# Shuffle the dataset
# random_state parameter allows the split to be reproduced (similar to how seed allows random numbers to be reproduced)
shakira_df = shakira_df.sample(frac=1, random_state=42)

# Split the data 75/25
shakira_df_train = shakira_df.sample(frac=0.75, random_state=42)
shakira_df_test = shakira_df.drop(shakira_df_train.index)

# #The two features of interest for this exercise are CONTENT and CLASS
x_train = shakira_df_train['CONTENT']
y_train = shakira_df_train['CLASS']

# Count vectorize x_train (the CONTENT feature)
count_vectorizer = CountVectorizer()
x_train_cv = count_vectorizer.fit_transform(x_train)

print("\n----------Shape of vectorized feature----------\n", x_train_cv.shape)

# tfidf transform x_train_cv (the CONTENT feature)
tfidf = TfidfTransformer()
x_train_tfidf = tfidf.fit_transform(x_train_cv)

print("\n----------Shape of tfidf-transformed feature----------\n", x_train_tfidf.shape)

# Fit the NB classifier with the count-vectorized, tfidf-transformed CONTENT (training_data_tfidf) and CLASS (y_train) features
nb_classifier = MultinomialNB().fit(x_train_tfidf, y_train)

# Carry out 5-fold cross validation and determine the accuracy scores
cvs = cross_val_score(nb_classifier, x_train_tfidf, y_train, scoring='accuracy', cv=5)
print ("\n----------Accuracy of the 5 runs----------\n", cvs)
print ("\n----------AVERAGE accuracy of the 5 runs----------\n", cvs.mean())

# Set up x_test and y_test
x_test = shakira_df_test['CONTENT']
y_test = shakira_df_test['CLASS']

# Count vectorize and tfidf transform x_test (CONTENT)
x_test_cv = count_vectorizer.transform(x_test)
x_test_tfidf = tfidf.transform(x_test_cv)

# Test nb_classifier model with transformed test data
y_prediction = nb_classifier.predict(x_test_tfidf)

# Calculate confusion matrix values
cm = confusion_matrix(y_test, y_prediction)

# Plot the confusion matrix
cm_visual = ConfusionMatrixDisplay(cm, display_labels=['Not Spam','Spam'])
cm_visual.plot()

# Create new comments for the model to process
new_comments = [
    'Shakira has the best voice', 
    'i keep returning to this. this is the best song!',
    'Love this song',
    'SHAKIRA IS THE QUEEN!!!!',
    'Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.',
    'Join FakeCrypto.com and earn $4000 a month!!'
]

# Count vectorize and tfidf transform new_comments
new_comments_cv = count_vectorizer.transform(new_comments)
new_comments_tfidf = tfidf.transform(new_comments_cv)

# Classify the new comments with the model
# After execution, new_comments_prediction is an array of 1s and 0s (in place of the comments)
# 1 corresponds to spam
# 0 corresponds to not spam
new_comments_prediction = nb_classifier.predict(new_comments_tfidf)

print("\n----------Classify new comments----------")

# zip() allows the loop to iterate through 2 lists (new_comments and new_comments_prediction)
for comment, prediction in zip(new_comments, new_comments_prediction):
    if (prediction == 0):
        comment_prediction = "Not spam"
    elif (prediction == 1):
        comment_prediction = "Spam"
    
    print("Comment: ", comment, "\nPrediction: ", comment_prediction, "\n") 