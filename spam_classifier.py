import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

# load data
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
data = pd.read_csv(url, sep="\t", header=None, names=["label", "message"])
print(data.head())
print(data["label"].value_counts())

# ham = 0, spam = 1
data["label"] = data["label"].map({"ham": 0, "spam": 1})

# split data
X_train, X_test, y_train, y_test = train_test_split(data["message"], data["label"], test_size=0.2, random_state=42)

# convert text to numbers
cv = CountVectorizer()
X_train = cv.fit_transform(X_train)
X_test = cv.transform(X_test)

# train model
model = MultinomialNB()
model.fit(X_train, y_train)

# check accuracy
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# try some new messages
messages = ["Win a free prize now, call this number", "Are we meeting tomorrow?"]
for m in messages:
    result = model.predict(cv.transform([m]))
    if result[0] == 1:
        print(m, "-> spam")
    else:
        print(m, "-> not spam")
