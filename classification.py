import numpy as np
import pandas as pd
import pickle as pk
from sklearn.svm import SVC

rng = np.random.default_rng()
df = pd.read_csv("dataset_heart.csv")
df = df.drop(columns=["fasting blood sugar"])

df = df.to_numpy()
rng.shuffle(df)
variables =df[:, :-1]
target = df[:, -1]

split = int(0.8 * len(target))

training_variables = variables[:split, :]
validation_variables = variables[split:, :]

training_target = target[:split]
validation_target = target[split:]

classifier = SVC()
classifier.fit(training_variables, training_target)
accuracy = classifier.score(validation_variables, validation_target)

print("The mean accuracy is " + str(accuracy))

pkl_filename = "classifier.pkl"
with open(pkl_filename, 'wb') as file:
    pk.dump(classifier, file)