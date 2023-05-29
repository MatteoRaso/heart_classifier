'''
Copyright 2023 Matteo Raso

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

import numpy as np
import pandas as pd
import pickle as pk
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier

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

classifier = GradientBoostingClassifier()
classifier.fit(training_variables, training_target)
accuracy = classifier.score(validation_variables, validation_target)

print("The mean accuracy is " + str(accuracy))

pkl_filename = "classifier.pkl"
with open(pkl_filename, 'wb') as file:
    pk.dump(classifier, file)
