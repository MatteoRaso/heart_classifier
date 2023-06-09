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

import pandas as pd
import scipy.stats as st

df = pd.read_csv("dataset_heart.csv")
# Error in the dataset, space accidently made part of the name.
group1 = df[df["sex "] == 0]
group2 = df[df["sex "] == 1]

stat, p = st.ttest_ind(group1["heart disease"], group2["heart disease"])

if p < 0.05:
    print("Sex is an important variable.")

else:
    print("Sex is insignificant.")

group1 = df[df["fasting blood sugar"] == 0]
group2 = df[df["fasting blood sugar"] == 1]

stat, p = st.ttest_ind(group1["heart disease"], group2["heart disease"])

if p < 0.05:
    print("Fasting blood sugar is an important variable.")

else:
    print("Fasting blood sugar is insignificant.")

group1 = df[df["resting electrocardiographic results"] == 0]
group2 = df[df["resting electrocardiographic results"] == 2]

stat, p = st.ttest_ind(group1["heart disease"], group2["heart disease"])

if p < 0.05:
    print("Resting electrocardiographic results are an important variable.")

else:
    print("Resting electrocardiographic results are insignificant.")

group1 = df[df["exercise induced angina"] == 0]
group2 = df[df["exercise induced angina"] == 1]

stat, p = st.ttest_ind(group1["heart disease"], group2["heart disease"])

if p < 0.05:
    print("Exercise induced angina is an important variable.")

else:
    print("Exercise induced angina is insignificant.")
