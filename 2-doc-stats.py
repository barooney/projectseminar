# imports
from collections import Counter
import reverse_geocoder
import itertools
import json
import nltk
import numpy as np
from operator import itemgetter
import os
import pandas as pd
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression
import sys
from tqdm import tqdm
from models import Business, Review
from matplotlib import pyplot as plt

# Choose the state to filter reviews for
if len(sys.argv) >= 2:
	STATE_TO_FILTER = sys.argv[1]
else:
	print("The script requires one parameter, the state:")
	print("")
	print("python3 ./2-doc-stats.py <state>")
	print("for example, python3 ./2-doc-stats.py Illinois")
	exit(0)

df = pd.read_json('./data/intermediate/' + STATE_TO_FILTER + '_reviews.json', lines=True)
cnt = Counter(df['funny'])

cnt = dict(sorted(cnt.items(), key = lambda x:x[0] , reverse=False))

key = cnt.keys()
df = pd.DataFrame(cnt, index=key).sort_index()
df.drop(df.columns[1:], inplace=True)

print(df)

plt.scatter(cnt.keys(), cnt.values())

###
X = np.array(list(cnt.keys())).reshape(-1, 1)  # values converts it into a numpy array
print(str(X))
Y = np.array(list(cnt.values())).reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
print(str(Y))
linear_regressor = LinearRegression()  # create object for the class
linear_regressor.fit(X, Y)  # perform linear regression
Y_pred = linear_regressor.predict(X)  # make predictions
plt.plot(X, Y_pred, color='red')
###

# plt.xlim((0,100))
# plt.ylim((0,35000))
#plt.xscale("log")
plt.title('Distribution of funny votes in ' + STATE_TO_FILTER)
plt.yscale("log")
plt.show()