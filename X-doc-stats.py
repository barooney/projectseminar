# imports

# standard library
from collections import Counter
from operator import itemgetter
import itertools
import sys
import os
import json
import argparse
import os.path
import math

# third party modules
import reverse_geocoder
import nltk
import numpy as np
import numpy.polynomial.polynomial as poly
import pandas as pd
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures 
from tqdm import tqdm
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy import optimize

# application specific modules
from models import Business, Review


# Choose the state to filter reviews for
parser = argparse.ArgumentParser(description="The script generates stats for a given state. Make sure the state file has been created before. \
                                 One parameter is required, the state: python3 ./2-doc-stats.py <state>. \
                                     For example, python3 ./2-doc-stats.py Illinois")
parser.add_argument("state_input", help="Enter the state to generate stats for: for example, python3 ./2-doc-stats.py Illinois",
                    type=str)
args = parser.parse_args()
STATE_TO_FILTER = args.state_input

assert os.path.exists('./data/intermediate/' + STATE_TO_FILTER + '_reviews.json')


print("Generating graph for " + STATE_TO_FILTER)
df = pd.read_json('./data/intermediate/' + STATE_TO_FILTER + '_reviews.json', lines=True)
cnt = Counter(df['funny'])

cnt = dict(sorted(cnt.items(), key = lambda x:x[0] , reverse=False))

key = cnt.keys()
df = pd.DataFrame(cnt, index=key).sort_index()
df.drop(df.columns[1:], inplace=True)

plt.scatter(cnt.keys(), cnt.values())

# linreg
X = np.array(list(cnt.keys()))#.reshape(-1, 1)  # values converts it into a numpy array
Y = np.array(list(cnt.values()))#.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
print(X)
print(Y)

#p = poly.Polynomial.fit(X,Y,25)
#plt.plot(*p.linspace())

#mymodel = np.poly1d(np.polyfit(X, Y, 3))

#plt.plot(mymodel.linespace())

#Y_pred = np.polynomial.polynomial.polyfit(X, Y, deg=2)


def func(x, a, b):
    #return len(list(cnt.values())) / (a*((x-2)**2)+1)
    return math.e**(a*(-x)+b)
    #return x**(-a)

popt, pcov = curve_fit(func, X, Y)

params, params_covariance = optimize.curve_fit(func, X, Y)


plt.plot(X, func(X, params[0], params[1]),
         label='Fitted function')



# Y_pred = np.polyfit(X, Y, deg=3)

# print(Y_pred)
# Y_pred_form = ((Y_pred[3]) * X**3) +((Y_pred[2]) * X**2) + ((Y_pred[1]) * X) + (Y_pred[0])
# plt.plot(X, Y_pred_form)
# ###

plt.title('Distribution of funny votes in ' + STATE_TO_FILTER)
plt.yscale("log")
plt.show()

# try:
# 	os.mkdir('./doc')
# 	os.mkdir('./doc/images')
# except FileExistsError:
#     pass

# file = open('./doc/images/' + STATE_TO_FILTER + '.png', 'wb')
# plt.savefig(file)
# file.close()