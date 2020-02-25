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
from scipy.interpolate import UnivariateSpline
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
parser.add_argument("state_input", help="Which state to show", type=str)
parser.add_argument("-s", help="Whether to show the graph or save it in the docs folder.", type=bool, default=True)
args = parser.parse_args()
STATE_TO_FILTER = args.state_input
SHOW_GRAPH = args.s

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

for i, x in enumerate(X):
    print(str(i) + "\t" + str(x) + "\t" + str(Y[i]))

# Y_pred = np.polyfit(X, Y, 2)
# p = np.poly1d(Y_pred)
# xp = np.linspace(0, np.max(X), 1000)
# plt.plot(xp, p(xp), 'g', lw=1)

spl = UnivariateSpline(X, Y)
xs = np.linspace(0, np.max(X), 1000)
plt.plot(xs, spl(xs), 'r', lw=1)
# ###

plt.title('Distribution of funny votes in ' + STATE_TO_FILTER)
plt.yscale("log")

if SHOW_GRAPH:
    plt.show()
else:
    try:
        os.mkdir('./doc')
        os.mkdir('./doc/images')
    except FileExistsError:
        pass

    file = open('./doc/images/' + STATE_TO_FILTER + '.png', 'wb')
    plt.savefig(file)
    file.close()