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
parser.add_argument("-s", help="Whether to show the graph or save it in the docs folder.", type=bool, default=False)
args = parser.parse_args()
STATE_TO_FILTER = args.state_input
SHOW_GRAPH = args.s

filename = ''
if STATE_TO_FILTER == 'all':
    filename = './data/yelp/yelp_academic_dataset_review.json'
else:
    filename = './data/intermediate/' + STATE_TO_FILTER + '_reviews.json'

# assert os.path.exists(filename)

print("Generating graph for " + STATE_TO_FILTER)
# df = pd.read_json(filename, lines=True)
# cnt = Counter(df['funny'])

# cnt = dict(sorted(cnt.items(), key = lambda x:x[0] , reverse=False))

# key = cnt.keys()
# df = pd.DataFrame(cnt, index=key).sort_index()
# df.drop(df.columns[1:], inplace=True)

X = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,113,114,115,116,117,118,119,121,122,123,124,125,126,127,128,130,131,132,133,134,135,137,139,140,141,142,143,144,145,146,148,149,151,153,154,155,156,157,158,159,161,162,163,165,166,167,168,169,170,171,172,174,175,176,177,178,179,180,181,182,183,184,185,187,188,190,192,193,201,202,203,208,214,217,222,231,237,239,241,247,249,254,256,259,264,266,267,275,276,277,286,287,290,294,322,332,344,345,356,358,373,377,389,409,412,418,440,460,512,523,534,589,614,625,628,696,703,970,1290]
Y = [5312173,813583,255974,111881,59525,35781,23065,15972,11461,8387,6409,4765,3682,3029,2437,2014,1757,1407,1266,1067,939,817,676,659,573,516,460,404,414,319,334,265,270,247,213,196,181,168,156,158,121,132,112,104,94,94,96,100,83,48,59,66,57,52,49,48,39,31,39,30,24,36,35,31,24,27,23,17,20,24,18,12,15,21,15,12,13,14,21,14,11,15,6,9,8,6,8,6,11,5,8,7,7,6,9,8,6,5,6,3,2,4,6,3,4,4,5,7,6,7,1,3,4,3,1,1,2,2,3,2,5,4,4,2,4,3,1,1,7,1,3,1,3,2,1,2,4,3,2,3,7,2,4,2,1,1,2,1,4,4,4,2,2,1,3,2,2,3,6,8,2,4,2,3,1,1,2,2,3,1,3,1,2,1,2,1,1,1,1,1,1,1,3,1,1,1,1,1,1,1,1,1,1,1,4,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,1,1,1,1,1,1,1,1,1,2,1,1,1,1,1,1,1,1,1,1,1]
cnt = Counter(dict(zip(X, Y)))

print(cnt.most_common())

plt.scatter(cnt.keys(), cnt.values(), s=10)

# linreg
# X = np.array(list(cnt.keys()))#.reshape(-1, 1)  # values converts it into a numpy array
# Y = np.array(list(cnt.values()))#.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column

print(X)
print(Y)

# for i, x in enumerate(X):
#     print(str(i) + "\t" + str(x) + "\t" + str(Y[i]))

# Y_pred = np.polyfit(X, Y, 2)
# p = np.poly1d(Y_pred)
# xp = np.linspace(0, np.max(X), 1000)
# plt.plot(xp, p(xp), 'g', lw=1)

# spl = UnivariateSpline(X, Y)
# xs = np.linspace(0, np.max(X), 1000)
# plt.plot(xs, spl(xs), 'r', lw=1)
# ###

title = ''
if STATE_TO_FILTER == 'all':
    title = 'Distribution of funny votes'
else:
    title = 'Distribution of funny votes in ' + STATE_TO_FILTER
plt.title(title)
plt.yscale("log")
plt.xlabel("number of funny votes")
plt.ylabel("number of reviews")

print(SHOW_GRAPH)
if int(SHOW_GRAPH):
    plt.show()
else:
    try:
        os.mkdir('./doc')
        os.mkdir('./doc/images')
    except FileExistsError:
        pass

    filename = './doc/images/' + STATE_TO_FILTER
    plt.savefig(filename + '.png', format="png")
    plt.savefig(filename + '.pdf', format="pdf")