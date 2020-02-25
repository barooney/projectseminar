# imports

# stand library
from collections import Counter
import os
import sys
import itertools
import argparse

# third party modules
import reverse_geocoder
import json
import nltk
from tqdm import tqdm

# application specific
from models import Business, Review

# Choose the state to filter reviews for
parser = argparse.ArgumentParser(description="The script filters reviews for a given state. \
                                 One parameter is required, the state: python3 ./2-doc-stats.py <state>. \
                                     For example, python3 ./2-doc-stats.py Illinois")
parser.add_argument("state_input", help="Enter the state to filter reviews for: for example, python3 ./1-doc-stats.py Illinois",
                    type=str)
args = parser.parse_args()
STATE_TO_FILTER = args.state_input

# DEFAULTS

# define folder structure
base_path = os.getcwd()
data_path = base_path + '/data'
intermediate_data_path = data_path + '/intermediate'

try:
    os.mkdir(data_path)
    print("Folder created.")
except FileExistsError:
    print("Folder already exists.")

try:
    os.mkdir(intermediate_data_path)
    print("Folder created.")
except FileExistsError:
    print("Folder already exists.")

businesses = dict()
reviews = dict()

# import businesses
with open(data_path + '/yelp/yelp_academic_dataset_business.json', encoding="utf8") as businesses_file:
    for l in tqdm(businesses_file.readlines()):
        b = Business(json.loads(l))
        businesses[b.business_id] = b

print("# of Businesses: " + str(len(businesses)))

# Filter businesses by state.

states = dict()
def add_or_update(state, business):
    if state in states:
        states[state].add(business)
    else:
        states[state] = set([business])

business_list = list(businesses.values())

# Find coordinates by using the reverse_geocoder
coordinates = [(c.latitude, c.longitude) for c in business_list]
res = reverse_geocoder.search(coordinates)
ctr = 0
possible_states = set()
for r in res:
    state = r['admin1']
    possible_states.add(state)
    if state == STATE_TO_FILTER:
        add_or_update(state, business_list[ctr])
    ctr += 1
print(possible_states)
for s in states:
    print(s + ": " + str(len(states[s])))

	# List all businesses of the given states
business_ids = set()
for b in states[STATE_TO_FILTER]:
    business_ids.add(b.business_id)

# Get the number of businesses to look for reviews for
print("# Businesses to be reviewed: " + str(len(business_ids)))

# Load all reviews with respect to the given businesses
with open(data_path + '/yelp/yelp_academic_dataset_review.json', encoding="utf8") as reviews_file:
    for l in tqdm(reviews_file.readlines()):
        r = Review(json.loads(l))
        if r.business_id in business_ids:
            r.text = r.text.lower()
            reviews[r.review_id] = r

print("# Reviews loaded: " + str(len(reviews.values())))

businesses_intermediate_file = open(intermediate_data_path + '/' + STATE_TO_FILTER + '_businesses.json', 'w')
for b in tqdm(businesses):
    if businesses[b].business_id in business_ids:
        json.dump(businesses[b].__dict__, businesses_intermediate_file)
        businesses_intermediate_file.write("\n")
businesses_intermediate_file.close()

reviews_intermediate_file = open(intermediate_data_path + '/' + STATE_TO_FILTER + '_reviews.json', 'w')
for r in tqdm(reviews):
    json.dump(reviews[r].__dict__, reviews_intermediate_file)
    reviews_intermediate_file.write("\n")
reviews_intermediate_file.close()