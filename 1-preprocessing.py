# imports
from collections import Counter
import reverse_geocoder
import itertools
import json
import nltk
import os
import sys
from tqdm import tqdm
from models import Business, Review

# Choose the state to filter reviews for
if len(sys.argv) >= 2:
	STATE_TO_FILTER = sys.argv[1]
else:
	print("The script requires one parameter, the state:")
	print("")
	print("python3 ./1-preprocessing.py <state>")
	print("for example, python3 ./1-preprocessing.py Illinois")
	exit(0)

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