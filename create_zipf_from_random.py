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
from models import Review

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

review_ids = []
with open(data_path + '/intermediate/reviews_to_filter.txt', encoding='utf8') as review_ids_file:
	for l in tqdm(review_ids_file.readlines()):
		review_ids.append(l.strip())

reviews = dict()

# Load all reviews with respect to the given businesses
with open(data_path + '/intermediate/all_reviews_zipf.json', encoding="utf8") as reviews_file:
    for l in tqdm(reviews_file.readlines()):
        r = Review(json.loads(l))
        if r.review_id in review_ids:
            r.text = r.text.lower()
            reviews[r.review_id] = r

print("# Reviews loaded: " + str(len(reviews.values())))

reviews_intermediate_file = open(intermediate_data_path + '/' + "random-small-zipf_-1.0.2.4.9.34.9999999999_reviews.json", 'w')
for r in tqdm(reviews):
    json.dump(reviews[r].__dict__, reviews_intermediate_file)
    reviews_intermediate_file.write("\n")
reviews_intermediate_file.close()