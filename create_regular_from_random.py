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

# Load all reviews with respect to the given businesses
reviews_intermediate_file = open(data_path + '/intermediate/new_random_small_unzipfed.json', 'w')
with open(data_path + '/yelp/yelp_academic_dataset_review.json', encoding="utf8") as reviews_file:
	for l in tqdm(reviews_file.readlines(), total=6685900):
		review_id = l[15:37]
		if review_id in review_ids:
			tqdm.write(review_id)
			review_ids.remove(review_id)
			reviews_intermediate_file.write(l)

reviews_intermediate_file.close()