import pandas as pd
from collections import Counter
from tqdm import tqdm
import json
import sys

df_basis = pd.read_json('./data/yelp/yelp_academic_dataset_review.json', lines=True)

# set bins
max_funny = df_basis['funny'].max()
bins =  [-1,0,1,2,5,9999999999999999]
labels=[number for number in range(1,len(bins))]

#### assign bins
df_basis['funniness_category'] = pd.cut(df_basis.funny, bins, labels=labels)

counter_obj = Counter(df_basis['funniness_category'].tolist())
print("before filling: ", counter_obj)
max_reviews_in_highest_bin = counter_obj.most_common(1)[0][1]
print("max reviews in highest bin: " + str(max_reviews_in_highest_bin))

min_bin = 999999999999999

asdf = dict()

for bin_number, num_reviews in tqdm(counter_obj.items()):
	asdf[bin_number] = df_basis.query('funniness_category==@bin_number')
	if df_basis.query('funniness_category==@bin_number').shape[0] < min_bin:
		min_bin = df_basis.query('funniness_category==@bin_number').shape[0]

print(str(min_bin))
print(asdf)

output_file = open('./data/intermediate/random-small.json', 'w')
for e in asdf:
	asdf[e] = asdf[e].sample(frac=1).head(2000)
	for row in asdf[e].iterrows():
		row[1].to_json(output_file)
		output_file.write("\n")
		
