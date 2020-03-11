import pandas as pd
from collections import Counter
from tqdm import tqdm
import json
import sys
import argparse



def create_test_sample(bins):
   
    
    # read file with all reviews from which stop words were already removed
    df_basis = pd.read_json('./data/intermediate/zipf_all_reviews.json', lines=True)
    
    
    # set bins
    max_funny = df_basis['funny'].max()
    #bins =  [-1,1,3,5,10,100,9999999999999999]
    labels=[number for number in range(1,len(bins))]
    
    #### assign bins
    df_basis['funniness_category'] = pd.cut(df_basis.funny, bins, labels=labels)
    
    counter_obj = Counter(df_basis['funniness_category'].tolist())
    print("before filling: ", counter_obj)
    max_reviews_in_highest_bin = counter_obj.most_common(1)[0][1]
    print("max reviews in highest bin: " + str(max_reviews_in_highest_bin))
    
    min_bin = 999999999999999
    
    bin_separations = dict()
    
    for bin_number, num_reviews in tqdm(counter_obj.items()):
    	bin_separations[bin_number] = df_basis.query('funniness_category==@bin_number')
    	if df_basis.query('funniness_category==@bin_number').shape[0] < min_bin:
    		min_bin = df_basis.query('funniness_category==@bin_number').shape[0]
    
    print(str(min_bin))
    print(bin_separations)
    
    bin_names_as_strings = map(lambda x: str(x), bins)
    bins_as_file_name = ".".join(bin_names_as_strings)
    
    output_file = open('./data/intermediate/random-small_' + bins_as_file_name + '_reviews.json', 'w')
    for bin in bin_separations:
    	bin_separations[bin] = bin_separations[bin].sample(frac=1).head(2000)
    	for row in bin_separations[bin].iterrows():
    		row[1].to_json(output_file)
    		output_file.write("\n")
		
