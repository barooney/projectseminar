#!/usr/bin/python3

import pandas as pd
import sys
from models import Business, Review

STATE_TO_FILTER = ''
# Choose the state to filter reviews for
if len(sys.argv) >= 2:
	STATE_TO_FILTER = sys.argv[1]
else:
	print("The script requires one parameter, the state:")
	print("")
	print("python3 ./2-separate_labeled_and_unlabeled.py <state>")
	print("for example, python3 ./2-separate_labeled_and_unlabeled.py Illinois")
	exit(0)

print(STATE_TO_FILTER)
df = pd.read_json('./data/intermediate/' + STATE_TO_FILTER + '_reviews.json', lines=True)
labeled = df.query('funny>0 or cool>0 or useful>0')
unlabeled = df.query('funny==0 and cool==0 and useful==0')
print(df.shape)
print(labeled.shape)
print(unlabeled.shape)

labeled.to_json('./data/intermediate/' + STATE_TO_FILTER + '_labeled.json', orient='records', lines=True)
unlabeled.to_json('./data/intermediate/' + STATE_TO_FILTER + '_unlabeled.json', orient='records', lines=True)

