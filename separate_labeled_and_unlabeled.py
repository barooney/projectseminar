import pandas as pd

df = pd.read_json('./data/intermediate/Illinois_reviews.json', lines=True)
labeled = df.query('funny>0 or cool>0 or useful>0')
unlabeled = df.query('funny==0 and cool==0 and useful==0')
print(df.shape)
print(labeled.shape)
print(unlabeled.shape)

labeled.to_json('./data/intermediate/Illinois_labeled.json', orient='records', lines=True)
unlabeled.to_json('./data/intermediate/Illinois_unlabeled.json', orient='records', lines=True)