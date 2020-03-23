from nltk import sent_tokenize, word_tokenize, pos_tag
from tqdm import tqdm
import pandas as pd
from matplotlib import pyplot as plt
from collections import Counter
from softmax_logistic_regression import train_modelsoftmax_regression

df_basis = pd.read_json('./data/intermediate/Illinois_reviews.json', lines=True)
cnt = Counter()
df = pd.DataFrame()
for i, j in tqdm(df_basis.iterrows(), total=len(df_basis)):
	if j['funny'] == 0 or j['text'] == "":
		continue
	funny_votes = j['funny']
	text = j['text']
	sentences = sent_tokenize(text)
	votes_per_sentence = funny_votes / len(sentences)
	cnt.update([votes_per_sentence])
	# print(str(len(sentences)) + " " + str(votes_per_sentence))
	for sent in sentences:
		j['text'] = sent
		j['funniness_category'] = votes_per_sentence
		df = df.append(j)
	# 	print("\t" + str(votes_per_sentence) + ": " + sent)

train_modelsoftmax_regression(df, 'no_cond')
train_modelsoftmax_regression(df, 'labeled')



cnt = Counter(el for el in cnt.elements() if el >= 0.1 and cnt[el] >= 10)
# print(len(df))
# s

plt.scatter(cnt.keys(), cnt.values(), marker="x", lw=1)
plt.xscale('log')
plt.show()
