# -*- coding: utf-8 -*-

from nltk import sent_tokenize, word_tokenize, pos_tag
from tqdm import tqdm
import pandas as pd
from matplotlib import pyplot as plt
from collections import Counter

df_basis = pd.read_json('./data/intermediate/Illinois_short_reviews.json', lines=True)
cnt = Counter()
df = pd.DataFrame()
for i, j in df_basis.iterrows():
    if j['funny'] == 0 or j['text'] == "":
        continue
    funny_votes = j['funny']
    text = j['text']
    sentences = sent_tokenize(text)
    votes_per_sentence = funny_votes / len(sentences)
    cnt.update([votes_per_sentence])
    print(str(len(sentences)) + " " + str(votes_per_sentence))
    for sent in sentences:
        print("\t" + str(votes_per_sentence) + ": " + sent)
        j['text'] = sent
        df = df.append(j)
       
cnt = Counter(el for el in cnt.elements() if el >=0.1  and cnt[el] >= 10) 
print(len(df))
print(cnt.most_common(9999))

plt.scatter([key for key, value in cnt.items() if value>1], [value for value in cnt.values() if value>1], marker="x", lw=1)
plt.xscale('log')
# x = [key for key in cnt.keys()]
# y = [value for value in cnt.values()]
# plt.loglog(x, y)

# plt.xlabel('rank')
# plt.ylabel('votes per sentence')

# plt.grid()
# #plt.xticks(indexes + 0.5, plotting_counting.keys(), rotation='vertical')

# plt.show()