
# standard library
from collections import Counter

# third party modules
import pandas as pd
from nltk.tokenize import TweetTokenizer


# read review file
df = pd.read_json('./data/intermediate/Illinois_reviews.json', lines=True)

# get all words of all texts
list_of_texts = df['text'].tolist()
tknzr = TweetTokenizer()
list_of_lists = [tknzr.tokenize(text) for text in list_of_texts]
words = [word for liste in list_of_lists for word in liste]



def compute_zipf_table(WORDS, sort_parameters=("rank", "ascending"), num_rows=10):
    ' WORDS = list of words'
    zipf_values = [[wort, frequ, rank, frequ*rank] for rank, (wort, frequ) in enumerate(Counter(WORDS).most_common(len(WORDS)), 1)]
    return zipf_values

zipf_table = compute_zipf_table(words)

zipf_df = pd.DataFrame(zipf_table, columns = ['word', 'frequnency', 'rank', 'frequ_rank'])


#print(zipf_df['word'].loc[:500])

#print(zipf_df.loc[0:500,['word']].tolist())


#zipf_df[:500].to_csv(r'500_most_frequent_words.csv',index=False)    

zipf_df_500 = zipf_df[:500]

#print(zipf_df_500.head(10))

list_500_words = zipf_df_500['word'].tolist()

print(list_500_words)

stop_words = ["food", "place", "good", "like", "time",
                 "great", "service", "nice", "order", "best", "restaurant"\
                     "ordered", "much"]