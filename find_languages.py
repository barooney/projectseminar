from langdetect import detect
import pandas as pd
from tqdm import tqdm
from collections import Counter
import numpy as np

tqdm.pandas()
# raed all reviews with stop words for language detection
df = pd.read_json('./data/yelp/' + 'yelp_academic_dataset_review.json', lines=True)
#df = pd.read_json('./data/intermediate/' + 'Illinois_reviews.json', lines=True)

def find_lang(text):
    try:
        return detect(text)
    except:
        return "-no-lang-"

df["language_code"] = df['text'].progress_apply(lambda text_as_string: find_lang(text_as_string))


list_lang_codes = df["language_code"].tolist()

new_cnt = Counter(list_lang_codes)

dict_rel_freq = {lang_code:int(freq)/len(list_lang_codes) for lang_code, freq in new_cnt.items()}

print(dict_rel_freq)

#print(df.query('language_code != "en"'))





