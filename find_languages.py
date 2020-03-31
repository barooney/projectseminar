from langdetect import detect
import pandas as pd
from tqdm import tqdm
from collections import Counter
import numpy as np

tqdm.pandas()
# raed all reviews with stop words for language detection
df = pd.read_json('./data/intermediate/random-small-unzipfed-5000_-1.0.2.4.9.34.9999999999_reviews.json', lines=True)
#df = pd.read_json('./data/intermediate/' + 'Illinois_reviews.json', lines=True)

def find_lang(text):
    try:
        lang = detect(text)
        if lang == 'ca':
            print(text)
        return lang
    except:
        return "-no-lang-"

df["language_code"] = df['text'].progress_apply(lambda text_as_string: find_lang(text_as_string))


list_lang_codes = df["language_code"].tolist()

new_cnt = Counter(list_lang_codes)
print(new_cnt)

dict_rel_freq = {lang_code:int(freq)/len(list_lang_codes) for lang_code, freq in new_cnt.items()}

print(dict_rel_freq)

#print(df.query('language_code != "en"'))





