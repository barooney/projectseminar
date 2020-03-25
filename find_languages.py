from langdetect import detect
import pandas as pd
from tqdm import tqdm


#tqdm.pandas()
df = pd.read_json('./data/intermediate/' + 'Illinois_reviews.json', lines=True)


df["language_code"] = df['text'].apply(lambda text_as_string: detect(text_as_string))

print(df.head(100))





