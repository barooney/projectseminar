import sqlite3
import json
from models import Business, Review
from tqdm import tqdm

conn = sqlite3.connect('data/intermediate/database.db')

{"review_id":"Q1sbwvVQXV2734tPgoKj4Q","user_id":"hG7b0MtEbXx5QzbzE6C_VA","business_id":"ujmEBvifdJM6h6RLv4wQIg","stars":1.0,"useful":6,"funny":1,"cool":0,"text":"Total bill for this horrible service? Over $8Gs. These crooks actually had the nerve to charge us $69 for 3 pills. I checked online the pills can be had for 19 cents EACH! Avoid Hospital ERs at all costs.","date":"2013-05-07 04:34:36"}


c = conn.cursor()
c.execute('CREATE TABLE IF NOT EXISTS reviews (review_id text, user_id text, business_id text, stars real, useful real, funny real, cool real, text text, zipf text, date text)')

reviews = dict()
with open('data/yelp/yelp_academic_dataset_review.json', encoding="utf8") as reviews_file:
	for l in tqdm(reviews_file.readlines()):
		r = Review(json.loads(l))
		reviews[r.review_id] = r
		reviews[r.review_id].__dict__['zipf'] = ''


with open('data/intermediate/zipf_all_reviews.json', encoding="utf8") as reviews_file:
	for l in tqdm(reviews_file.readlines()):
		r = Review(json.loads(l))
		reviews[r.review_id].__dict__['zipf'] = r.text

for r in tqdm(reviews):
	c.execute('INSERT INTO reviews VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', [(reviews[r].review_id), (reviews[r].user_id), (reviews[r].business_id), (reviews[r].stars), (reviews[r].useful), (reviews[r].funny), (reviews[r].cool), (reviews[r].text), (reviews[r].zipf), (reviews[r].date)])

conn.commit()
conn.close
