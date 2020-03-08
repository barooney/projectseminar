import collections
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import pandas as pd
import spacy
import nltk
from tqdm import tqdm

df = pd.read_json('./data/intermediate/random-small_reviews.json', lines=True)

annotations = []
for ent in tqdm(df['text']):
	#spacy
	# doc = nlp(ent)
	#nltk
	doc = nltk.word_tokenize(ent)

	annotations.append(doc)
df['annotations'] = annotations

documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(df['annotations'])]
num_docs = len(documents)
train_corpus = documents[:(int(num_docs * 0.8))]
test_corpus = [doc[0] for doc in documents[(int(num_docs * 0.8)):]]
print(train_corpus[:2])
print(test_corpus[:2])
print("num docs: " + str(num_docs))
print("num train docs: " + str(len(train_corpus)))
print("num test docs: " + str(len(test_corpus)))
model = Doc2Vec(vector_size=512, window=2, min_count=1, workers=4)
model.build_vocab(train_corpus)
model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)

# bis hier hin verstanden, rest aus dem Tutorial kopiert:
# https://radimrehurek.com/gensim/auto_examples/tutorials/run_doc2vec_lee.html#sphx-glr-auto-examples-tutorials-run-doc2vec-lee-py

ranks = []
second_ranks = []
for doc_id in range(len(train_corpus)):
    inferred_vector = model.infer_vector(train_corpus[doc_id].words)
    sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
    rank = [docid for docid, sim in sims].index(doc_id)
    ranks.append(rank)

    second_ranks.append(sims[1])

counter = collections.Counter(ranks)
print(counter)

print('Document ({}): «{}»\n'.format(doc_id, ' '.join(train_corpus[doc_id].words)))
print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
for label, index in [('MOST', 0), ('SECOND-MOST', 1), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
    print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(train_corpus[sims[index][0]].words)))