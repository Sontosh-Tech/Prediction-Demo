import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
import pickle
nltk.download('wordnet')

df=pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t')

wl=WordNetLemmatizer()
corpus=[]
for i in range(0,1000):
    review=re.sub('[^a-zA-Z]',' ',df['Review'][i]).lower()
    review=word_tokenize(review)
    review=[wl.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
    review=' '.join(review)
    corpus.append(review)


cv=TfidfVectorizer()
X=cv.fit_transform(corpus).toarray()
y=df.iloc[:,-1].values


classifier=GaussianNB()
classifier.fit(X,y)

pickle.dump(classifier,open('nlp_tfidf.pkl','wb'))