import pandas as pd
import numpy as np
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
import pickle

df=pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t')

ps=PorterStemmer()
corpus=[]
for i in range(0,1000):
    review=re.sub('[^a-zA-Z]',' ',df['Review'][i]).lower()
    review=word_tokenize(review)
    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review=' '.join(review)
    corpus.append(review)


cv=CountVectorizer(max_features=1500)
X=cv.fit_transform(corpus).toarray()
y=df.iloc[:,-1].values


classifier=GaussianNB()
classifier.fit(X,y)

pickle.dump(classifier,open('nlp_bow.pkl','wb'))