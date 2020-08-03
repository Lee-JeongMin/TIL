# Latent Dirichlet Allocation (LDA)
# ---------------------------------
import numpy as np
import re
import pickle
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA

from sklearn.datasets import fetch_20newsgroups

# news data를 읽어와서 저장해 둔다.
newsData = fetch_20newsgroups(shuffle=True, 
                             random_state=1, 
                             remove=('headers', 'footers', 'quotes'))

with open('./dataset/news.data', 'wb') as f:
   pickle.dump(newsData , f, pickle.HIGHEST_PROTOCOL)

# 저장된 news data를 읽어온다.
with open('./dataset/news.data', 'rb') as f:
    newsData  = pickle.load(f)

# 첫 번째 news를 조회해 본다.
news = newsData.data
print(len(news))
print(news[0])

# news 별로 분류된 target을 확인해 본다.
print(newsData.target_names)
print(len(newsData.target_names))

# preprocessing.
# 1. 영문자가 아닌 문자를 모두 제거한다.
news1 = []
for doc in news:
    news1.append(re.sub("[^a-zA-Z]", " ", doc))

# 2. 불용어를 제거하고, 모든 단어를 소문자로 변환하고, 길이가 3 이하인 
# 단어를 제거한다
stop_words = stopwords.words('english')
news2 = []
for doc in news1:
    doc1 = []
    for w in doc.split():
        w = w.lower()
        if len(w) > 3 and w not in stop_words:
            doc1.append(w)
    news2.append(' '.join(doc1))
    
print(news2[0])

# TF-IDF matrix를 생성한다.
tf_vector = TfidfVectorizer(max_features = 700)
tfidf = tf_vector.fit_transform(news2)
print(tfidf.shape)

vocab = tf_vector.get_feature_names()
print(vocab[:20])

# Latent Dirichlet Allocation (LDA)
# ---------------------------------
# Return 값이 Document-Topic distribution이다.
# iteration 횟수가 max_iter까지 가면 아직 수렴하지 않은 것이다.
# 아직 수렴하지 않은 경우 mat_iter를 증가시켜야 한다.
# mat_iter를 증가시켜도 수렴하지 못하는 경우는 preprocessing 등을 좀 더 정밀하게 해야 한다.
# perplexity는 감소해야 한다.
model = LDA(n_components = 20, 
            learning_method='online', 
            evaluate_every=5, 
            max_iter=1000, 
            verbose=1)

doc_topic = model.fit_transform(tfidf)

# 문서 별 Topic 번호를 확인한다. (문서 10개만 확인)
for i in range(10):
    print('문서-{:d} : topic = {:d}'.format(i, np.argmax(doc_topic[i:(i+1), :][0])))

# topic_term 행렬에서 topic 별로 중요 단어를 표시한다
topic_term = model.components_
for i in range(len(topic_term)):
    idx = np.flipud(topic_term[i].argsort())[:10]
    print('토픽-{:2d} : '.format(i+1), end='')
    for n in idx:
        print('{:s} '.format(vocab[n]), end='')
    print()

newsData.keys()

# 문서별로 분류된 코드를 확인해 본다.
# x, y : 문서 번호
def checkTopic(x, y):
    print("문서 %d의 topic = %s" % (x, newsData.target_names[newsData.target[x]]))
    print("문서 %d의 topic = %s" % (y, newsData.target_names[newsData.target[y]]))

checkTopic(1, 5)
checkTopic(0, 2)
checkTopic(4, 7)
