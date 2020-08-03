# TF-IDF matrix를 SVD로 분해한다.
# C = U.S.VT
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

statements = [
            'ruled india',
            'Chalukyas ruled Badami',
            'So many kingdoms ruled India',
            'Lalbagh is a botanical garden in India'
        ]

# TF-IDF matrix를 생성한다.
tf_vector = TfidfVectorizer(max_features = 8)
tfidf = tf_vector.fit_transform(statements)
print(tfidf.shape)

# SVD (Singular Vector Decomposition)으로 TF-IDF를 분해한다.
# U, S, VT 행렬의 의미 --> Latent Semantic Analysis (LSA)
# U 행렬 ~ 차원 = (문서 개수 X topic 개수) : 문서당 topic 분포
# S 행렬 ~ 차원 = (topic 개수 X topic 개수)
# VT 행렬. 차원 = (topic 개수 X 단어 개수) : topic 당 단어 빈도 (분포)
U, S, VT = np.linalg.svd(tfidf.toarray(), full_matrices = True)

print('U\n',U.round(2), '\n')
print('S\n',S.round (2), '\n')
print('VT\n',VT.round(2), '\n')

# S를 행렬 형태로 변환한다.
s = np.zeros(tfidf.shape)
s[:S.shape[0], :S.shape[0]] = np.diag(S)
print(s.round(2), '\n')
