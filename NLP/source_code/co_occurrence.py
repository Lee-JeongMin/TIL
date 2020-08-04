# co-occurrence encoding 연습
# ---------------------------
from sklearn.feature_extraction.text import CountVectorizer

docs = ['성진과 창욱은 야구장에 갔다',
        '성진과 태균은 도서관에 갔다',
        '성진과 창욱은 공부를 좋아한다']

count_model = CountVectorizer(ngram_range=(1,1))
x = count_model.fit_transform(docs)

# 문서에 사용된 사전을 조회한다.
print(count_model.vocabulary_)

# co-occurrence 행렬을 조회한다. Compact Sparse Row(CSR) format
# (row, col) value
print(x)

# 행렬 형태로 표시한다.
print(x.toarray())
print()
print(x.T.toarray())

#x.T의 의미
#          1 2 3  - 문장
#갔다    [[1 1 0] - '갔다'라는 단어는 문장-1과 문장-2에 쓰였음.
#공부를   [0 0 1] - '공부를'은 문장-3에만 쓰였음.
#도서관에 [0 1 0]
#성진과   [1 1 1]
#야구장에 [1 0 0]
#좋아한다 [0 0 1]
#창욱은   [1 0 1]
#태균은   [0 1 0]]

xc = x.T * x # this is co-occurrence matrix in sparse csr format
xc.setdiag(0) # sometimes you want to fill same word cooccurence to 0
print(xc.toarray())

#              0       1       2        3        4         5        6       7
#             갔다  공부를  도서관에  성진과  야구장에  좋아한다  창욱은  태균은
#0 갔다        0       0       1        2        1         0        1       1
#1 공부를      0       0       0        1        0         1        1       0
#2 도서관에    1       0       0        1        0         0        0       1
#3 성진과      2       1       1        0        1         1        2       1
#4 야구장에    1       0       0        1        0         0        1       0
#5 좋아한다    0       1       0        1        0         0        1       0
#6 창욱은      1       1       0        2        1         1        0       0
#7 태균은      1       0       1        1        0         0        0       0

# ngram_range(min_n = 1, max_n = 2)인 경우
#count_model = CountVectorizer(ngram_range=(1,2))
#x = count_model.fit_transform(docs)

# 문서에 사용된 사전을 조회한다.
#print(count_model.vocabulary_)

xc = x.T * x # this is co-occurrence matrix in sparse csr format
xc.setdiag(0) # sometimes you want to fill same word cooccurence to 0
print(xc.toarray())

# Co-occurrence matrix를 SVD로 분해한다.
# C = U.S.VT
# numpy를 이용한 SVD 예시
import numpy as np
C = xc.toarray()
U, S, VT = np.linalg.svd(C, full_matrices = True)
print(np.round(U, 2), '\n')
print(np.round(S, 2), '\n')
print(np.round(VT, 2), '\n')

# S를 정방행렬로 바꾼다.
s = np.diag(S)
print(np.round(s, 2))

# A = U.s.VT를 계산하고, A와 C가 일치하는지 확인한다.
A = np.dot(U, np.dot(s, VT))
print(np.round(A, 1))
print(C)

# sklearn을 이용한 SVD 예시
from sklearn.decomposition import TruncatedSVD

# 특이값 (S)이 큰 4개를 주 성분으로 C의 차원을 축소한다.
svd = TruncatedSVD(n_components=4, n_iter=7)
D = svd.fit_transform(xc.toarray())

U = D / svd.singular_values_
S = np.diag(svd.singular_values_)
VT = svd.components_

print("\nU, S, VT :")
print(np.round(U, 2), '\n')
print(np.round(S, 2), '\n')
print(np.round(VT, 2), '\n')

print("C를 4개 차원으로 축소 : truncated (U * S)")
print(np.round(D, 2))

# U * S * VT 하면 원래 C의 차원과 동일해 진다. U * S가 축소된
# 차원을 의미하고, V는 축소된 차원을 원래 차원으로 되돌리는 역할을
# 한다 (mapping back)
