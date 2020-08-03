# Skipgram 예시
# 소설 alice in wonderland에 사용된 단어들을 2차원 feature로 vector화 한다.
# -----------------------------------------------------------------------
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from nltk import pos_tag
from nltk.stem import PorterStemmer
import collections
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model

# 전처리
def preprocessing(text):
    text2 = "".join([" " if ch in string.punctuation else ch.lower() for ch in text])
    tokens = nltk.word_tokenize(text2)

    stopwds = stopwords.words('english')
    tokens = [token for token in tokens if token not in stopwds and len(word)>=3]

    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]

    tagged_corpus = pos_tag(tokens)    
    
    Noun_tags = ['NN','NNP','NNPS','NNS']
    Verb_tags = ['VB','VBD','VBG','VBN','VBP','VBZ']

    lemmatizer = WordNetLemmatizer()

    def prat_lemmatize(token,tag):
        if tag in Noun_tags:
            return lemmatizer.lemmatize(token,'n')
        elif tag in Verb_tags:
            return lemmatizer.lemmatize(token,'v')
        else:
            return lemmatizer.lemmatize(token,'n')
    
    pre_proc_text =  " ".join([prat_lemmatize(token,tag) for token,tag in tagged_corpus])             

    return pre_proc_text

# 소설 alice in wonderland를 읽어온다.
lines = []
fin = open("./dataset/alice_in_wonderland.txt", "r")
for line in fin:
    if len(line) == 0:
        continue
    lines.append(preprocessing(line))
fin.close()

# 단어들이 사용된 횟수를 카운트 한다.
counter = collections.Counter()

for line in lines:
    for word in nltk.word_tokenize(line):
        counter[word.lower()] += 1

# 사전을 구축한다.
# 가장 많이 사용된 단어를 1번으로 시작해서 번호를 부여한다.
word2idx = {w:(i+1) for i,(w,_) in enumerate(counter.most_common())}
idx2word = {v:k for k,v in word2idx.items()}

# Trigram으로 학습 데이터를 생성한다.
xs = []     # 입력 데이터
ys = []     # 출력 데이터
for line in lines:
    # 사전에 부여된 번호로 단어들을 표시한다.
    embedding = [word2idx[w.lower()] for w in nltk.word_tokenize(line)]
    
    # Trigram으로 주변 단어들을 묶는다.
    triples = list(nltk.trigrams(embedding))
    
    # 왼쪽 단어, 중간 단어, 오른쪽 단어로 분리한다.
    w_lefts = [x[0] for x in triples]
    w_centers = [x[1] for x in triples]
    w_rights = [x[2] for x in triples]
    
    # 입력 (xs)      출력 (xy)
    # ---------    -----------
    # 중간 단어 --> 왼쪽 단어
    # 중간 단어 --> 오른쪽 단어
    xs.extend(w_centers)
    ys.extend(w_lefts)
    xs.extend(w_centers)
    ys.extend(w_rights)

# 학습 데이터를 one-hot 형태로 바꾸고, 학습용과 시험용으로 분리한다.
vocab_size = len(word2idx) + 1  # 사전의 크기

ohe = OneHotEncoder(categories = [range(vocab_size)])
X = ohe.fit_transform(np.array(xs).reshape(-1, 1)).todense()
Y = ohe.fit_transform(np.array(ys).reshape(-1, 1)).todense()
Xtrain, Xtest, Ytrain, Ytest, xstr, xsts = train_test_split(X, Y, xs, test_size=0.2)
print(Xtrain.shape, Xtest.shape, Ytrain.shape, Ytest.shape)

# 딥러닝 모델을 생성한다.
BATCH_SIZE = 128
NUM_EPOCHS = 100

input_layer = Input(shape = (Xtrain.shape[1],), name="input")
first_layer = Dense(300, activation='relu',name = "first")(input_layer)
first_dropout = Dropout(0.5, name="firstdout")(first_layer)
second_layer = Dense(2, activation='relu', name="second")(first_dropout)
third_layer = Dense(300,activation='relu', name="third")(second_layer)
third_dropout = Dropout(0.5,name="thirdout")(third_layer)
fourth_layer = Dense(Ytrain.shape[1], activation='softmax', name = "fourth")(third_dropout)

model = Model(input_layer, fourth_layer)
model.compile(optimizer = "rmsprop", loss="categorical_crossentropy")

# 학습
hist = model.fit(Xtrain, Ytrain, 
                 batch_size=BATCH_SIZE,
                 epochs=NUM_EPOCHS,
                 validation_data = (Xtest, Ytest))

# Loss history를 그린다
plt.plot(hist.history['loss'], label='Train loss')
plt.plot(hist.history['val_loss'], label = 'Test loss')
plt.legend()
plt.title("Loss history")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()

# Extracting Encoder section of the Model for prediction of latent variables
encoder = Model(input_layer, second_layer)

# Predicting latent variables with extracted Encoder model
reduced_X = encoder.predict(Xtest)


# 시험 데이터의 단어들에 대한 2차원 latent feature인 reduced_X를
# 데이터 프레임으로 정리한다.
final_pdframe = pd.DataFrame(reduced_X)
final_pdframe.columns = ["xaxis","yaxis"]
final_pdframe["word_indx"] = xsts
final_pdframe["word"] = final_pdframe["word_indx"].map(idx2word)

# 데이터 프레임에서 100개를 샘플링한다.
rows = final_pdframe.sample(n = 100)
labels = list(rows["word"])
xvals = list(rows["xaxis"])
yvals = list(rows["yaxis"])

# 샘플링된 100개 단어를 2차원 공간상에 배치한다.
# 거리가 가까운 단어들은 서로 관련이 높은 것들이다.
plt.figure(figsize=(15, 15))  

for i, label in enumerate(labels):
    x = xvals[i]
    y = yvals[i]
    plt.scatter(x, y)
    plt.annotate(label,xy=(x, y), xytext=(5, 2), textcoords='offset points',
                 ha='right', va='bottom', fontsize=15)
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.show()
