# Skipgram Negative Sampling 예시
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
from tensorflow.keras.layers import Input, Dense, Dropout, concatenate
from tensorflow.keras.models import Model
import random
from tensorflow.keras.optimizers import Adam

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
lines = [preprocessing(line) for line in fin if len(line) > 0]
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

xs = []     # 입력 데이터
ys = []     # 출력 데이터

def trigrams(lines, word2idx):
    for line in lines:
    
        embedding = [word2idx[w.lower()] for w in nltk.word_tokenize(line)]
        
        triples = list(nltk.trigrams(embedding))
        
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
        label=[1]*len(xs) 
    
    new_xs, new_ys,new_label = [], [], []
    for x in xs: 
        random_wordidx = random.choice(list(idx2word.keys()))
        new_xs.append(x)
        new_ys.append(random_wordidx)
        new_label.append(0)
        
    xs.extend(new_xs)
    ys.extend(new_ys)
    label.extend(new_label)


    return xs, ys, label

def bigrams(lines, word2idx):
    for line in lines:
        # 사전에 부여된 번호로 단어들을 표시한다.
        embedding = [word2idx[w.lower()] for w in nltk.word_tokenize(line)]
        
        # bigram으로 주변 단어들을 묶는다.
        bigram = list(nltk.bigrams(embedding))
        
        # 왼쪽 단어, 중간 단어, 오른쪽 단어로 분리한다.
        w_lefts = [x[0] for x in bigram]
        w_rights = [x[1] for x in bigram]
    
        # 입력 (xs)      출력 (xy)
        # ---------    -----------
        # 왼쪽 단어 --> 오른쪽 단어
        xs.extend(w_lefts)
        copy_xs = xs.copy()
        ys.extend(w_rights)
        label=[1]*len(xs) 
    
    new_xs, new_ys,new_label = [], [], []
    for x in xs: 
        random_wordidx = random.choice(list(idx2word.keys()))
        new_xs.append(x)
        new_ys.append(random_wordidx)
        new_label.append(0)
        
    xs.extend(new_xs)
    ys.extend(new_ys)
    label.extend(new_label)

    return xs, ys, label

xs, ys, label = trigrams(lines, word2idx)
# xs, ys, label = bigrams(lines, word2idx)

# 학습 데이터를 one-hot 형태로 바꾸고, 학습용과 시험용으로 분리한다.
vocab_size = len(word2idx) + 1  # 사전의 크기

ohe = OneHotEncoder(categories = [range(vocab_size)])
input_X = ohe.fit_transform(np.array(xs).reshape(-1, 1)).todense()
target_X = ohe.fit_transform(np.array(ys).reshape(-1, 1)).todense()

input_train, input_test, target_train, target_test, label_train, label_test ,xstr, xsts \
    = train_test_split(input_X, target_X, label, xs, test_size=0.2)
label_train = np.asarray(label_train)
label_test = np.asarray(label_test)

print(input_train.shape, input_test.shape, target_train.shape, target_test.shape, len(label_train), len(label_test))
print(len(xstr), len(xsts))

# 딥러닝 모델을 생성한다.
BATCH_SIZE = 128
NUM_EPOCHS = 100

# one-hot 안쓰려면 Dense레이어가 아니라 keras에서 제공하는 embedding layer를 사용하면 된다.
input_layer = Input(shape = (input_train.shape[1],), name="input")
first_layer = Dense(300, activation='relu',name = "first")(input_layer)
first_dropout = Dropout(0.5, name="firstdout")(first_layer)
second_layer = Dense(2, activation='relu', name="second")(first_dropout)


input_layer2 = Input(shape = (target_train.shape[1],), name="input2")
first_layer2 = Dense(300, activation='relu',name = "first2")(input_layer2)
first_dropout2 = Dropout(0.5, name="firstdout2")(first_layer2)
second_layer2 = Dense(2, activation='relu', name="second2")(first_dropout2)


merge = concatenate([second_layer, second_layer2])
merge_Output = Dense(128, activation='relu')(merge)
merge_Output = Dense(1, activation='sigmoid')(merge_Output)
# dot을 쓰면 dense안써도 되고 대신에 activation함수를 사용해야함

model = Model([input_layer, input_layer2], merge_Output)
model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.01))
print(model.summary())
    
# 학습
hist = model.fit([input_train, target_train], label_train, 
                 batch_size=BATCH_SIZE,
                 epochs=NUM_EPOCHS,
                 shuffle=True,
                 validation_data = ([input_test, target_test], label_test))

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
print(input_test)
reduced_X = encoder.predict(input_train)

# 시험 데이터의 단어들에 대한 2차원 latent feature인 reduced_X를
# 데이터 프레임으로 정리한다.
final_pdframe = pd.DataFrame(reduced_X)
final_pdframe.columns = ["xaxis","yaxis"]
final_pdframe["word_indx"] = xstr
final_pdframe["word"] = final_pdframe["word_indx"].map(idx2word)

# 데이터 프레임에서 100개를 샘플링한다.
rows = final_pdframe.sample(n = 100)
labels = list(rows["word"])
xvals = list(rows["xaxis"])
yvals = list(rows["yaxis"])

# 샘플링된 100개 단어를 2차원 공간상에 배치한다.
# 거리가 가까운 단어들은 서로 관련이 높은 것들이다.
plt.figure(figsize=(20, 20))  

for i, label in enumerate(labels):
    x = xvals[i]
    y = yvals[i]
    plt.scatter(x, y)
    plt.annotate(label,xy=(x, y), xytext=(5, 2), textcoords='offset points',
                  ha='right', va='bottom', fontsize=15)
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
# plt.xlim(0, 1.2)
# plt.ylim(0, 0.8)
plt.show()
