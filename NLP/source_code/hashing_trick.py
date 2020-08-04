# Hashing trick을 이용한 word embedding
# -------------------------------------
import tensorflow as tf
from tensorflow.keras.preprocessing.text import hashing_trick
from tensorflow.keras.layers import Input, Embedding, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
    
samples = ['너 오늘 이뻐 보인다', 
           '나는 오늘 기분이 더러워', 
           '끝내주는데, 좋은 일이 있나봐', 
           '나 좋은 일이 생겼어', 
           '아 오늘 진짜 짜증나', 
           '환상적인데, 정말 좋은거 같아']
labels = [[1], [0], [1], [1], [0], [1]]

# hash 테이블로 문서를 수치화한다.
VOCAB_SIZE = 10 # vocabulary 크기 (hash table)를 10개로 한정한다.
sequences = [hashing_trick(s, VOCAB_SIZE) for s in samples]
sequences = np.array(sequences)
labels = np.array(labels)
print(sequences)

# Embedding layer 내부의 출력층 개수임. 단어의 latent feature 개수
EMB_SIZE = 8

# 딥러닝 모델을 빌드한다.
xInput = Input(batch_shape=(None, sequences.shape[1]))
embed_input = Embedding(input_dim=VOCAB_SIZE + 1, output_dim=EMB_SIZE)(xInput)
embed_input1 = tf.reduce_mean(embed_input, axis=-1)

hidden_layer = Dense(128, activation=tf.nn.relu)(embed_input1)
output = Dense(1, activation='sigmoid')(hidden_layer)
model = Model(xInput, output)
model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.01))

# 학습
model.fit(sequences, labels, epochs=100)

# 추정
pred = model.predict(sequences)
print(np.round(pred, 0))
