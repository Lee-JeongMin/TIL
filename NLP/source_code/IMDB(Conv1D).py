# IMDB using Word Embedding and Conv1D
# ----------------------------------------------------
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D
from tensorflow.keras.datasets import imdb
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# set parameters
max_features = 6000 # max_features : 최대 단어수
max_length = 400

# 학습 데이터는 자주 등장하는 단어 6,000개로 구성한다.
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

print(len(x_train), 'train observations')
print(len(x_test), 'test observations')
print(x_train[0])  # 6,000 이하의 word index로 구성돼 있다.

wind = imdb.get_word_index()
revind = dict((v,k) for k,v in wind.items())

def decode(sent_list):
    new_words = []
    for i in sent_list:
        # 0 : padding, 1 : 문서 시작, 2 : OOV로 사용함.
        # 실제 word index에서 3을 빼야함.
        # revind에서 i-3을 조회하고, 없으면 '*'로 채우라는 의미.
        new_words.append(revind.get(i-3, '*'))
    comb_words = " ".join(new_words)
    return comb_words

# 문장의 시작은 항상 '*'로 시작할 것임. 중간에 있는 '*'는 OOV일 것임.
decode(x_train[0])
   
# Pad sequences for computational efficiency
x_train = sequence.pad_sequences(x_train, maxlen=max_length)
x_test = sequence.pad_sequences(x_test, maxlen=max_length)

print(x_train[0])
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

# 각 문장의 OOV 개수 확인.
(x_train[0] == 2).sum()
(x_train[1] == 2).sum()

# Deep Learning architecture parameters
batch_size = 32
embedding_dims = 60
num_kernels = 260        # convolution filter 개수
kernel_size = 3          # convolution filter size
hidden_dims = 300
epochs = 10

model = Sequential()
model.add(Embedding(max_features, embedding_dims, input_length=max_length))
model.add(Dropout(0.2))
model.add(Conv1D(num_kernels, kernel_size, padding='valid', activation='relu', strides=1))
model.add(GlobalMaxPooling1D())
model.add(Dense(hidden_dims))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')
print(model.summary())

# 학습
hist = model.fit(x_train, y_train, 
                 batch_size=batch_size, 
                 epochs=epochs,
                 validation_data = (x_test, y_test))

# Loss history를 그린다
plt.plot(hist.history['loss'], label='Train loss')
plt.plot(hist.history['val_loss'], label = 'Test loss')
plt.legend()
plt.title("Loss history")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()

# 성능 확인
y_train_predclass = model.predict_classes(x_train, batch_size=batch_size)
y_test_predclass = model.predict_classes(x_test, batch_size=batch_size)

y_train_predclass.shape = y_train.shape
y_test_predclass.shape = y_test.shape

print (("Train accuracy:"),(np.round(accuracy_score(y_train,y_train_predclass),3)))  
print (("Test accuracy:"),(np.round(accuracy_score(y_test,y_test_predclass),3)))     
