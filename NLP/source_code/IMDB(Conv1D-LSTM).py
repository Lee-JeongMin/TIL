#  LSTM의 latent feature와  CNN latent feature를 합쳐서 사용
# --------------------------------------------------------- 

from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM,Bidirectional, concatenate
from tensorflow.keras.layers import Embedding 
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import imdb
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0' # 초기화할 GPU number

# set parameters
max_features = 6000 # max_features : 최대 단어수
max_length = 400

# 학습 데이터는 자주 등장하는 단어 6,000개로 구성한다.
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

wind = imdb.get_word_index()
revind = dict((v,k) for k,v in wind.items())

# Pad sequences for computational efficiency
x_train = sequence.pad_sequences(x_train, maxlen=max_length)
x_test = sequence.pad_sequences(x_test, maxlen=max_length)

# Deep Learning architecture parameters
batch_size = 32
embedding_dims = 60
num_kernels = 260        # convolution filter 개수
kernel_size = 3          # convolution filter size
hidden_dims = 300
epochs = 10
nOutput = 1


xInput = Input(batch_shape=(None, max_length))
xEmbed = Embedding(max_features, embedding_dims)(xInput)

# LSTM 모델
xLSTM = Bidirectional(LSTM(64))(xEmbed)

# CNN 모델
xDropout1 = Dropout(0.2)(xEmbed)
xConv1 = Conv1D(filters=num_kernels, kernel_size=kernel_size, padding='valid', strides=1, activation='relu')(xDropout1)
xPool1 = MaxPooling1D(pool_size=kernel_size, strides=1, padding='valid')(xConv1)
xHidden = Dense(hidden_dims)(xPool1)
xDropout2 = Dropout(0.5)(xHidden)
xHidden2 = Dense(100)(xDropout2)
xFlat = Flatten()(xHidden2)

merge = concatenate([xLSTM, xFlat])
merge_Output = Dense(16, activation='relu')(merge)
merge_Output = Dense(nOutput)(merge_Output)


model = Model(xInput, merge_Output)
model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.01))


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
y_hat = model.predict(x_test, batch_size=32)
y_hat_class = np.round(y_hat, 0)
y_hat_class.shape = y_test.shape

print (("Test accuracy:"),(np.round(accuracy_score(y_test,y_hat_class),3)))     
