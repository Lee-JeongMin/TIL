# IMDB using Word Embedding and LSTM
# ----------------------------------------------------

# IMDB 감정 분류 : Word Embedding & Bidirectional LSTM
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, Bidirectional, LSTM
from tensorflow.keras.datasets import imdb
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

max_features = 6000
max_length = 400

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

x_train = sequence.pad_sequences(x_train, maxlen=max_length)
x_test = sequence.pad_sequences(x_test, maxlen=max_length)

xInput = Input(batch_shape=(None, max_length))
xEmbed = Embedding(max_features, 60)(xInput)
xLstm = Bidirectional(LSTM(64))(xEmbed)
xOutput = Dense(1, activation='sigmoid')(xLstm)
model = Model(xInput, xOutput)
model.compile(loss='binary_crossentropy', optimizer='adam')

# 학습
hist = model.fit(x_train, y_train, 
                 batch_size=32, 
                 epochs=10,
                 validation_data = (x_test, y_test))

# Loss history를 그린다
plt.plot(hist.history['loss'], label='Train loss')
plt.plot(hist.history['val_loss'], label = 'Test loss')
plt.legend()
plt.title("Loss history")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()

y_hat = model.predict(x_test, batch_size=32)
y_hat_class = np.round(y_hat, 0)
y_hat_class.shape = y_test.shape

print (("Test accuracy:"),(np.round(accuracy_score(y_test,y_hat_class),3)))
