# TF-IDF python 코드로 구현하기
# ------------------------------
import nltk
import numpy as np

# 1. dictionary를 생성한다.
def makeVocab(sentences):
    words = [word for sentence in sentences for word in sentence.split()]
    words = list(set(words))
    words.sort()
    return {word: idx for idx, word in enumerate(words)}

# 2. TF를 생성한다.
def makeTF(sentences):
    vocab = makeVocab(sentences)
    tf = np.zeros((len(vocab), len(sentences)))
    for i, sentence in enumerate(sentences):
        freq = nltk.FreqDist(nltk.word_tokenize(sentence))
        for key in freq.keys():
            tf[vocab[key], i] = freq[key] / len(sentence)
    return tf

# 3. IDF를 생성한다.
def makeIDF(sentences, tf):
    df = tf.shape[1] - (tf == 0.0).sum(axis=1)
    return np.log(tf.shape[1] / (0+df))

# 4. TFIDF를 생성한다.
def makeTFIDF(sentences):
    tf = makeTF(sentences)
    idf = makeIDF(sentences, tf)
    return np.multiply(tf, idf.reshape(tf.shape[0], 1))

sentences = ['gold silver truck', 'shipment of gold damaged in a fire', 'delivery of silver arrived in a silver truck', 'shipment of gold arrived in a truck']
tfidf = makeTFIDF(sentences);
print(tfidf.round(4))


