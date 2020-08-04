# Word Embedding

> 학습을 통해 단어를 수치화(벡터화)하는 방법

## 카운트 기반 방법

* SVD, LSA, PCA, HAL

## 예측 방법

* word2vec, NNLM, RNNLM









### Hashing Trick

> vocabulary 크기를 미리 지정하고 (hash table) hash()함수를 사용해 단어들을 hash table에 대응시키는 방식

Hashing trick의 단점은 `collision`발생이다. 다른 두 단어가(ex, 'apple', 'tiger') 같은 hash값을 가지는 경우가 발생한다. DB에서는 이를 처리하기위해 overflow table을 사용한다.

반면, 메모리 및 실행 시간을 단축시키고, key의 영역이 넓을수록 효율적이다.



### Co-occurrence matrix

> 빈도기반의 단어 수치화

co-occurrence matrix는 단어간 유사도가 존재할 수 있으며 corpus가 크다면 계산량이 많이지는 단점이 있다. 

### 소스 코드

* [IMDB(Conv1D)](https://github.com/Lee-JeongMin/TIL/blob/master/NLP/source_code/IMDB(Conv1D).py)
* [IMDB(LSTM)](https://github.com/Lee-JeongMin/TIL/blob/master/NLP/source_code/IMDB(LSTM).py)
* [IMDB(Conv1D-LSTM)](https://github.com/Lee-JeongMin/TIL/blob/master/NLP/source_code/IMDB(Conv1D-LSTM).py)
* [hashing trick](https://github.com/Lee-JeongMin/TIL/blob/master/NLP/source_code/hashing_trick.py)
* [co-occurrence matrix](https://github.com/Lee-JeongMin/TIL/blob/master/NLP/source_code/co_occurrence.py)

