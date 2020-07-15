

# 자연어 처리(Natural Language Processing)

> 자연어란 우리가 사용하는 언어를 의미하고 자연어처리란 자연어의 의미를 분석하여 컴퓨터가 처리할 수 있도록 하는 인공지능 주요 분야 중 하나이다.
>
> 현재, 문서 분류, 기계 번역, 질의 응답 시스템, 문서 요약, 문서 교정, 음성 인식, 대화 시스템, 사용자 감성 분석 등에 사용되는 분야이다.



### NLTK(Natutal Language Toolkit) 

> Python 언어로 개발된 자연어 처리 및 문석 분석용 패키지이며, NLTK가 제공하는 주요 기능은 말뭉치, 토큰 생성, 형태소 분석, 품사 태깅이 있다.
>
> [NLTK book link](https://www.nltk.org/book/)

##### 설치

``` bash
pip install nltk
(conda install nltk)
```

##### 데이터 다운로드

```python
import nltk
nltk.download()
```

* 아래와 같이 데이터 다운로드 진행

<img src="../markdown-images/nltk_book_download.PNG" alt="nltk_book_download" style="zoom: 80%;" />

### 말뭉치(Corpus)

> 문서들의 집합을 의미한다.



NLTK를 사용해 코퍼스를 핸들링할 수 있다.

* corpus.fileids() : 새롭게 만든 말뭉치에 포함된 파일 리스트를 반환
* corpus.raw() : 말뭉치에 있는 '원천(raw)' 텍스트를 반환
* corpus.sents() :  모든 문장을 리스트로 반환
* corpus.words() : 모든 단어를 리스트에 넣어서 반환
* nltk.word_tokenize(text`(=corpus.raw())`) :  단어로 토큰화
* nltk.sent_tokenize(text`(=corpus.raw())`) :  문장으로 토큰화
* nltk.PorterStemmer() / nltk.LancasterStemmer() : 어간 추출
*  stopwords.words('english') : 영어 불용어
* nltk.FreqDist(text) : 입력 단어 목록에 매핑되는 단어와 해당 빈도가 포함된 걔체 반환
* wordnet.synsets(단어) :  유의어의 묶음
* wordnet.synsets(단어.품사.순번) :  단어.품사.순번(ex; tree.n.01)로 구성
* wordnet.synset(단어).lemmas() :  단어의 원형
* wordnet.synset(단어).hypernyms() : 단어의 상위어
* wordnet.synset(단어).hyponyms() :  단어의 하위어
* x.path_similarity(y) :  두 synset간(x, y)의 의미론적 유사도(0~1)
* 이 외의 함수는 [여기](https://moonnightfiction.tistory.com/entry/NLTK-02?category=779001)를 참고하면 잘 정리되어있다.



