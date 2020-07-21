

# 자연어 처리(Natural Language Processing)

> 자연어란 우리가 사용하는 언어를 의미하고 자연어처리란 자연어의 의미를 분석하여 컴퓨터가 처리할 수 있도록 하는 인공지능 주요 분야 중 하나이다.
>
> 현재, 문서 분류, 기계 번역, 질의 응답 시스템, 문서 요약, 문서 교정, 음성 인식, 대화 시스템, 사용자 감성 분석 등에 사용되는 분야이다.



### NLTK(Natutal Language Toolkit) 

> Python 언어로 개발된 자연어 처리 및 문석 분석용 패키지이며, NLTK가 제공하는 주요 기능은 말뭉치, 토큰 생성, 형태소 분석, 품사 태깅이 있다.
>
> [NLTK book link](https://www.nltk.org/book/)

#### 설치

``` bash
pip install nltk
(conda install nltk)
```

#### 데이터 다운로드

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



### 편집거리(Edit distance)

> 레벤슈타인 거리(Levenshtein distance)라고 불리며,  문자열 s1을 문자열 s2로 교정하는데 드는 최소 횟수를 의미한다. 

문자열을 편집하는 액션을 3가지가 있다.

1. 삭제(delect) : '아이스크림' -> '아이'로 바꾸기 위해서 '스크림'을 삭제
2. 삽입(insert) : '아이' -> '아이스크림'으로 바꾸기 위해서 '스크림'을 삽입
3. 치환(substitution) : '아이오아이' -> '아이스크림'으로 바꾸기 위해서 '오아이'를 '스크림'으로 치환

자세한 설명 : https://lovit.github.io/nlp/2018/08/28/levenshtein_hangle/

### 품사 태깅(Part of speech Tagging)

> 품사란 문장의 구성 성분을 의미한다. (한국어의 경우 8품사가 있다.) 
>
> 품사 태깅이랑 문장에 사용된 단어들에 알맞는 품사를 정하는 것을 의미한다.
>
> 품사 태깅의 목적은 "문장의 의미(semantic) 파악", "문법(syntax)에 맞는 문장을 생성"이다.



NLTK를 사용해 품사 태깅이 가능하다.

* nltk.pos_tag(word_token) : 튜플로 묶은 단어와 품사를 리스트로 반환



#### Markov Chain

>현재 상태(state)의 확률은 이전 상태에만 의존한다.

![markov_chain](../markdown-images/markov_chain-1594877336923.PNG)

#### Hidden Markov Model

> 은닉 마코프 모델(HMM)은 각 상태(state)가 마코프 체인을 따르되 은닉(Hidden)되어 있는 것

![HMM](../markdown-images/HMM.PNG)

* 예시를 위한 그림 (Markov 모델에 관련해 사용한 이미지의 [출처](#참고-문헌)는 아래에 있다.)

![HMM_example](../markdown-images/HMM_example-1594877446961.PNG)



##### Forward-Backward 알고리즘

>전향-후향 알고리즘은 확률추정의 문제를 해결하기 위해 사용되는 알고리즘이다. 
>
>초기 확률(π), 천이확률(a), 출력확률(b)이 주어진 상황에서 관측 데이터 X 시퀀스가 나올 확률을 계산하는 알고리즘

관측데이터(X)가 Walk(`t-1`) -> Clean(`t`) 인 경우를 예를 들면, 

`t-1`인 경우

<span style = "background-color: #F5F6CE">Sunny -> Walk</span> : 0.4 x 0.6 = 0.24   // `t-1`이 맑고 걸을 확률 · · · ⓐ

<span style = "background-color: #E0F8F7">Rainy -> Walk</span> :  0.6 x 0.1 = 0.06  // `t-1`이 비가오고 걸을 확률 · · · ⓑ

`t`인 경우

`t-1`이 ⓐ 인 경우,

<span style = "background-color: #F5F6CE">Sunny -> Walk </span>-><span style = "background-color: #F8E0F1"> Rainy -> Clean</span> : 0.24 x 0.4 x 0.5 = 0.048  · · · ⓒ

<span style = "background-color: #F5F6CE">Sunny -> Walk </span>-> <span style = "background-color: #D8F6CE">Sunny -> Clean</span> : 0.24 x 0.6 x 0.1 = 0.0144  · · · ⓓ

ⓒ + ⓓ = 0.0624  



`t-1`이 ⓑ 인 경우,

<span style = "background-color: #E0F8F7">Rainy -> Walk</span> -> <span style = "background-color: #F8E0F1">Rainy -> Clean</span> : 0.06 x 0.7 x 0.5 = 0.021  · · · ⓔ

<span style = "background-color: #E0F8F7">Rainy -> Walk</span> -><span style = "background-color: #D8F6CE">Sunny -> Clean</span> : 0.06 x 0.3 x 0.1 = 0.0018  · · · ⓕ

ⓔ + ⓕ = 0.0228



∴ 관측데이터(X) : Walk -> Clean의 시퀀스가 나올 확률은 0.0624 + 0.0228 = 0.0852이다.

##### Viterbi decoding 알고리즘

>초기 확률(π), 천이확률(a), 출력확률(b), 관측데이터(X)가 주어신 상태에서 가장 가능성이 있는 히든상태(Z)의 시퀀스를 추정하는 알고리즘

관측데이터(X)가 Walk(`t-3`) -> Clean(`t-2`) -> Shop(`t-1`) -> Walk(`t`) 인 경우를 예를 들면, 

`t-3`인 경우

<span style = "background-color: #F5F6CE">Sunny -> Walk</span> : 0.4 x 0.6 = 0.24   // `t-1`이 맑고 걸을 확률 · · · ⓐ

<span style = "background-color: #E0F8F7">Rainy -> Walk</span> :  0.6 x 0.1 = 0.06  // `t-1`이 비가오고 걸을 확률 · · · ⓑ



`t-2`인 경우

`t-2`의 날씨가 <span style = "background-color: #F8E0F1">Rainy</span>인 경우

<span style = "background-color: #E0F8F7">Rainy -> Walk</span> -> <span style = "background-color: #F8E0F1">Rainy -> Clean</span> : 0.06 x 0.7 x 0.5 = 0.021  

<span style = "background-color: #F5F6CE">Sunny -> Walk </span>-><span style = "background-color: #F8E0F1"> Rainy -> Clean</span> : 0.24 x 0.4 x 0.5 = 0.048  · · · **max**

`t-2`의 날씨가 <span style = "background-color: #D8F6CE">Sunny </span>인 경우

<span style = "background-color: #F5F6CE">Sunny -> Walk </span>-> <span style = "background-color: #D8F6CE">Sunny -> Clean</span> : 0.24 x 0.6 x 0.1 = 0.0144  · · · **max**

<span style = "background-color: #E0F8F7">Rainy -> Walk</span> -><span style = "background-color: #D8F6CE">Sunny -> Clean</span> : 0.06 x 0.3 x 0.1 = 0.0018 



`t-1`인 경우

`t-1`의 날씨가 <span style = "background-color: #F6D8CE">Rainy</span>인 경우

<span style = "background-color: #F5F6CE">Sunny -> Walk </span>-><span style = "background-color: #F8E0F1"> Rainy -> Clean</span> -> <span style = "background-color: #F6D8CE">Rainy -> Shop</span> : 0.4 x 0.7 x 0.048  = 0.01344 · · · **max**

<span style = "background-color: #F5F6CE">Sunny -> Walk </span>-> <span style = "background-color: #D8F6CE">Sunny -> Clean</span>  -> <span style = "background-color: #F6D8CE">Rainy -> Shop</span> : 0.4 x 0.4 x 0.0144 = 0.002304

`t-1`의 날씨가 <span style = "background-color: #A9A9F5">Sunny </span>인 경우

<span style = "background-color: #F5F6CE">Sunny -> Walk </span>-><span style = "background-color: #F8E0F1"> Rainy -> Clean</span>  -> <span style = "background-color: #A9A9F5">Sunny -> Shop</span> : 0.3 x 0.3 x 0.048 = 0.00432 · · · **max**

<span style = "background-color: #F5F6CE">Sunny -> Walk </span>-> <span style = "background-color: #D8F6CE">Sunny -> Clean</span>  -> <span style = "background-color: #A9A9F5">Sunny -> Shop</span> : 0.3 x 0.6 x 0.0144 = 0.002592



`t`인 경우

`t`의 날씨가 <span style = "background-color: #D8D8D8">Rainy</span>인 경우

<span style = "background-color: #F5F6CE">Sunny -> Walk </span>-><span style = "background-color: #F8E0F1"> Rainy -> Clean</span> -> <span style = "background-color: #F6D8CE">Rainy -> Shop</span>  -> <span style = "background-color: #D8D8D8">Rainy -> Walk</span> : 0.1 x 0.7 x 0.01344 = 0.000941 · · · **max**

<span style = "background-color: #F5F6CE">Sunny -> Walk </span>-><span style = "background-color: #F8E0F1"> Rainy -> Clean</span>  -> <span style = "background-color: #A9A9F5">Sunny -> Shop</span>  -> <span style = "background-color: #D8D8D8">Rainy -> Walk</span> : 0.1 x 0.4 x 0.00432 = 0.000173

`t`의 날씨가 <span style = "background-color: #ADCBDF">Sunny </span>인 경우

<span style = "background-color: #F5F6CE">Sunny -> Walk </span>-><span style = "background-color: #F8E0F1"> Rainy -> Clean</span> -> <span style = "background-color: #F6D8CE">Rainy -> Shop</span>  -> <span style = "background-color: #ADCBDF">Sunny -> Walk</span> : 0.6 x 0.3 x 0.01344 = 0.002419 · · · <span style = "color: #FF0000">**max**</span>

<span style = "background-color: #F5F6CE">Sunny -> Walk </span>-><span style = "background-color: #F8E0F1"> Rainy -> Clean</span>  -> <span style = "background-color: #A9A9F5">Sunny -> Shop</span> -> <span style = "background-color: #ADCBDF">Sunny -> Walk</span> : 0.6 x 0.6 x 0.00432 = 0.001555



∴ 관측데이터(X) :  Walk -> Clean -> Shop -> Walk의 시퀀스가 나온 경우 

​     Z는 Sunny -> Rainy -> Rainy -> Sunny 가 0.2419%로 확률이 가장 높다.



##### Baum Welch 알고리즘

>관측데이터(X)만 주어진 경우 초기 확률(π), 천이확률(a), 출력확률(b), 히든상태(Z)를 추론하고 Z의 시퀀스를 찾아내는 알고리즘



### 문서 정보 추출(Infromation Extraction from Text)

>텍스트 문서로부터 특정 질문에 대한 정보를 추출하는 것



#### 정규 표현식(Regular Expression)

<img src="../markdown-images/jhkim-140117-RegularExpression-21.png" alt="jhkim-140117-RegularExpression-21" style="zoom:75%;" />

<img src="../markdown-images/jhkim-140117-RegularExpression-191.png" alt="jhkim-140117-RegularExpression-191" style="zoom:75%;" />

<img src="../markdown-images/jhkim-140117-RegularExpression-08-1.png" alt="jhkim-140117-RegularExpression-08-1" style="zoom:75%;" />



#####  re 패키지 기본 method

* re.compile()

* re.match(pattern, string, flags) : **문자열의 처음**부터 시작하여 패턴이 일치되는 것이 있는지 확인한다.

```python
print(re.match('su', 'super'))
# 결과 
# <re.Match object; span=(0, 2), match='su'>
# span=(0, 2) : 0번째 문자부터 2번때 문자 전까지
print(re.match('per','super'))
# 결과
# None
```

문자열 처음부터 찾기 때문에 중간의 문자는 따로 일치하는지 확인할 수 없기 때문에 None를 반환한다.

* re.search(pattern, string, flags) : 패턴이 일치되는 것이 있는지 확인한다.

```python
print(re.search('su', 'super'))
# 결과
# <re.Match object; span=(0, 2), match='su'>

print(re.search('per','super'))
# 결과
# <re.Match object; span=(2, 5), match='per'>
```

re.match와 기능은 같지만 문자열 처음부터 일치해야하는 것은 아니다. 그래서 중간 문자를 검색해도 결과가 출력되는걸 확인할 수 있다.

* re. sub(pattern, repl, string, count, flags) :  패턴에 일치되는 문자열을 다은 문자열로 바꿔주는 것이다.

```python
print(re.sub('\d{4}', 'XXXX', '010-1234-5678'))
# 결과
# 010-XXXX-XXXX
print(re.sub('\d{4}', 'XXXX', '010-1234-5678', count=1))
# 결과
# 010-XXXX-5678
print(re.sub('[a-z]', '', 'super123'))
# 결과
# 123
```

count 인자는 최대 몇 개까지 치환할 것인지를 의미한다. re.sub를 통해 문자열에서 일부분을 제거할 수도 있다. 

* re.split(pattern, string, maxsplit, flags) :  문자열에서 패턴을 기준으로 나누는 것

```python
result = re.split('<[^<>]*>',
                  '<html> Wow <head> header </head> <body> Hey </body> </html>')

result = list(map(lambda x: x.strip(), result))
result = list(filter(lambda x: x != '', result))
print(result)
# ['Wow', 'header', 'Hey']
```

* re.findall(pattern, string, flags) : 문자열 패턴과 일치되는 모든 부분은 찾는다.

```python
print(re.findall('s', 'supper'))
# 결과
# ['s']
print(re.findall('p', 'supper'))
# 결과
# ['p', 'p']
print(re.findall('a', 'supper'))
# 결과
# []
print(re.findall('sss', 'sssss'))
# 결과 
# ['sss']
```

마지막 코드의 결과를 보면 'sssss'로 s가 5개가 있음에도 'sss'가 한개만 출력이 되었다. "non-overlapping"로 반환된 리스트는 서로 겹치지 않는다는 의미이다.

* re.finditer(pattern, string, flags) : re.findall과 비슷하지만 일치된 문자열의 리스트 대신에 matchObj를 반환한다.

```python
matchObj = re.finditer('s', 'supper')
print(matchObj)
# <callable_iterator object at 0x000001A5343AAF88>
for Obj in matchObj:
    print(Obj)
    #<re.Match object; span=(0, 1), match='s'>
```

* re.fullmatch(pattern, string, flags) : 패턴과 문자열이 남는 부분 없이 완벽하게 일치하는지 검사한다.

```python
print(re.fullmatch('su', 'super'))
# 결과
# None
print(re.fullmatch('super', 'super'))
# 결과
#<re.Match object; span=(0, 5), match='super'>
```

완벽하게 일치해야지만 결과가 나오는것을 확인할 수 있다.

##### match Object 메서드

| Method  | Descrption                                         |
| ------- | -------------------------------------------------- |
| group() | 일치된 문자열을 반환                               |
| start() | 일치된 문자열의 시작 위치를 반환                   |
| end()   | 일치된 문자열의 끝 위치를 반환                     |
| span()  | 일치된 문자열의 (시작 위치, 끝 위치)를 튜플로 반환 |

```python
matchObj = re.search('match', "'matchObj' is a good name, but 'm' is convenient.")
print(matchObj)
# <re.Match object; span=(1, 6), match='match'>
print(matchObj.group())
# match
print(matchObj.start())
# 1
print(matchObj.end())
# 6
print(matchObj.span())
# (1, 6)
```

#### Chunking

> 여러 개의 품사로 구(Phrase)를 만드는 것을 chunking이라고 하고, 이 구(Phrase)를 Chunk라고 한다.

* 문법 정의 방법

```python
grammar = " NP : {<DT>?<JJ>*<NN>}"
# <>는 tag pattern을 의미
# 문장에서 (DT+JJ+JJ+NN), (DT+JJ+NN), (DT+NN), (JJ+NN) 등의 시퀀스를 모두 NP(명사구)로 판단한다.
```

#### Chinking

> 특정 부분을 Chunk 밖으로 빼내는 것을 의미

```python
grammar = r""" 
    NP : {<.*>+}         # 모든 것을 Chunk함
         }<VDB|IN>+{"""  # VBD랑 IN은 Chunk에서 빼낸다.
```





### 참고 문헌

* 마코프 이미지 출처 : https://blog.naver.com/chunjein/221034077798
* 정규 표현식 이미지 출처 :  http://www.nextree.co.kr/p4327/

