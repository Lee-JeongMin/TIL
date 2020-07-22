## 문서 요약 (Text Summary)

### TextRank  알고리즘

> PageRank 알고리즘을 확장시켜 문서 내의 핵심문장을 추출하는 알고리즘

![textrank](C:%5CUsers%5Cstudent%5CDesktop%5CTIL%5Cmarkdown-images%5Ctextrank.PNG)

위와 같은 사진의 경우를 예를 들어 생각해보자.

1) 각 문장 (A, B, C, D)의 TextRank를 초기화한다.
$$
TR(A) = TR(B) = TR(C) = TR(D) = {\operatorname{1}\over\operatorname{4}\text{(문장의 수)}} = 0.25
$$
2) 문장 간의 유사도를 측정한다.
$$
Similarity(S_i, S_j) = 	{\operatorname{\left| w_k | w_k ∈ S_i \& w_k ∈ S_j \right|}\over\operatorname{log(	\left| S_i \right|)+log(	\left| S_j \right|)}}
$$

* 문장 S_i와 문장 S_j의 단어의 개수에서 두 문장에서 공통으로 들어간 단어의 개수로 나누면 된다.

3) TR  공식을 통해 TextRank 계산 - 반복
$$
TR(S_i) = (1-d) + d \times	\sum_{V_j∈In(V_i)}{\operatorname{w_{ji}}\over\operatorname{\sum_{V_k∈Out(V_j)}}}TR(S_j)
$$


(if d=1)
$$
TR(A) ={\operatorname{0.2}\over\operatorname{0.2+0.4}} \times0.25 + {\operatorname{0.3}\over\operatorname{0.3}} \times0.25 = 0.33
$$

$$
TR(B) ={\operatorname{0.2}\over\operatorname{0.2+0.3}} \times0.25 + {\operatorname{0.4}\over\operatorname{0.4}} \times0.25 = 0.35
$$

따라서 , 문장 B가 중요함을 알 수 있다.

**TextRank가 높은 순서대로 뽑으면 문서A를 대표함을 의미한다.**



### Anaphora Resolution (조응어 해석)

>이전에 나온 표현에서 의미를 빌려오는 것을 대용어라고 한다. 이를 딥러닝이 알고 파악하도록하는 것은 아직 힘들다.

대용어란, 예를 들면 

"John is a man. He walks." 라는 문장에서 "He"는 "John"을 재표현한 것이다. 이런 관계를 해석하는 것을 조응어 해석이라고 한다.



### Word Sense Disambiguation (WSD, 단어의 중의성 분석)

>단어는 여러 의미를 지니고 있고 이는 문장의 문맥 파악을 통해 의미를 알 수 있다. 이와 같이, 문장 내에서 이 의미를 파악하는 것을 WSD라고 한다. 
>
>의미를 파악하기 위해선 단어사전이 필요하고 주변단어와의 비교가 필요하다.

### Lesk Algorithm

>문장에 사용된 중의성을 가진 단어의 뜻을 단어사전(워드넷)에 뜻풀이, 예제 문항에 등장하는 단어와 분석 문장의 단어와 비교하여 겹치는 단어가 많은 뜻풀이를 선택해 단어의 의미를 파악하는 알고리즘이다.



### VADER-Sentiment-Analysis

> 규칙기반 알고리즘으로 10명이 느끼는 감정 상태를 조사해서 얻은 점수(-4 ~ +4)를 기반으로 문장의 감정상태를 추정한다.

* nltk.downloader.download('vader_lexicon','download_dir='./dataset/)을 다운 받으면 된다.