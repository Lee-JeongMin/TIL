# 마크다운 문법 정리

## 제목(heading)

제목은 `#`의 갯수로 표현된다. 

H6, 제목을 6단계까지 구분하여 작성할 수 있다.

제목을 잘 작성해두면, Overview나 목차로써 활용 가능하다.

### 제목 3

#### 제목 4

##### 제목 5

###### 제목 6



## 목록

* 순서가 없는 목록은 `*`으로 작성된다.
  * tab을 누르면, 하위 목록으로 작성 가능하다.
  * 엔터
* shift + tab을 누르면 상위 목록 레벨으로 작성 가능하다.

1. 순서가 있는 목록은 2.
2. `1.`으로 시작하면 된다,
   1. 하위 목록으로도 작성 가능하다.
      * 섞어서 사용도 가능하다.

## 코드블록(```)

```python
import random
numbers = range(1, 46)
print(random.sample(numbers,6))
```

다양한 프로그래밍 언어에 대한 syntax highlighting 기능을 접할 수 있다.

```html
<h1>
    안녕
</h1>
```

### 인라인 코드블록(`)

인라인 코드 블록은 아래와 같이 활용될 수 있다.

본 프로젝트는 `tensorflow`를 활용하였습니다.

출력을 하기 위해서는 `print`함수를 활용한다.

## 텍스트 기타 문법

*강조(기울임) italic*

**강조(굵게) bold**

~~취소선~~

## 링크

[구글](https://google.co.kr)

[git 문서](./git.md)

## 이미지

![hot-air-balloons-4381674_1920](C:\Users\student\Desktop\hot-air-balloons-4381674_1920.jpg)

* 기본 설정으로 이미지를 가져오면 절대 경로로 표기된다.(C드라이브/~~)
* 이 경우, Github나 다른 컴퓨터에는 해당 이미지가 없어 깨지는 현상이 발생할 수 있다. 따라서 아래의 설정을 해주자.
  * 파일-환경설정-이미지- 사진과 같이 설정 변경![캡처](markdown-images/%EC%BA%A1%EC%B2%98.PNG)

## 표

| 순번 | 이름   |
| ---- | ------ |
| 1    | 홍길동 |
| 2    | 김철수 |



## 수학 수식

`$$`을 사용하여 수식을 입력할 수 있다. $$ 사이에 수식을 입력하면 된다.

![캡처](markdown-images/%EC%BA%A1%EC%B2%98-1595406042478.PNG)
$$
\vec{a}, \lVert a \rVert, {\operatorname{y}\over\operatorname{x}}, \sqrt{2}
$$

$$
+, -, \times, \div, \{\}, \cdots, \text{한글 작성}
$$

$$
a^2, a_2, a^{2+2}, a_{ij},\sum_{k=1}^n k^2, 	\frac{2}{4}, 	{2 \over 4}, \left(\frac{2}{4} \right), {n \choose k}
$$

$$
	\begin{pmatrix} x & y \\ z & v \end{pmatrix}, 	\begin{Bmatrix} x & y \\ z & v \end{Bmatrix}, 	\begin{vmatrix} x & y \\ z & v \end{vmatrix}, 	\begin{Vmatrix} x & y \\ z & v \end{Vmatrix}, 	\begin{matrix} x & y \\ z & v \end{matrix}
$$























