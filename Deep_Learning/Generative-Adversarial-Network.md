# 생성적 적대 신경망(GAN)

>생성 모델(Generator)로 진짜같은 모방데이터를 생성하고 분류 모델(Discriminator)로 진짜데이터와 가짜데이터를 구별하는 알고리즘 (비지도학습)



### GAN 개념

![GAN모델그림](markdown-images/GAN%EB%AA%A8%EB%8D%B8%EA%B7%B8%EB%A6%BC.png)

* Generator 모델은 랜덤 데이터(Noise)를 입력 받아 훈련을 통해 실제 데이터와 유사한 가짜 데이터를 생성한다.
*  Discriminator 모델은 실제 데이터(1)와 가짜 데이터(0)를 판별한다.
* Discriminator 모델은 0 ~ 1사이의 값(sigmoid)을 출력한다. (0 = 가짜, 1 = 진짜)
* Generator 모델은 Discriminator 모델이 구별하지 못하도록 실제 데이터와 유사한 가짜 데이터를 생성하고, Discriminator 모델은 실제 데이터와 가짜 데이터를 구분하도록 학습한다.

![gan설명](markdown-images/gan%EC%84%A4%EB%AA%85.png)

* **x** : 실제 데이터 / **z** : 랜덤데이터 / **<span style="color:blue">파란선</span> **: 모델D의 출력 / **<span style="color:green">초록선</span>** : G(z)의 분포 /  **검정선** : x의 분포

(a) : 학습 전 상태이다.

(b) : 학습 중간 상태로 D는 데이터를 구별한다.

(c) : 학습 중간 상태로 <span style="color:green">초록선</span>이 <span style="color:black">검정선</span>과 비슷한 분포로 흐르게 된다.

(d) : 학습이 끝난 후 <span style="color:green">초록선</span>은 <span style="color:black">검정선</span>과 같아지게 되고 <span style="color:blue">파란선</span>은 더이상 구분할 수 없게 된다.



### GAN학습을 위한 Loss function

![GAN모델설명](markdown-images/GAN%EB%AA%A8%EB%8D%B8%EC%84%A4%EB%AA%85.png)

* GAN에서 모델 (G)와 모델 (D)를 학습시키기 위해서 위와 같은 Loss function V(D,G)을 사용한다.
*  <span style="color:orange">모델 (G)</span>는 V(D,G)가 **최소**가 되도록 weight와 bias를 업데이트하고, <span style="color:skyblue">모델 (D)</span>는 V(D,G)가 **최대**가 되도록 weight와 bias를 업데이트한다.
* <span style="color:orange">모델 (G)</span>는 D(G(z))가 1에 가까운 값이 나오도록 업데이트하고, <span style="color:skyblue">모델 (D)</span>는 D(x)는 1, D(G(z))는 0에 가까운 값이 나오도록 업데이트 한다.
* 학습이 잘된경우는 모델 (D)가 x와 G(x)를 구별하지 못한다는 것은 즉, x ≈ G(z)라는 것이다.



![gan_loss](markdown-images/gan_loss.png)

* D(G(z))가 0일 때, 모델 (G)는 `minV(G)`가 된다.  
* D(x)가 1이고, D(G(z))가 0일 때, 모델 (D)는 `maxV(D)`가 된다.

### GAN의 한계

* GAN은 고해상도의 이미지 생성이 불가, 학습의 불안정의 문제가 있다. 이러한 문제를 해결하기 위해 수많은 변형된 GAN이 등장한다. 그 중 안정적인 학습이 가능한 DCGAN이 등장하였다.

### DCGAN

> GAN에 Deep Convolution 레이어를 쌓아서 만든 알고리즘

![DCGAN](markdown-images/DCGAN.png)

### DCGAN 개념

* 이미지의 위치 정보를 유지하기 위해 선형 레이어와 풀링 레이어는 배제하고 **합성곱(Convolution)**과 **Transposed Convolution**으로 네트워크 구조를 만든다.
* **배치 정규화(Batch Normalization)**를 사용해 입력 데이터 분포가 치우쳐져 있는 경우 평균과 분산을 조정해 학습이 안정적으로 이뤄지도록 돕는다.
* 모델 (G)에는 ReLU를 사용하고, 모델 (D)에는 LeakyReLU를 사용한다. 
* 아래의 이미지는 안정적인 DCGAN을 구현하는 가이드라인이다.

![DCGAN_guideline](markdown-images/DCGAN_guideline.PNG)



### 참고 문헌

* [GAN](https://arxiv.org/abs/1406.2661)
* [DCGAN](https://arxiv.org/abs/1511.06434)









