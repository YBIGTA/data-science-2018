---
layout: post
title: "Face Recognition"
subtitle: "recognizing face image with one shot learning"
mathjax: true
date: 2018-02-02 23:45:00
categories: ml
---

* 작성자: YBIGTA 10기 김지중
* reference: [Coursera Convolutional Neural Network](https://www.coursera.org/learn/convolutional-neural-networks/home/welcome) Week4


### 얼굴 인식 관련 두 가지 문제

1. Verification

   주어진 이미지가 특정 사람인지 아닌지 검증하는 문제 (1:1 문제)

   input: face image, 식별 id

   output: 해당 id에 맞는 사람일 확률

2. Recognition

   주어진 이미지가 K명의 사람 중 누구인지 맞추는 문제

   input: face image

   output: K-d 확률 벡터

직관적으로 생각해도 verification이 recognition보다 쉽다. 레이블이 많으면 실수할 확률이 높고, 실제로도 그렇게 관측된다. verification 모델을 만들고, 충분히 높은 accuracy를 달성한 이후 recognition 모델로 넘어가는 것이 좋다. 본 강의는 verification 모델을 중심으로 진행한다.



### One Shot Learning

일반적으로 image classification 문제는 많은 수의 이미지 데이터를 기반으로 한다 (ImageNet 데이터셋의 경우, 총 class 수는 1000개, 총 이미지 수는 1.2m개.) 근데 일부 직장에서는 출입시 id카드 대신 얼굴 인식을 사용한다고 한다. 직원당 100장 200장의 이미지 데이터를 모아야하는 것일까?

이를 극복하기 위해 제안된 개념이 one shot learning이다. one shot learning을 통해 적은 수의 sample로도 recognition이 가능하다. 개념적으로, one shot learning은 극단적인 경우 하나의 sample로 training을 해도 해당 class에 대해서 recognition이 가능하다. 

이 때 모델을 어떻게 짜야할까? 가장 떠올리기 쉬운 방법은 그냥 CNN classification 모델을 만드는 것이다. 하지만 이는 잘 작동하지 않는다. 트레이닝 샘플이 적어 robust할 수가 없다. 뿐만 아니라, 새로운 class가 추가된다면 뒷단의 softmax layer를 수정하고 다시 학습시켜야할 것이다. 이를 해결하고자  "similarity fucntion"을 사용한다. 

![Imgur](https://i.imgur.com/ZOP8JR9.png)

먼저, 두 이미지 사이의 거리를 잰다. 이 때 계산된 거리가 임계치($\tau$) 이하라면, 두 이미지는 같은 사람이라고 판단한다. 임계치보다 크다면 다른 사람이라고 판단한다. 즉, 거리를 기반으로 유사성을 판단하는 것이다. 그럼 이 거리는 어떻게 계산할까? 두 이미지를 pixel-wise로 유클리디안 거리를 계산하면 될까?



### Siamese network

불가능한건 아니지만, 보다 좋은 방법이 있다. Siamese network가 그 예시다. 기본 구조는 아래와 같다.

![Imgur](https://i.imgur.com/aOh2isb.png)

* input: 서로 다른 두 이미지
* output: 이미지들의 embedding vector (여기선 128-d)

서로 다른 두 이미지를 input으로 받아 CNN을 통해서 embedding vector로 만들어준다. 이 때 이 embedding vector간의 유클리디안 거리를 계산한다. embedding vector가 원 이미지를 잘 represent한다면, 계산된 거리를 기준으로 동일 인물인지 아닌지를 판단할 수 있다. 그럼, 네트워크를 어떻게 학습시켜야 좋은 representation이 될까?



### Trieplet Loss

이 때 등장하는 개념이 triplet loss다. 

![Imgur](https://i.imgur.com/0fl4oxk.png)

기준이 되는 한 이미지를 고른다. 이를 anchor라고 부른다. anchor와 동일한 인물의 이미지를 고른다. 이를 positive라 부른다. anchor와 다른 인물의 이미지를 고르며, 이를 negative라 부른다. 이 때 (anchor,positive,negative)의 묶음을 triplet이라 부른다. 우리의 목적은 같은 사람의 이미지끼리는 유사한, 다른 사람의 이미지끼리는 서로 상이한 벡터로 표현하는 것이다. 이를 위해서는 anchor-positive 간 거리가 anchor-negative간 거리보다 작아지도록 학습을 시켜야 한다. 

$ d(A,P) \le d(A,N)$

이 때, 두 이미지간 거리는 embedding vector의 거리로 볼 수 있다. 따라서,

$ {\left \Vert f(A) - f(P)\right \Vert}^2 \le {\left \Vert f(A) - f(N)\right \Vert}^2 $

이렇게만 학습시킨다면, positive-distance가 negative-distance보다 작게끔 학습이 되겠지만, 두 거리간 차이가 보장되지 않는다. 안정적인 학습을 위해서는 두 거리간 gap을 보장해야한다. 이 때문에  위 식에 margin($\alpha$)를 도입한다.

$ {\left \Vert f(A) - f(P)\right \Vert}^2 +\alpha \le {\left \Vert f(A) - f(N)\right \Vert}^2 $

이를 통해 positive-distance가 negative-distance보다 특정 margin만큼 작게끔 학습할 수 있다. 이 부등식에서 우변을 좌측으로 넘기면 아래와 같다.

$ {\left \Vert f(A) - f(P)\right \Vert}^2 +\alpha - {\left \Vert f(A) - f(N)\right \Vert}^2 \le 0$

이러한 목적을 loss function으로 표현하면 아래와 같다.

$ L(A,P,N) = max({\left \Vert f(A) - f(P)\right \Vert}^2 +\alpha - {\left \Vert f(A) - f(N)\right \Vert}^2,0)$

이제 loss function까지 살펴보았다. 그래서 트레이닝은 어떻게 시킨다는 것일까? 데이터셋에서 모든 가능한 triplet을 구성하면 이 목적함수를 쉽게 달성하는 triplet이 많을 것이다. 완전 다른 두 사람의 이미지간의 거리는 클 거니깐. 또한, 모든 triplet을 만들어서 학습시키는 것 자체가 엄청나게 큰 연산. 그래서 [curriculum learning](https://ronan.collobert.com/pub/matos/2009_curriculum_icml.pdf)을 도입한다. 쉽게 말해 학습시키기 어려운 triplet만 골라낸다. 여기서 학습시키기 어려운 triplet이란, (1) 같은 사람인데 거리가 먼 케이스 (2) 다른 사람인데 거리가 가까운 케이스, 두 가지로 구분할 수 있다. (1)을 hard-positive, (2)를 hard-negative라 부른다. 효율적인 학습을 위해 트레이닝 셋에서 hard-positive와 hard-negative를 추출하여 이들로부터 loss를 계산한다. 이러한 방법론은 구글의 논문 [FaceNet](https://arxiv.org/abs/1503.03832)에서 제안되었으며, 스터디원 10기 강병규의 [블로그](https://kangbk0120.github.io/articles/2018-01/face-net)에 요약글이 게시되어있다. 



### Face Verification and Binary Classification

다시 본론으로 돌아와, Siamese Network를 face verification 문제에 어떻게 적용할 수 있을까? 문제가 face verification으로 한정된다면, triplet loss보다 더 쉬운 방법으로 학습이 가능하다.

![Imgur](https://i.imgur.com/hFfa733.png)

Simaese Network 뒷단에 logistic unit을 붙여주는 것이다. **두 벡터의 차이**를 input으로 갖는 logistic unit (FC + sigmoid)를 붙여주면, 0~1의 확률값을 output으로 갖게 된다. 따라서 두 이미지가 같은 사람이면 1, 다른 사람이면 0으로 표현하는 것이 가능해진다. **두 벡터의 차이**를 나타내는 방법에는 여러가지가 있는데, 강의에서는 크게 두 가지를 설명한다. 

(1) $L_1$ distance 형식 (파란색 글씨)

(2) $\chi^2$ distance 형식 (초록색 글씨)

Facebook의 논문 [DeepFace](https://www.cs.toronto.edu/~ranzato/publications/taigman_cvpr14.pdf)에서는 (2)의 방법이 적용되었다고 한다.



