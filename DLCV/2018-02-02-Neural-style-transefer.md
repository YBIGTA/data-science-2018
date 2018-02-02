---
layout: post
title: "Neural Style Transfer"
subtitle: "원본 이미지에 새로운 스타일을 입히는 방법"
mathjax: true
date: 2018-02-02 23:50:00
categories: ml
---



### What is Neural Style Transfer

![Imgur](https://i.imgur.com/GYSN3QN.png)

이미지 생성 혹은 합성의 일종으로 생각할 수 있다. 특정 이미지에 다른 이미지로부터 추출한 스타일을 입혀 새로운 이미지를 만드는 것이다. 이 때 생성된 이미지의 내용을 담고있는 이미지를 Content(C), 스타일을 담고 있는 이미지를 Style(S), 생성된 이미지를 Generated Image(G)라고 부른다. 



### Cost Function

이미지가 얼마나 잘 생성되었는 지 평가하기 위해 cost function J(G)를 정의해보자.

(1) 기본적으로, 원래 이미지와 비슷한 content를 담고 있어야한다. 이를 $J_{content}$라고 표현한다.

(2) 또한, 입혀주고자 하는 스타일이 잘 반영되어있어야 한다. 이를 $J_style$이라 부른다.

이 두가지 cost를 가중결합하여 전체 cost를 표현할 수 있다. 아래 식에서 $\alpha$,$\beta$는 하이퍼파라미터로, $\alpha$를 통해 content 표현력을, $\beta$를 통해 style 표현력을 조정할 수 있다.

$ J(G) =\alpha  J_{content}(C,G) + \beta J_{style}(S,G)$



### Content Cost Function

Content cost는 아래와 같은 방법으로 계산된다.

(1) pre-trained ConvNet을 불러온다.

(2) C(Content Image), G(Generated Image)를 네트워크에 집어넣는다.

(3) $l$번째 hidden layer까지만 foward한다. 이 때의 output을 각각  $a^{(l)[C]}$,$ a^{(l)[G]}$라 한다.

* $l$은 너무 얕지도, 너무 깊지도 않은 중간 단계의 레이어로 뽑는다고 한다.

(4) $a^{(l)[C]}$,$ a^{(l)[G]}$가 유사하다면, 두 이미지는 유사한 content를 갖는다고 가정한다.

따라서, content cost는 아래와 같이 정의될 수 있다.

$ J_{content}(C,G) = \frac{1}{2} {\left \Vert a^{(l)[C]}-a^{(l)[G]} \right \Vert}^2 $



### Style Cost Function

이미지의 style을 어떻게 정의내릴 수 있을까? 논문 [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)에서는 "correlation between activations across channels"라고 정의내린다. 조금 더 상세히 알아보자.

![Imgur](https://i.imgur.com/DKQMbuX.png)

위와 같이 임의의 이미지 혹은 activation map을 conv-layer에 통과시키면 필터 개수만큼의 새로운 activation map이 나온다. 이 activation map을 이용하여, 각 필터를 통해 포착된 부분을 이미지로 복원할 수 있다. 아래는 9개의 필터에서 나온 activation map을 이미지로 복원한 결과이다.

![Imgur](https://i.imgur.com/wtxHkwJ.png)

빨간색으로 표시된 부분에서는 해당 필터가 수직의 직선 모양들을, 파란색 부분에서는 필터가 주황색의 색감을 탐지한다는 것을 유추할 수 있다. 즉, conv-layer를 거친 activation map은 서로 다른 특징을 담고 있고, 이 activation map들의 상관관계(correlation)을 이미지의 style로 정의내리는 것이다. 

그렇다면 채널간 correlation은 어떻게 계산할 수 있을까? 먼저 3d 텐서 형태의 activation 아래와 같이 매트릭스 형태로 만든다. 이를 unroll이라고도 부르는데, 매트릭스를 flatten해서 벡터로 만드는 과정과 비슷하다고 볼 수 있다. 이 매트릭스를 $A$라고 부르자.

![Imgur](https://i.imgur.com/iTma3Gm.png)

채널간 correlation은 $AA^{T}$로 표현된다. 이를 Gram Matrix라 부른다. 

![Imgur](https://i.imgur.com/oqBCAlx.png)

스타일 이미지(S)의 activation을 Gram Matrix로 만든 것을 $S^{(l)[G]}$, 생성 이미지(G)를 같은 방법으로 변환한 것을 $G^{(l)[G]}$라고 부르자. 두 Gram Matrix는 각각 S의 style, G의 style 정보를 담고 있다. 두 매트릭스가 유사하다면, S와 G는 유사한 스타일을 지녔다고 이야기할 수 있다. 이에 따라, style cost는 아래와 같이 정의될 수 있다.

![Imgur](https://i.imgur.com/lTzmUGu.png)



### Generating Image

이미지 생성의 과정은 아래와 같다.

![Imgur](https://i.imgur.com/4poAyuV.png)

먼저, 생성 이미지 G를 랜덤하게 initialize한다. 이후 계산된 cost인 $J(G)$를 backpropagate한다. 생성 이미지 G까지 gradient를 역전파시키고, 이를 이용하여 G를 update한다. 네트워크는 얼려둔 상태에서 G를 직접 update하는 형태다. 

Style cost의 개념은 조금 난해했지만, 구현 자체는 어렵지 않다고 생각했다. 강의에서 제공되는 programming assignment와 [pytorch tutorial](http://pytorch.org/tutorials/advanced/neural_style_tutorial.html)을 참고하여 시도해보았다. 

* content 이미지: 요새 즐겨보는 고양이 유튜브 채널 [크림 히어로즈](https://www.youtube.com/channel/UCmLiSrat4HW2k07ahKEJo4w)의 [루루](https://namu.wiki/w/%ED%81%AC%EB%A6%BC%ED%9E%88%EC%96%B4%EB%A1%9C%EC%A6%88#s-3.3.2) 고양이 사진
* style 이미지: 이중섭 화가의 그림 [흰 소](https://ko.wikipedia.org/wiki/%ED%9D%B0_%EC%86%8C) 
* ConvNet: vgg-19(pretrained with ImageNet)
* hidden layer: conv4_2(10번째 conv-layer)

![Imgur](https://i.imgur.com/H008vTm.png)

그림에 표현된 특유의 굵은 선과 역동적인 느낌이 반영되었으면 하는 바람이었는데, 결과는 아래와 같다. 유화 느낌은 반영이 되었지만, 의도한 느낌은 살리지 못했다. 원인을 알아보기 위해 [원 논문](https://arxiv.org/abs/1508.06576)을 읽어보았더니 style cost를 계산할 때 여러 레이어를 사용할 수록 style이 더 잘 반영되는 것 같다. 강의에서는 편의를 위해 하나의 레이어를 뽑는다고 이야기한 것 같다. 관련 내용은 [여기](http://sanghyukchun.github.io/92/#92-reconst-img)에 잘 설명되어있다.

![Imgur](https://i.imgur.com/Y2qCAAJ.png)

스탠포드 Fei-Fei Li 교수님이 참여한 [논문](https://arxiv.org/abs/1603.08155)에서는 보다 고해상도의 이미지를 빠른 시간에 생성하는 style transfer 방법론을 제시한다. 또한, [fast style transfer](https://github.com/lengstrom/fast-style-transfer) 등 이와 관련된 좋은 소스코드들이 많이 올라와있다. 코드를 아직 제대로 들여다보지 않았으나, 이를 활용하면 보다 좋은 품질의 이미지를 얻을 수 있지만 vgg 네트워크를 처음부터 다시 학습시켜야하는 것 같다. 