---
layout: post
title: Information Theory에 대한 정리
excerpt: "Information theory, entropy 등에 대해 공부한 뒤 간략하게 정리해봤습니다."
categories: [Study]
comments: true
use_math: true
---

## Information Theory

Deep Learning Book Ch.3은 Probability and Information Theory라는 제목입니다. 오늘 발제는 이 중에서도 Information Theory에 대해서 초점을 맞춰서 진행해보려고 합니다. 정보 이론은 말 그대로, 정보라는 개념을 구체화/수치화했다고 생각하시면 됩니다.  이때 가장 기본적인 개념은 드물게 일어나는 사건일수록, 자주 일어나는 사건에 비해 더 많은 정보를 담고 있다는 것입니다. 예를 들어 생각해보자면, "오늘 아침에 해가 떴다"와 "오늘 아침 달이 떴다"라는 정보가 있다고 해봅시다. 전자의 경우에는 너무 당연해서 제가 여러분들 앞에서 "오늘 아침에 봤더니 해가 떴더라"하면 저를 미친놈 취급할테지만, "오늘 아침에 봤더니 달이 떠있더라"라고 하면 "아 진짜?"정도의 반응은 나올 겁니다. 이처럼 자주 일어나는 사건은 의미가 별로 없지만, 드물게 일어날수록 의미가 커집니다.

이 개념들을 수학적으로 끌고 나간다고 생각해봅시다. 세 가지 정도로 정리할 수 있습니다.

- 자주 일어나는 사건일수록 더 적은 정보량을 가집니다. 정말정말 극단적인 경우, 무조건 일어나는 사건(확률이 1인 경우)에는 정보량이 0입니다.
- 드물게 발생하는 사건일수록 더 높은 정보량을 갖습니다.
- 독립적인 사건들은 additive information을 가집니다. 동전을 두 번 던져 둘 다 앞면이 나왔다고 해봅시다. 이 정보량은 동전을 한번 던져 앞면이 나온 정보량의 두 배여야합니다.

이 세 가지 특성을 따라 사건 $\mathrm{x} = x$의 self-information을 정의합시다.

$$ I(x) = -\log{P(x)} $$

이 self-information을 정말 간단하게 정의한다면 "놀람의 정도, degree of surprise"라고 할 수 있습니다. 제가 올해도 연애를 못할 확률을 0.95라고 해봅시다. 그렇다면 이들 각각에 담긴 정보량을 구해보면 아래와 같을 겁니다.

- 올해도 연애를 못했다면: -log(0.95) =  0.0512932...
- 올해에 연애를 했다면: -log(0.05) = 2.9957...

곧 제가 연애를 했다는 소식을 들으면, 여러분들은 연애를 못했을 때에 비해 대략 58배 정도 놀라는겁니다. 이때 self-information은 한 개의 사건만을 다룹니다 우리가 더 관심있을만한 부분은 전체 확률 분포에서의 정보량이겠죠. 이를 Shannon entropy라고 합니다.

$$H(\mathrm{x}) = \mathbb{E}_{x\sim P}\left[ I(x) \right] = -\mathbb{E}_{x\sim P}\left[ \log{P(x)}\right] = \sum{P(x)I(x)}$$

섀넌 엔트로피는 결국 어떤 분포에서 사건이 일어났을 때, 이 사건이 가지는 정보량의 기대값입니다. 그렇다면 제 연애의 평균적인 정보량, 섀넌 엔트로피는 0.95 * -log(0.95) + 0.05 * -log(0.05) = 0.198정도 입니다. 만약 확률분포가 deterministic, 즉 확률분포에서 어떤 사건이 나올지가 분명하다면 낮은 엔트로피를 가지고, uniform distribution에 가까울수록 높은 엔트로피를 가집니다. 동전이 2개가 있다고 해봅시다. 한 동전을 던질 떄는 무조건 앞면이 나온다고 했을 때, 이 동전의 엔트로피는 1 * -log(1) + 0 * -log(0)이므로 0이 됩니다. 다른 동전을 던졌을 때에는 앞면과 뒷면이 나올 확률이 동일하다고 한다면, 이 때의 엔트로피는 0.693정도가 됩니다.

같은 확률변수 $\mathrm{x}$에 대한 2개의 확률분포 $P(\mathrm{x})$과 $Q(\mathrm{x})$가 있다고 해봅시다. Kullback-Leibler Divergence, KL Divergence, 혹은 KLD은 이 두 분포사이의 차이를 의미합니다.

$$D_{KL}(P \| Q) = \mathbb{E}_{x\sim P} \left[\log{\frac{P(x)}{Q(x)}}\right] = \mathbb{E}_{x\sim P}\left[ \log{P(x)} - \log{Q(x)}\right]$$

KLD는 non-negative합니다. 즉 KLD가 0이라면, 두 분포 $P$와 $Q$가 동일하다는 의미입나다(이산형의 경우). 이때 KLD가 0이상의 값을 가지면서 두 분포의 차이를 표현하므로 "거리"를 나타낸다고 생각하실 수도 있겠습니다만. KLD는 symmetric하지 않습니다. 즉 $D_{KL}(P\|Q)$와 $D_{KL}(Q\|P)$는 다를 수도 있습니다.

![image](https://user-images.githubusercontent.com/25279765/37524677-ee355cfe-296d-11e8-9e6e-017ca33e5d22.png)

>p가 두 개의 정규분포, q는 한 개의 정규분포라고 했을 때, q를 KLD를 최소화하도록 학습을 시킨다고 해봅시다. KLD에서 $(P \| Q)$냐 $(Q \| P)$냐에 따라서 최적화된 q*는 달라질 수도 있는 것이죠.

딥러닝 관점에서 생각해보면, 모델은 결국 데이터의 분포를 추정하는 것입니다. 실제 데이터의 분포를 $P(x)$라고 하고, 모델이 추정한 분포를 $Q(x)$라고 하면, KLD는 결국 모델이 예측한 분포와 실제 분포가 얼마나 다른지를 의미한다고 볼 수 있습니다. KLD에서 Cross Entropy라는 것을 정의할 수도 있습니다.

$$H(P, Q) = H(P) + D_{KL}(P \| Q)$$

딥러닝을 통해 classification 문제를 해결하고자 할 때 마지막에 cross entropy를 통해 loss를 구하고 이를 역전파시키는 방식으로 학습을 시키죠. 이때 이는 결국 모델이 추정하는 분포 $Q$가 실제 데이터의 분포 $P$가 유사하도록 최적화하는 것과 동일하다고 볼 수 있습니다. $H(P)$의 경우, 학습이 진행된다고 해도 바뀌지 않는 값이니까요.

$$\begin{eqnarray}
H(P, Q) &=& H(P) + D_{KL}(P \| Q) \\
&=& -\mathbb{E}_{x\sim P} \left[ \log{P(x)} \right] + \mathbb{E}_{x\sim P} \left[ \log{P(x)} - \log{Q(x)} \right] \\
&=& -\mathbb{E}_{x\sim P} \left[\log{Q(x)}\right] \\
&=& -\sum{P(x)\log{Q(x)}} \\
\end{eqnarray}$$

classification의 경우, $P$를 label이라고 볼 수 있겠죠. 옳은 레이블에서만 $P$는 1의 값을 가질겁니다. 우리의 목표는 학습을 통해서 $Q$가 옳은 데이터를 옳다고 분류하기를 원하기에 결국 Cross entropy를 최소화하도록 학습을 시키는 것입니다.

## Reference

[https://ratsgo.github.io/statistics/2017/09/22/information/](https://ratsgo.github.io/statistics/2017/09/22/information/)

[http://blog.naver.com/PostView.nhn?blogId=gyrbsdl18&logNo=221013188633&parentCategoryNo=3&categoryNo=&viewDate=&isShowPopularPosts=true&from=search](http://blog.naver.com/PostView.nhn?blogId=gyrbsdl18&logNo=221013188633&parentCategoryNo=3&categoryNo=&viewDate=&isShowPopularPosts=true&from=search)
