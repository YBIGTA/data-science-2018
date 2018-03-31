# Ch4. Classification

### LDA vs QDA

![LDA QDA에 대한 이미지 검색결과](https://i.imgur.com/QMVFpRl.png)

LDA, QDA 모두 데이터 분포를 학습해 Decision boundary를 만들어 데이터를 분류하는 모델입니다. Decision boundary가 선형이면 LDA, 비선형이라면 QDA라고 부릅니다.

### LDA

![source: imgur.com](http://i.imgur.com/6ggd2F0.png)

**Q.** 위의 그림에서 어떤 축이 분류가 잘 된 축일까요? 

​      **오른쪽**, 두 범주의 *중심(평균)* 이 서로 멀고 각각의 *분산*이 작도록 나누는 직선(초록)

클래스 K의 데이터를 직선(점선)에 사영했을 때의 분포를 $$f_k(x)$$라고 정의합니다.

1개의 predictor를 가지고 sample들을 K개의 class로 나누는 경우를 생각해봅시다.  LDA에서는 모든 클래스의 분산이 같고, $$f_k(x)$$는 정규 분포를 따른다고 가정합니다. 

Bayes classifier  $$Pr(Y=k|X=x) = p_k(x)=\frac{\pi_kf_k(x)}{\sum_{l=1}^{K}\pi_lf_l(x)}$$

사전 확률은 $$\pi_k$$이고 $$f_k(x)$$는 클래스 k의 x에 대한밀도 함수입니다. 새로운 데이터 $$x'$$가 들어 왔을 때 Bayes classifier는 $$p_k(x')$$가 가장 큰 클래스로 $$x'$$ 를 분류합니다.



이제 Bayes classifier을 이용해서 LDA를 해봅시다. 

LDA의 목적은 다음과 같은 데이터의 분포가 있을 때, 새로운 데이터 x가 들어왔을 경우 X를 어떤 class에 넣을 것인지를 결정하는 경계를 찾는 것입니다. Decision boundary를 나타내는 직선의 방정식을 찾는 것은 아닙니다. '다음에 들어올 데이터가 어떻게 분류될 지를 기준으로, 현재의 데이터를 구역화 하는 함수를 찾는 방법인데, 하필 그 경계가 직선이다 '라고 이해하는 편이 더 자연스러울 것 같습니다.

새로운 x가 들어왔을 때 class k에 x가 assign될 확률은 $$p_k(x)=\frac{\pi_kf_k(x)}{\sum_{l=1}^{K}f_l(x)}=\frac{\pi_k\frac{1}{\sqrt{2\pi}\sigma}{exp(-\frac{1}{2\sigma^2}(x-\mu_k)^2})}{\sum_{l=1}^{K}\pi_l\frac{1}{\sqrt{2\pi}\sigma}{exp(-\frac{1}{2\sigma^2}(x-\mu_l)^2})}$$

입니다. ($$\because$$ $$f_k(x)$$는 정규분포)

$$\mu_k,\sigma$$는 모수이므로 다음과 같이 $$\hat{\mu_k},\hat{\sigma}$$로 추정합니다. 

![img](https://i.imgur.com/BzWeW2Y.png)

$$\pi_k$$는 class k에 대한 사전 확률인데, 데이터의 분포를 모르는 상황에서 class k에 대한 사전 확률을 구할 수 없으므로 추정값을 사용합니다.   $$\hat{\pi}_k$$는 class k에 대한 사전확률의 추정값이고, 전체 데이터 갯수 대비 class k에 속하는 데이터 갯수의 비율을 의미합니다.  아래의 그림에서 $$\hat{\pi}_{W1}$$은 넣어준 데이터 X에서 W1에 속하는 데이터의 비율일 것입니다.![source: imgur.com](http://i.imgur.com/VUhETyK.png)

Bayesian classification rule에 따라  $p_k(x)$를 가장 크게 만드는 k에 x를 assign합니다. 양변에 로그를 취해 정리해주었을 때 최대가 되게 하려면 다음 식

$$\delta_k(x)=x \centerdot \frac{\hat{\mu_k}}{\hat{\sigma}^2}- \frac{\hat{\mu_k}^2}{\hat{2\sigma}^2}+log(\hat{\pi_k})$$

이 최대가 되게 만들어주는 것과 같습니다. 이 경계를 따라 class k의 범위가 나뉘는데 경계는 직선입니다.

마찬가지로 predictor가 2개 이상인 경우도 비슷하게 진행되는데, 이 때는 각각의 predictor x에 대한 밀도 함수가 multivariate Gaussian distribution을 따르고 모든 클래스에 대해 각각의 평균값이 존재하며 공통의 공분산(covariance)를 가진다는 가정이 더 필요합니다.

![img](https://i.imgur.com/7LTVZcN.png)

참조: https://ratsgo.github.io/machine%20learning/017/03/21/LDA/



### LAB

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.metrics import confusion_matrix, classification_report, precision_score
```

sklearn.metrics 모듈은 classification performance의 척도들을 계산해줍니다. Classification을 하면서 precsion score, confusion matrix를 쉽게 구할 수 있습니다. classification_report는

cf)confusion matrix

![img](http://cfile7.uf.tistory.com/image/26523F46590C1F0B033019)

우선 LDA 모델을 만들어 prediction을 해봅시다.

```python
#2개의 class로 나누는 LDA
lda = LinearDiscriminantAnalysis()
pred = lda.fit(X_train, y_train).predict(X_test)
```

parameter를 살펴 보면 다음과 같습니다.

**solver** : string, optional

> ‘svd’: Singular value decomposition (default). classification과 차원 축소 둘 다에 사용 가능. covariance matrix를 사용하지 않기 때문에 feature가 많을 때 사용
>
> 'lsqr': Least squares solution, 최소 제곱해를 이용한다. classification에만 사용 가능. shirinkage를 쓸 수 있다
>
> 'eigen': Eigenvalue decomposition, classification과 차원 축소 둘 다에 사용 가능. shirinkage를 쓸 수 있다. covariance matrix를 이용하기 때문에 feature가 많은 경우 적합하지 않다.

**shrinkage** : string or float, optional

> - None: no shrinkage (default).
>
> - ‘auto’: automatic shrinkage using the Ledoit-Wolf lemma.
>
> - float between 0 and 1: fixed shrinkage parameter.
>
>   covariance matrix에 의한 영향을 조절할 때 shrinkage쓸 수 있다.
>
>   ‘lsqr’ and ‘eigen’ solvers에서만 사용

**priors** : array, optional, shape (n_classes,)

> 만약 사전 확률이 있다면 여기에 넣어주면 됨

**n_components** : int, optional

> 차원 축소 시 몇 개의 dimension으로 축소할 건 지 결정 

**store_covariance** : bool, optional

> class covariance matrix를 추가적으로 계산, ‘svd’ solver에서만 사용.

**tol** : float, optional, (default 1.0e-4)

> Rank estimation threshold, SVD solver에서만 사용.

![img](https://i.imgur.com/568Q2MY.png)

classification이 잘 됐는 지 확인하기 위해 classification report를 돌려 보면

```python
classification_report(y_test, pred, digits=3, labels=['Down','Up'])
```

![img](https://i.imgur.com/iCj1kkd.png)

*precision*은 classifer가 positive sample을 negative로 labeling하지 않을 능력을 나타내는 수치, *recall*은 classifier가 모든 positive sample을 찾을 수 있는 지를 나타내는 수치입니다. 더 자세한 건 http://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html#sklearn.metrics.precision_recall_fscore_support 참조.

