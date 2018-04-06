---
layout: post
title: "ISL_Chapter5_Lab"
author: "hyeju.kim"
categories: ML
tags: [ISL]
image: k fold cv.jpg
---

# Chapter 5 - Resampling Methods LAB by hyeju.kim

참고 사이트: 
- https://people.duke.edu/~ccc14/sta-663/ResamplingAndMonteCarloSimulations.html
- http://scikit-learn.org/stable/modules/cross_validation.html


k-fold cross validation을 실습해보면서 원리를 이해해보는 것이 목표!

- [Load dataset](#Load-dataset)


- **[k-fold Cross-Validation](#5.1-k-fold-Cross-Validation)**

    - [Details of k-fold CV](#5.1.1-k-fold-CV-가-어떻게-작동하는지-보자)

    - [mse change with iterations](#5.1.2-각-iteration별로-mse-측정)

    - [how to choose k?](#5.1.3--How-to-choose-k?)

    - [Repeated K-Fold CV](#5.1.4-Repeated-K-Fold)   

- **[bootstrap](#5.2-bootstrap)** bootstrap 관련 실습은 sample 만들어보는 것만 visualization 해보고 생략...관심 있으신 분들은 자료 참고!

    - [Bootstrap Samples Visualization](#5.2.1-visualizing-bootstrap-samples)

    - [참고자료 링크](#5.2.2-bootstrap-관련-자료)



```python
# %load ../standard_import.txt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn.linear_model as skl_lm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, LeaveOneOut, KFold, cross_val_score
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm
import statsmodels.formula.api as smf


%matplotlib inline
plt.style.use('seaborn-white')
```

    D:\Anaconda\lib\site-packages\statsmodels\compat\pandas.py:56: FutureWarning: The pandas.core.datetools module is deprecated and will be removed in a future version. Please use the pandas.tseries module instead.
      from pandas.core import datetools


### Load dataset

Dataset available on http://www-bcf.usc.edu/~gareth/ISL/data.html


```python
df1 = pd.read_csv('Data/Auto.csv', na_values='?').dropna()
df1.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 392 entries, 0 to 396
    Data columns (total 9 columns):
    mpg             392 non-null float64
    cylinders       392 non-null int64
    displacement    392 non-null float64
    horsepower      392 non-null float64
    weight          392 non-null int64
    acceleration    392 non-null float64
    year            392 non-null int64
    origin          392 non-null int64
    name            392 non-null object
    dtypes: float64(4), int64(4), object(1)
    memory usage: 30.6+ KB



```python
len(df1)
```




    392




```python
df1.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }
    
    .dataframe thead th {
        text-align: left;
    }
    
    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mpg</th>
      <th>cylinders</th>
      <th>displacement</th>
      <th>horsepower</th>
      <th>weight</th>
      <th>acceleration</th>
      <th>year</th>
      <th>origin</th>
      <th>name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>18.0</td>
      <td>8</td>
      <td>307.0</td>
      <td>130.0</td>
      <td>3504</td>
      <td>12.0</td>
      <td>70</td>
      <td>1</td>
      <td>chevrolet chevelle malibu</td>
    </tr>
    <tr>
      <th>1</th>
      <td>15.0</td>
      <td>8</td>
      <td>350.0</td>
      <td>165.0</td>
      <td>3693</td>
      <td>11.5</td>
      <td>70</td>
      <td>1</td>
      <td>buick skylark 320</td>
    </tr>
    <tr>
      <th>2</th>
      <td>18.0</td>
      <td>8</td>
      <td>318.0</td>
      <td>150.0</td>
      <td>3436</td>
      <td>11.0</td>
      <td>70</td>
      <td>1</td>
      <td>plymouth satellite</td>
    </tr>
    <tr>
      <th>3</th>
      <td>16.0</td>
      <td>8</td>
      <td>304.0</td>
      <td>150.0</td>
      <td>3433</td>
      <td>12.0</td>
      <td>70</td>
      <td>1</td>
      <td>amc rebel sst</td>
    </tr>
    <tr>
      <th>4</th>
      <td>17.0</td>
      <td>8</td>
      <td>302.0</td>
      <td>140.0</td>
      <td>3449</td>
      <td>10.5</td>
      <td>70</td>
      <td>1</td>
      <td>ford torino</td>
    </tr>
  </tbody>
</table>
</div>



## 5.1 k-fold Cross-Validation

### 5.1.1 k-fold CV 가 어떻게 작동하는지 보자

*sklearn.model_selection.KFold(n_splits=3, shuffle=False, random_state=None)*

- n_splits : fold의 개수. 만약 n_splits=10이면 10fold가 있고, 1개가 training set 나머지 9개가 test set으로 쓰이는 것이다.
- shuffle: 나누기 전에 shuffle할 건지 말 건지 선택.데이터 순서가 임의적이지 않을 경우 shuffle =True로 하기도 한다. 하지만 샘플들이 iid(independently and identicallly distributed) 하지 않다면 shuffling하지 않는 것이 오히려 낫다.(예.순서형 자료)
- random_state: int를 부여할 경우 seed와 같은 역할을 해서 shuffle을 동일하게 한다. 'RandomState instance'일 경우 random number를 생성한다. 'None'일 경우 np.random을 사용하여 random number를 생성한다. shuffle=True일 경우에 사용되는 파라미터이다. 


*Methods*

- get_n_splits([X, y, groups]): split 개수를 알려준다. n_splits에서 설정한 파라미터와 동일

  - split(X[, y, groups]): split되는 index를 알려준다. 자세한 것 밑에서 살펴보자. 

먼저 간단한 예제를 통해 k fold cv를 사용한 sample들의 모습을 살펴봅시다. 

원리를 이해하는 용도로 간단한 데이터를 보자!

y 예측변수는 1차원 데이터이므로 다음과 같은 형태를 가지고 있을 것이다.

**[1, 2, 3, 4]** 데이터의 경우


```python
from sklearn.model_selection import KFold
y = np.array([1, 2, 3, 4])
kf = KFold(n_splits=4)

print(kf)  

for train_index, test_index in kf.split(y):
    print("TRAIN:", train_index, "TEST:", test_index)
    y_train, y_test = y[train_index], y[test_index]

    print("y_train:",y_train)
    print("y_test:",y_test)
```

    KFold(n_splits=4, random_state=None, shuffle=False)
    TRAIN: [1 2 3] TEST: [0]
    y_train: [2 3 4]
    y_test: [1]
    TRAIN: [0 2 3] TEST: [1]
    y_train: [1 3 4]
    y_test: [2]
    TRAIN: [0 1 3] TEST: [2]
    y_train: [1 2 4]
    y_test: [3]
    TRAIN: [0 1 2] TEST: [3]
    y_train: [1 2 3]
    y_test: [4]


예측변수는 dimension이 1만 아니라 그 이상인 경우도 있다. 다음과 같은 데이터는 예측변수가 2개이고, observation이 4개인 경우이다.

**[[1, 2], [3, 4], [5, 6], [7, 8]]** 데이터
- n_splits=2의 경우 

sample을 두 개로 나누어 cross validation이 이루어진다.


```python
from sklearn.model_selection import KFold
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
kf = KFold(n_splits=2)
kf.get_n_splits(X)

print(kf)  

for train_index, test_index in kf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]

    print("X_train:",X_train)
    print("X_test:",X_test)
```

    KFold(n_splits=2, random_state=None, shuffle=False)
    TRAIN: [2 3] TEST: [0 1]
    X_train: [[5 6]
     [7 8]]
    X_test: [[1 2]
     [3 4]]
    TRAIN: [0 1] TEST: [2 3]
    X_train: [[1 2]
     [3 4]]
    X_test: [[5 6]
     [7 8]]


- n_splits=4의 경우

sample이 4개로 나누어져서, 3개는 train set, 1개는 test set으로 역할을 한다.

총 4번의 iteration이 이루어진다.


```python
from sklearn.model_selection import KFold
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
kf = KFold(n_splits=4)
kf.get_n_splits(X)

print(kf)  

for train_index, test_index in kf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    
    print("X_train:",X_train)
    print("X_test:",X_test)

```

    KFold(n_splits=4, random_state=None, shuffle=False)
    TRAIN: [1 2 3] TEST: [0]
    X_train: [[3 4]
     [5 6]
     [7 8]]
    X_test: [[1 2]]
    TRAIN: [0 2 3] TEST: [1]
    X_train: [[1 2]
     [5 6]
     [7 8]]
    X_test: [[3 4]]
    TRAIN: [0 1 3] TEST: [2]
    X_train: [[1 2]
     [3 4]
     [7 8]]
    X_test: [[5 6]]
    TRAIN: [0 1 2] TEST: [3]
    X_train: [[1 2]
     [3 4]
     [5 6]]
    X_test: [[7 8]]


### 5.1.2 각 iteration별로 mse 측정

**auto.csv에 적용한 후, 각 iteration별로 score(mse)를 측정해보자**

사실 iteration별로 mse가 어떻게 되는지 알 필요는 없지만, 최종 mse가 어떻게 계산되는지 궁금해서 mse_list를 만들고, 그것을 평균내 최종 mse를 구해본다.


```python
mse_list = []

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
#x variable은 shape이 (-1, p)가 되도록. p= # of input variables
#y variable의 shape을 (-1,1)로 reshape해도 되지만, 여기서는 각각 lm model을 돌려볼 거라 우선 그냥 형태를 그대로 하였다.
X = df1[['horsepower','weight']].values.reshape(-1,2)
y = df1.mpg.values


kf = KFold(n_splits=10)
for train_index, test_index in kf.split(X):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    #print("X_train:",X_train)
    #print("X_test:",X_test)
    #print("y_train:",y_train)
    #print("y_test:",y_test)
    
    #iteration마다 train set을 바탕으로 lm model을 돌린다.
    regr = skl_lm.LinearRegression()
    regr.fit(X_train,y_train)
    y_test_predict = regr.predict(X_test)
    #print(y_test_predict)
    
    #mse를 iteration마다 구한다.
    mse = mean_squared_error(y_test, y_test_predict, multioutput='uniform_average')
    mse_list.append(mse)
    print(mse_list)
    
#mse의 평균은?    
print(sum(mse_list)/len(mse_list))
```

    [12.089761487852584]
    [12.089761487852584, 18.989870878233752]
    [12.089761487852584, 18.989870878233752, 25.433070564663261]
    [12.089761487852584, 18.989870878233752, 25.433070564663261, 13.780969891433378]
    [12.089761487852584, 18.989870878233752, 25.433070564663261, 13.780969891433378, 9.1564195036570766]
    [12.089761487852584, 18.989870878233752, 25.433070564663261, 13.780969891433378, 9.1564195036570766, 6.608754507969457]
    [12.089761487852584, 18.989870878233752, 25.433070564663261, 13.780969891433378, 9.1564195036570766, 6.608754507969457, 11.736713758602779]
    [12.089761487852584, 18.989870878233752, 25.433070564663261, 13.780969891433378, 9.1564195036570766, 6.608754507969457, 11.736713758602779, 16.189583024974194]
    [12.089761487852584, 18.989870878233752, 25.433070564663261, 13.780969891433378, 9.1564195036570766, 6.608754507969457, 11.736713758602779, 16.189583024974194, 57.106389348821466]
    [12.089761487852584, 18.989870878233752, 25.433070564663261, 13.780969891433378, 9.1564195036570766, 6.608754507969457, 11.736713758602779, 16.189583024974194, 57.106389348821466, 35.20884107707159]
    20.6300374043


**iteration별로 mse의 추이를 그래프로 살펴보자.**


```python
fig, ax1 = plt.subplots(1,1, figsize=(10,4))

X = range(1,11)
Z = mse_list
# Right plot (all splits)
ax1.plot(X,Z)
ax1.set_title('10-fold cv mse')


for ax in fig.axes:
    ax.set_ylabel('Mean Squared Error')
    ax.set_xlabel('iteration')
    ax.set_xlim(0.5,10.5)
    ax.set_xticks(range(2,11,2));
```

![isl-chapter5-lab-hyeju kim_26_0](https://user-images.githubusercontent.com/32008883/38425004-b45660c4-39ed-11e8-861c-66b464e0fd01.png)




**사실 이 모든 과정을 한꺼번에 할 수 있다. 다음 3줄로!**


```python
X = df1[['horsepower','weight']].values.reshape(-1,2)
regr = skl_lm.LinearRegression()
cross_val_score(regr, X, df1.mpg, scoring='neg_mean_squared_error', cv=10).mean()*-1
```




    20.630037404327954



위에서 mse를 평균낸 값과 같음을 알 수 있다.

cross_val_score의 parameter중 scoring parameter는 어떤 것으로 accuracy를 측정할지 결정하는 것인데, regressor의 경우 r2 score이다.

여기서는 mse를 보기 위해 scoring을 상단과 같이 설정하였다.

참고: http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter

### 5.1.3  How to choose k?

여기에서 궁금한 점은, 어떻게 k를 선택하냐이다. 

k가 크면 클수록 bias는 줄어든다. 하지만 역시 bias-variance trade off 문제가 있다.

k = N(sample size)이면 Leave-One-Out Cross-Validation (LOOCV)로, 

가장 bias가 작지만 computational disadvantage와 high correlation 문제가 있다.

대개 K = 5, 10 or 20 정도로 쓰면 컴퓨테이션 cost를 줄이면서 LOOCV와 비슷하게 성능을 낼 수 있다.

그래서 k fold로 주어진 데이터 셋에 실험해보기로 하자. 

출처: 
http://vinhkhuc.github.io/2015/03/01/how-many-folds-for-cross-validation.html

-1은 Leave-One-Out을 뜻한다.


```python
from sklearn.cross_validation import cross_val_score, KFold, LeavePOut
import matplotlib.pyplot as plt

#output_file = "cross-validation-experiment-iris.png"


X    = df1[['horsepower','weight']].values.reshape(-1,2)
y    = df1.mpg.values
regr = skl_lm.LinearRegression()
n    = X.shape[0]

folds      = [2, 5, 10, 20, -1]
n_folds    = len(folds)
accuracies = []

# Run K-folds
for k in folds:
    cv     = KFold(n, n_folds=k) if k > 0 else LeavePOut(n, p=abs(k))
    scores = cross_val_score(regr, X, y, cv=cv)
    accuracies.append(100 * scores.mean())
    print("K = %d, accuracy: %0.2f%%" % (k, accuracies[-1]))

# Print chart
plt.figure()
plt.errorbar(range(1, n_folds + 1), accuracies, yerr=[5] * n_folds)  # Use 5% for the error bars
ax = plt.gca()
plt.xticks(range(0, n_folds + 2), [''] + [str(k) for k in folds] + [''])
plt.yticks(range(30, 110, 10))
plt.title("K-fold Cross-validation")
plt.xlabel("Folds")
plt.ylabel("% Acc")
#plt.savefig(output_file)
#print("Saved the chart into " + output_file)
```

    K = 2, accuracy: 11.71%
    K = 5, accuracy: 33.23%
    K = 10, accuracy: 39.69%
    K = 20, accuracy: 33.89%
    K = -1, accuracy: 0.00%





    Text(0,0.5,'% Acc')




![isl-chapter5-lab-hyeju kim_34_2](https://user-images.githubusercontent.com/32008883/38425046-c8d7da5a-39ed-11e8-9667-19c29cf112f8.png)




물론 주어진 데이터셋에 임시로 linear regression을 돌려서 accuracy가 높지는 않다. 

하단의 그래프는 주어진 링크에서 Iris data set으로 SVM을 적용한 결과이다.

이 경우 LOOCV와 K=10인 경우 accuracy가 비슷해서 결국 k=10을 써도 좋은 예측력을 보인다. 

![cross-validation-experiment-iris](https://user-images.githubusercontent.com/32008883/38423348-976e28b6-39e8-11e8-90db-119372426915.png)


몇 개의 추가적인 팁은 다음과 같다.

- k가 크면, less bias이지만, high variance와 high running time을 가진다. 
- 하지만 variance을 낮추기 위해 large k를 쓰기보다 CV를 반복하는 것도 좋은 방법이다. 예를 들어 1000-fold CV보다 100 * 10-fold CV를 하는 것이다.
- k를 정할 때 가능하다면 sample size의 인수를 쓰거나, stratified된 sample의 경우 sample의 group size을 사용하면 좋다.
- 너무 큰 k는 다른 iteration의 개수를 줄이며 sample combination의 개수를 줄인다. 
- 상단의 팁들은 small sample size일 경우 더 유효하다. sample size가 크면 k가 커져도 겹치는 iteration이 줄어든다. 

출처: https://stats.stackexchange.com/questions/27730/choice-of-k-in-k-fold-cross-validation

### 5.1.4 Repeated K-Fold

위에서 cv반복이 나왔으니 이를 구현할 수 있는 함수를 살펴보자.

rkf = RepeatedKFold(n_splits=5, n_repeats=2)

이 부분만 다르고 위와 동일하다.


```python
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_squared_error

mse_list = []

X = df1[['horsepower','weight']].values.reshape(-1,2)
y = df1.mpg.values

rkf = RepeatedKFold(n_splits=5, n_repeats=2)

for train_index, test_index in rkf.split(X):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    #print("X_train:",X_train)
    #print("X_test:",X_test)
    #print("y_train:",y_train)
    #print("y_test:",y_test)
    
    #iteration마다 train set을 바탕으로 lm model을 돌린다.
    regr = skl_lm.LinearRegression()
    regr.fit(X_train,y_train)
    y_test_predict = regr.predict(X_test)
    #print(y_test_predict)
    
    #mse를 iteration마다 구한다.
    mse = mean_squared_error(y_test, y_test_predict, multioutput='uniform_average')
    mse_list.append(mse)
    
print(mse_list)    
#mse의 평균은?    
print(sum(mse_list)/len(mse_list))






```

    [23.003492999491439, 14.433054399557268, 16.644418290134404, 20.267623228761789, 17.441707285492864, 16.767499787704033, 17.883332137030589, 18.543281941163048, 24.489271726865883, 12.664807962261417]
    18.2138489758


두 번 반복해서 cv를 시행했더니 mse가 낮아짐을 알 수 있다.

## 5.2 bootstrap

### 5.2.1 visualizing bootstrap samples

bootstrap한 sample들을 시각화하기 위해 df1.weight을 사용했다.

파란색 선이 원래 weight의 cdf를 나타낸 것이며, 

회색 선은 bootstrap sample들의 weight을 겹쳐 나타낸 것이다.


```python
def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""

    # Number of data points: n
    n = len(data)

    # x-data for the ECDF: x
    x = np.sort(data)

    # y-data for the ECDF: y
    y = np.arange(1, n+1) / n

    return x, y

```


```python
X = df1.weight.values

for _ in range(50):
    # Generate bootstrap sample: bs_sample
    bs_sample = np.random.choice(X, size=len(X))

    # Compute and plot ECDF from bootstrap sample
    x, y = ecdf(bs_sample)
    _ = plt.plot(x, y, marker='.', linestyle='none',
                 color='gray', alpha=0.1)

# Compute and plot ECDF from original data
x, y = ecdf(X)
_ = plt.plot(x, y, marker='.')

# Make margins and label axes
plt.margins(0.02)
_ = plt.xlabel('weight')
_ = plt.ylabel('ECDF')

# Show the plot
plt.show()

```

![isl-chapter5-lab-hyeju kim_46_0](https://user-images.githubusercontent.com/32008883/38425098-e80cb134-39ed-11e8-90fc-09706ecad92d.png)




### 5.2.2 bootstrap 관련 자료

bootstrap이 본래의 분포를 모를 때 standard error등을 추정하는데 쓰인다고 한다. 이에 관련해서는 하단의 논문을 보면 좋을 것 같다...

- 부트스트랩 실습 : https://campus.datacamp.com/courses/statistical-thinking-in-python-part-2/bootstrap-confidence-intervals?ex=4#skiponboarding
- 부트스트랩 coefficient estimate 에 관련된 논문 : https://socialsciences.mcmaster.ca/jfox/Books/Companion/appendix/Appendix-Bootstrapping.pdf

