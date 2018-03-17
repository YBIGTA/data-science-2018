

# Chapter2 Exercises(Conceptual)

1. (a) flexible better. n이 크고 p가 작아서 overfitting 위험성이 작기 때문이다.

   (b) flexible worse. 예측변수는 많은데, n이 적으면 overfitting될 수 있다.

   (c) flexible better. flexible methods가 non-linear relationship을 더 잘 보여준다.

   (d) flexible worse. 오차항의 분산이 크면 유연한 모델은 noise에 적합하고 variance를 크게 한다.

   ​



2. (a)

   - regression
   - inference
   - target : CEO salary


   - n = 500 firms in the US

   - p = profit, number of employees, industry

     ​

   (b) 

   - classification

   - prediction

   - target : a success or failure of a new product

   - n= 20 similar products

   - p = price charged for the product, marketing budget, competition price, and ten other variables.

     ​

   (c)

   - regression
   - prediction
   - target : the % change in the US dollar in relation to the weekly changes in the world stock markets.
   - n= weekly data(52 weeks) for all of 2012.
   - p = the % change in the US market, the % change in the British market, and the % change in the German market.





3. ​



(a) figure 2.12 + figure 2.17

![img](https://user-images.githubusercontent.com/32008883/37323314-3d5cb034-26c6-11e8-8c27-32a0d31f3ab0.jpg)



(b) 

​	1) bias : flexibility가 증가하면 bias가 감소한다.

​	2) variance : flexibility가 증가하면 variance는 증가한다. bias-variance trade off를 잘 보여주고 있다.

​	3) test error : bias와 variance, var($\epsilon$)를 합친 모양의 곡선이다.   

​	4) trarinig error : flexibility 가 증가할 수록 training error가 감소한다. irreducible error보다 작아질 때는 overfitting문제가 발생한다고 볼 수 있다.

​	5) bayes error:  var($\epsilon$) 와 같으므로 test error의 lower limit이다. 

4. (a) classification

   - 통장잔고, 신용등급, 연간 소득을 토대로 신용불량자 예측. 예측이 목표임. 
   -  매출액, 역과의 거리, 존속기간을토대로 앞으로의 폐업여부 결정, 예측
   - 52주 최고및 최저가, 20일 단기선 등을 토대로, 주식 등락 예측

   (b) regression

   - 사장의연봉 예측. 수익, 직원수,산업계에서의 평균연봉 
   - 원달러 환율 변화율 예측. 독일, 미국, 영국, 시장에서의원화가치 
   - 몸무게 예측. 체지방량, 체중, 부모님의 몸무게

   (c) clustering 

   - 영화장르 구분 
   - 고객군 나누기 
   - 암 종류 나누기



5.  flexible method와 inflexible method의 장단점을 서술하고 어떠한 경우에 좋은지 말하여라

1) bias & variance :  flexible method의 경우 non-linear model에 더 적합이 잘되며, bias가 작아지게 한다. 하지만 더 많은parameter를 추정해야하며, overfitting의 문제가 발생할 수 있다. 즉, variance는 커진다.

2) purpose(prediciton/inference) :  flexible method의 경우  예측에 유리하고,  inflexible method의 경우 예측보다는 해석을
할 경우에 좋다.



6.

1) paramatic metohds 는 먼저, f의 함수식에 대한 가정을 만든다. 그리고 이 모델을 fitting시키거나 training시킨다.

- 장점 : parameter  에 대한 추정을 하는 것이므로 f  자체를 추정하는 것보다 간단하다. non-parametic보다 많은 n을 필요로 하지 않는다. 
- 단점 :  모델이 true f와 같지 않을 수 있다. 또한 너무 복잡하게 모델을 세우면 overfitting문제가 발생할 수 있다.

2) non-parametic methods 는 f 의 함수 형태에 대한 명확한 가정을 하지 않는다. 너무 구불구불하지 않게 f를 추정하는 것이 목표이다. 

- 장점 : f에 대한 특정한 함수 형태에 대한 가정이 없기 때문에 더 많은 모양의 f들을 포용할 수 있다. 
- 단점 :  많은 n을 필요로 한다.



7.

​	(a) 

1) $\sqrt{9} = 3$

2) $\sqrt{4} = 2$

3) $\sqrt{1+9} = \sqrt{10}$

4) $\sqrt{1+4} = \sqrt{5}$

5) $\sqrt{1+1} = \sqrt{2}$

6) $\sqrt{1+1+1} = \sqrt{3}$ 



​	(b) 5th 관측치에 가장 가깝기 때문에 Green class에 속한다.

​	

​	(c) 가장 가까운 3개의 관측치는 2,5,6이다. Green class에 대한 추정된 확률은 1/3이고 Red class는 2/3이다. 따라서 KNN classifier는 test 관측치에 대하여 Red class에 속할 것이라 결론 내릴 것이다.

​	(d) small. K가 커지면, classifier는 덜 유연해지며 linear에 가까운 경계를 만들고자 한다. 그래서, 만약 non linear 문제라면 K가 작아야 한다.



