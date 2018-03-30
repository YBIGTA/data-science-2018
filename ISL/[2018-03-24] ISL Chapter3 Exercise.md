1. TV ($$\beta_1$$),radio($$\beta_2$$), newspaper($$\beta_3$$)에 대한 nullhypothesis는 각각$$\beta_1 = 0, \beta_2 = 0, \beta_3=0$$으로TV, newspaper, radio는 sales에 영향을주지 않는다는 것을 의미한다.  TV와radio는 p-value가 작으므로 null hypothesis를 기각할 수 있어 radio와 TV가 sales에 영향을 준다는 의미이다. 반면 newspaper는 p-value가 크므로 null hypothesis를 기각할 수 없고 이는 newspaper가 sales에 영향을 미치지 않는다는 이야기이다.

2. KNN regression model은 KNN classifier와 비슷하지만 KNN classifier는 probability가 가장 높은 class에 test observation을 배정하고, KNN regression model은 가장 인접한 K 개의 observation을 평균내어 response를 예측한다./ 방법은 같으나 target이 양적(quantitative)이냐 양적(qualitative)이냐에 따라 다르다.

3. Predictor: $$X_1 = GPA, X_2 = IQ,X_3=Gender, X_4 = Interaction between GPA and IQ, X_5 = Interaction between GPAand gender$$

   Response: starting salary

   (a)  iii) IQ와 GPA가 고정 값일 때 female이면 $$\beta_5$$가 음수이므로 male이 salary가 더 높다. $$X_5=GPA * Gender$$이므로GPA가 충분히 클 때만 $$\beta_5$$가 response에 반영된다.

   (b)  $$50 + 20*4.0 + 0.07*110 + 35 +(4.0*110)*0.01 + (4.0*10)*(-10)$$

   (c)  False, p-value를 이용해 유의미한변수 인 지 확인해 봐야한다.

4. (a) Linear regression의 RSS > Cubic regression의 RSS :flexibility가 Cubic regression이 더 높다

   (b) Linear regression의 RSS < Cubic regression의 RSS: flexibility가 더 높으면 test error는 커지는 경향이 있다.

   (c) Linear regression의 RSS > Cubic regression의 RSS: flexibility 와 상관없이 항상 cubic regression의 training RSS가 더 낮다.

   (d) Linear regression의 RSS < Cubic regression의 RSS: information이 부족하다. 왜냐하면 linear보다 cubic에 가까울수록 cubic regression의 test RSS가 linear regression보다 낮을 수 있기 때문이다. 하지만 linear에 가깝다면 linear regression의 test RSS가 더 낮다.

5. $$\hat{y_i}=\hat{\beta}x_i = \frac{\sum_{i = 1}^nx_iy_i}{\sum_{k = 1}^nx_{k }^2}x_i=\frac{\sum_{j = 1}^nx_ix_jy_j}{\sum_{k = 1}^nx_{k }^2}=\sum_{j = 1}^n\frac{x_ix_j}{\sum_{k = 1}^nx_{k }^2}y_j$$

6. $$y=\hat{\beta_0}+\hat{\beta_1}x$$ 

   $$\hat{\beta_0}=\overline{y}-\hat{\beta_1}\overline{x}$$ by p.62 3.4이므로 $$y=\overline{y}-\hat{\beta_1}\overline{x}+\hat{\beta_1}x$$이다.  $$\overline{x}$$을 식에 넣으면 $$y=\overline{y}-\hat{\beta_1}\overline{x}+\hat{\beta_1}\overline{x}=\overline{y}$$

   따라서 위 식은 $$\overline{x},\overline{y}$$를 지난다.

7. https://math.stackexchange.com/questions/129909/correlation-coefficient-and-determination-coefficient 출처참조