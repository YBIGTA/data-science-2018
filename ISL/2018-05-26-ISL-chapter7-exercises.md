---
layout: post
title: "ISL_Chapter7_Exercises"
author: "hyeju.kim"
categories: ML
tags: [ISL]
image: chapter2_exercise.jpg
---

# Chapter7 Exercises

## Conceptual

1-(a)

for all $x <= \xi,$ $f(x) = \beta_0 + \beta_1x + \beta_2x^2 + \beta_3x^3 $ 

if $f(x) = f_1(x),$

$ a_1 = \beta_0, b_1 = \beta_1, c_1 = \beta_2 , d_1 = \beta_3 $

1-(b)

 for all $x > \xi,$

 $f(x) = \beta_0 + \beta_1x + \beta_2x^2 + \beta_3x^3 + \beta_4(x-\xi)^3$ 

$=\beta_0 + \beta_1x + \beta_2x^2 + \beta_3x^3 + \beta_4(x^3-3x^2\xi+3x\xi^2-\xi^3) $ 

$ =(\beta_0 - \beta_4\xi^3) + (\beta_1+3\xi^2\beta_4)x + (\beta_2-3\xi\beta_4)x^2 + (\beta_3+\beta_4)x^3 $

 if $f(x) = f_2(x),$

$ a_2 = \beta_0 - \beta_4\xi^3, b_2 =\beta_1+3\xi^2\beta_4, c_2 =\beta_2-3\xi\beta_4 , d_2 = \beta_3+\beta_4 $



1-(c)

$f_1(\xi)= a_1 + b_1\xi + c_1\xi^2 +d_1\xi^3 $

$= \beta_0 + \beta_1\xi + \beta_2\xi^2 + \beta_3\xi^3$ by 1-(a)

$f_2(\xi)=a_2+b_2\xi + c_2\xi^2 +d_2\xi^3$

$=  (\beta_0 - \beta_4\xi^3)+(\beta_1+3\xi^2\beta_4)\xi +( \beta_2-3\xi\beta_4 )\xi^2 +( \beta_3+\beta_4 )\xi^3$ by 1-(b)

$= \beta_0 + (\beta_1) \xi + (\beta_2) \xi^2 + (\beta_3)\xi^3$ 

$\therefore f_1(x) = f_2(x) $



1-(d)

$f_1'(\xi) = b_1 + 2c_1\xi + 3d_1\xi^2$

$= \beta_1 + 2\beta_2\xi + 3\beta_3\xi^2$ by 1-(a)

$f_2'(\xi) = b_2 + 2c_2\xi + 3d_2\xi^2$ 

$ = (\beta_1+3\xi^2\beta_4) + 2( \beta_2-3\xi\beta_4 )\xi + 3( \beta_3+\beta_4 )\xi^2 $ by 1-(b)

$ = \beta_1 + 2\beta_2\xi + 3\beta_3\xi^2 $ by 1-(b)

$\therefore f_1'(x) = f_2'(x) $



1-(e)

$f_1''(\xi) =  2c_1 + 6d_1\xi$

$= 2\beta_2 + 6\beta_3\xi$ by 1-(a)

$f_2''(\xi) =2c_2 + 6d_2\xi$ 

$ =  2( \beta_2-3\xi\beta_4 )+ 6( \beta_3+\beta_4 )\xi $ by 1-(b)

$= 2\beta_2 + 6\beta_3\xi$ 

$\therefore f_1''(x) = f_2''(x) $



2. ??

2-(a) to minimize $g^ {(0)}$, $g(x) = k$

2-(b) to minimize $g^{(1)}, g(x) α x^2$

2-(c) to minimize $g^{(2)}, g(x) α x^3$

2-(d) to minimize $g^{(3)}, g(x) α x^4$

2-(e) penalty term no matters, therefore the formula just choose g based on minimizing RSS(linear regression)



3.

for $x>=1,$

$Y=\beta_0+\beta_1X + \beta_2(X-1)^2+\epsilon$

for $x<1,$

$Y=\beta_0+\beta_1X +\epsilon$

because $\hat{\beta_0}=1, \hat{\beta_1}=1,\hat{\beta_2}=-2$,

$Y=1+X -2(X-1)^2 = -2X^2+5X-1$ for x>=1

$Y=1+X$   for x<1



4. ​

for $-2<=X<0$

$b_1(X)=  0, b_2(x) = 0$

for $0<=X<1$

$b_1(X)=1,b_2(X)=0$

for $1<=X<=2$

$b_1(X)=1-(X-1)=-X+2,b_2(X)=0$

because $\hat{\beta_0}=1, \hat{\beta_1}=1,\hat{\beta_2}=3$,

$Y = \hat{\beta_0} =1 $ for $-2<=X<0$

$Y = \hat{\beta_0} + \hat{\beta_1} = 2 $ for $0<=X<=1$

$Y = \hat{\beta_0} + \hat{\beta_1}X = -X+3 $ for $1<=X<=2$



5.

5-(a) $\lambda \to \infty, \hat{g_2} $ would be smaller training RSS because it will be a higher order polynomial

5-(b) $\lambda \to \infty, \hat{g_1} ​$ would be smaller test RSS because $\hat{g_2}​$ could overfit

5-(c) if $\lambda=0$, 

 $\hat{g_1}$ 's training,test RSS= $\hat{g_2}$ 's training,test RSS



