# Recursive least squares example in Python

## Goal
We want to estimate parameters based on a number
of observations.  
With Recursive Least Squares (RLS) we want to do it
in an online fashion,  
updating the prediction whenever we get a new observation.  
The old observations can be discarded.

The basic concept is as follows:

![](./images/rls_concept.png)

## Algorithm
For the algorithm we introduce the following variables:

- the data $\{ (x_1, y_1), ..., (x_n, y_n) \}$

- $\beta$: the predicted parameters

- $H$: estimation matrix

- $S$: temporary variables to calculate $K$

- $K$: Kalman gain

- $P$: covariance matrix

- $\lambda$: forgetting factor with $0 < \lambda \leq 1$

- $\delta$: initial value for P

The equations for one prediction step:

- $H_k = (1, x_k)$

- $S_k = \lambda  + H_k P_{k-1} H_k^T$

- $K_k = P_{k-1} H_k^T S_k^{-1}$

- $\beta_k = \beta_{k-1} + K_k (y_k - H_k \beta_{k-1})$

- $P_k = \lambda^{-1} ( P_{k-1} - K_k S_k K_k^T)$


**Initial values:**  
The initial value for $\beta$ is often set 
to 0, with $\delta$ we can adjust our trust in that value.  
A high $\delta$ (typically around 100) implies low trust. 


## Examples
The first example compares batch least squares regression (bls) 
with RLS using 200 samples.  
![](./images/rls_lin.png)


In the second example we estimate a sin curve with RLS using 300 samples.  
![](./images/rls_sin.png)
