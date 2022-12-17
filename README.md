# Final Project of the course "Numerical Linear Algebra"
## Team: Not Recommended

### Team members
- Konstantin Shlychkov
- Alexander Kharitonov
- Alina Bogdanova
- Kira Kuznetsova
- Alex Odnakov

## Topic of the project
Fast Matrix Factorization for Recommender Systems with Implicit Feedback

## Abstract 
Matrix factorizations are widely used in recommender systems to build a latent representation for user-item interactions. One of the most popular methods, although no more state-of-the-art, is ALS. ALS is mostly used as the first stage model in a hybrid recommender system architecture. It aims to quickly select a list of potential candidates for subsequent ranking, however, its quality and speed directly depends on the chosen dimension of the hidden space --- the larger it is, the better prediction quality we get, but more time it takes to converge. In this work we will propose an improvement for this algorithm, which reduce the time complexity of ALS from cubic to quadratic in terms of dimension of latent space. Also we will carry out a series of experiments to compare the quality of the two proposed models.

## List of sources 
1. ALS --- Collaborative Filtering for Implicit Feedback Datasets (http://yifanhu.net/PUB/cf.pdf)
2. eALS --- Fast Matrix Factorization for Online Recommendation with Implicit Feedback (https://arxiv.org/pdf/1708.05024.pdf)
