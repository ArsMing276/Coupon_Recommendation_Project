# Coupon_Recommendation_Project

This is a kaggle competition project, Here is a short description: "Using past purchase and browsing behavior, this competition asks you to predict which coupons a customer will buy in a given period of time. The resulting models will be used to improve Ponpare's recommendation system, so they can make sure their customers don't miss out on their next favorite thing." it's website is https://www.kaggle.com/c/coupon-purchase-prediction

1. Merged several large tables (coupon area, detail, list, visit info etc.) together in R.
2. Implemented Artificial Neural Network with backpropagation in Matlab. Used Gender, Age, Coupon Valid Time, Price,
etc. as input layer neurons and Coupon Location, Genres, Discount Rate as output layer neurons. Calculated the distances
between behavior predictions for potential users and testing coupon’s info to determine which coupon to recommend.
3. Implemented composite SVM with Sequential Minimal Optimization in Matlab to classify each customer into a specific
(Coupon Location, Genres, Discount Rate) Category using the same features as in ANN for coupon recommendation.
4. Implemented Matrix Factorization with Alternating Least Squares to successfully predict the missing purchasing numbers, found users’ behavior based on this
matrix and then recommended the testing coupons with small distances to each user’s behavior pattern.
5. Tried Tree Models such as Random Forest, Boosting, Bagging, etc
6. Evaluated the goodness of our models by plotting ROC, PRC curves, using cross-validation to calculate MSE, etc.
