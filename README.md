# Overview
This is an MVP prototype of an app to predict the attendance lift that a bobblehead promotion will have on attendance. It pools together data for MLB games from 2012 to train a gradient boosting regression trees.

## Business Scenario:
Teams would like help from the MLB league office to create a better forecasting method for running bobblehead promotions. Ideally, they want to understand the expected sales lift from running a promotion. 

## Methodology:
+ Train ML model using Catboost library
+ Predict attendance with bobblehead
+ Predict attendance without bobblehead
+ Simulate 10,000 games using a normal distribution with a mean of the point prediction and standard deviation of the test set RMSE
+ Compute the attendance lift
+ Compute the confidence (percentage of bobblehead simulations > non-bobblehead simulations)

## Use App:

Install libraries in a fresh environment. The list includes dependencies and those used for jupyter lab. 

```
pip install requirements.txt
```

Run the following command from the terminal to launch the app. 
```
streamlit run app.py
```
