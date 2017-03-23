import pandas as pd

# Calculate mean mortality rate and use it as the predicted value. 

train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')

predictions = test[['Id']].copy()
predictions['mortality_rate'] = train["mortality_rate"].mean()

predictions.to_csv('mean.csv', index = False)
