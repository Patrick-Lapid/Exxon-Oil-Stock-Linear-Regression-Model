import pandas as pd
import sklearn
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew
import statsmodels.api as sm
import math
import pickle

price_data = pd.read_excel("oil_exxon.xlsx")

#Set index to the data column and convert it to a true datetime object
price_data.index = pd.to_datetime(price_data['date'])
price_data = price_data.drop(['date'], axis=1)

#Rename Column
new_column_names = {'exon_price':'exxon_price'}
price_data = price_data.rename(columns = new_column_names)
#Drop any missing values
price_data = price_data.dropna()

x = price_data[['oil_price']]
y = price_data.drop('oil_price', axis=1)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2, random_state=1)


linear = linear_model.LinearRegression()
linear.fit(x_train, y_train)
acc = linear.score(x_test, y_test)
intercept = linear.intercept_[0]
coefficient = linear.coef_[0][0]
print('Our coefficient is: {:.2}'.format(coefficient))
print('Our intercept is: {:.4}'.format(intercept))


prediction = linear.predict(x_test)

x2 = sm.add_constant(x)
model = sm.OLS(y,x2)
est = model.fit()

#Calculate mean squared error
model_mse = mean_squared_error(y_test,prediction)
#Calculate mean absoulte error
model_mae = mean_absolute_error(y_test, prediction)
#Calculate root mean squared error
model_rmse = math.sqrt(model_mse)

print("Mean Squared error: {:.3}".format(model_mse))
print("Mean Absolute error: {:.3}".format(model_mae))
print("Root Mean Squared error: {:.3}".format(model_rmse))

acc = linear.score(x_test, y_test)
print("Accuracy: ", acc)

print(est.summary())

#Plot outputs
plt.scatter(x_test, y_test, color = 'gainsboro', label = "Price")
plt.plot(x_test, prediction, color = 'royalblue', linewidth = 3, linestyle = '-', label = "Regression Line")
plt.title("Linear Regression Model Exxon vs. Oil")
plt.xlabel("Oil")
plt.ylabel("Exxon Mobile")
plt.legend()
plt.show()

#Save Model and load it back using pickle
with open("my_linear_regression.sav", "wb") as f:
    pickle.dump(linear, f)

pickle_in = open("my_linear_regression.sav", 'rb')
loaded_linear = pickle.load(pickle_in)
#Predict values using loaded model
loaded_prediction = loaded_linear.predict([[67.33]])
loaded_prediction = loaded_prediction[0][0]
print("Prediction using loaded model for oil price $67.33 is: ${:.4}".format(loaded_prediction))


'''
#Plot the relationship between the columns
plt.plot(x, y, 'o', color = 'cadetblue', label = 'Daily Price' )
plt.title("Exxon Vs. Oil")
plt.xlabel("Exxon Mobile")
plt.ylabel("Oil")
plt.legend()
plt.show()
'''


"""
#Calculate the excess kurtosis using the fisher method
price_data.hist(grid=False, color = 'cadetblue')
exxon_kurtosis = kurtosis(price_data['exxon_price'], fisher=True)
oil_kurtosis = kurtosis(price_data['oil_price'], fisher=True)
#Calculate the skewness
exxon_skew = skew(price_data['exxon_price'])
oil_skew = skew(price_data['oil_price'])
#Display all values
print("Exxon Kurtosis: {:.2}".format(exxon_kurtosis))
print("Oil Kurtosis: {:.2}".format(exxon_kurtosis))
print("Exxon Skew: {:.2}".format(exxon_skew))
print("Oil Skew: {:.2}".format(oil_skew))
"""

