''' Using the Diabetes dataset that is in scikit-learn, answer the questions below and create a scatterplot
graph with a regression line '''

import matplotlib.pylab as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

diabetes = datasets.load_diabetes()
#how many samples and How many features?
#print(diabetes.data.shape)


# What does feature s6 represent?
#print(diabetes.DESCR)
# s6 glu, bloodsugar levels

# set for regession line
data_train, data_test, target_train, target_test = train_test_split(
    diabetes.data, diabetes.target, random_state=11)

lr= LinearRegression()

lr.fit(data_train,target_train)

coef = lr.coef_
intercept = lr.intercept_



#print out the coefficient

print(coef[9])

#print out the intercept

print(intercept)

# 3 use predict to test your model
predicted = lr.predict(data_test)
expected = target_test
# create a scatterplot with regression line

plt.plot(expected, predicted, ".")




x = np.linspace(0,330,100)
print(x)
y = x
plt.plot(x,y)

plt.show()