# The Iris dataset is referred to as a “toy dataset” because it has only 150 samples and four features. 
# The dataset describes 50 samples for each of three Iris flower species—Iris setosa, Iris versicolor and Iris 
# virginica. Each sample’s features are the sepal length, sepal width, petal 
# length and petal width, all measured in centimeters. The sepals are the larger outer parts of each flower 
# that protect the smaller inside petals before the flower buds bloom.
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt2
#EXERCISE
# load the iris dataset and use classification
# to see if the expected and predicted species
# match up
iris = load_iris()


# display the shape of the data, target and target_names
#print("iris data shape")
#print(iris.data.shape)
#print("iris target shape")
#print(iris.target.shape)
#print(iris.target_names)

# display the first 10 predicted and expected results using
# the species names not the number (using target_names)
data_train, data_test, target_train, target_test = train_test_split(
    iris.data, iris.target, random_state=11)

#print(data_train.shape)
#print(target_train.shape)
#print(data_test.shape)
#print(target_test.shape)


knn = KNeighborsClassifier()

knn.fit(X=data_train , y=target_train)
predicted = knn.predict(X=data_test)
expected = target_test



#print("iris predicted")
#print(predicted[:10])
#print("iris expected")
#print(expected[:10])

predicted = [iris.target_names[x] for x in predicted]
expected = [iris.target_names[x] for x in expected]


#print("iris predicted")
#print(predicted[:10])
#print("iris expected")
#print(expected[:10])

# display the values that the model got wrong
wrong = [(p,e) for (p,e) in zip(predicted, expected) if p != e]
#print("wrong values")
#print(wrong)
# visualize the data using the confusion matrix

confusion = confusion_matrix(y_true=expected, y_pred=predicted)
confusion_df = pd.DataFrame(confusion, index=iris.target_names, columns=iris.target_names)

figre = plt2.figure()
axes = sns.heatmap(confusion_df,
             annot=True, cmap=plt2.cm.nipy_spectral_r)
plt2.xlabel('Expected')
plt2.ylabel('Predicted')


plt2.show()
