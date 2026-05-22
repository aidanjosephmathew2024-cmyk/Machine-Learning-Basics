from sklearn import datasets
from sklearn.model_selection import train_test_split  #To split the data as train and test data
from sklearn.neighbors import KNeighborsClassifier  #To tain the data
from sklearn.metrics import accuracy_score  #To find accuracy of the predictions my giving it both the prdiction array and compare with the actucal labels

iris = datasets.load_iris()

#separate the features and labels

features = iris.data
labels = iris.target

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.5) #It X-train, X-test, Y-train, Y-test
# now the data is split into 50% in train data and 50% in test data


#Using K-Neighbors to train the data

my_classifier = KNeighborsClassifier()
my_classifier.fit(features_train, labels_train) #giving it the data to train

prediction = my_classifier.predict(features_test) #Testing use data without labels and try making connections


#Check the accuracy of the predictions
print(accuracy_score(labels_test, prediction))


#Now testing using data we made
iris1 = [[4.7, 2.5, 3.1, 1.2]]
iris_prediction = my_classifier.predict(iris1)

if iris_prediction[0] == 0:
    print("Setosa")

if iris_prediction[0] == 1:
    print("Versicolor")

if iris_prediction[0] == 2:
    print("Virginica")