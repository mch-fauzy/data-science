
In [1]:

!wget -O Iris.csv https://raw.githubusercontent.com/mch-fauzy/Data-Science/main/Iris-Flower-ML/Iris.csv

--2021-01-26 15:42:41--  https://raw.githubusercontent.com/mch-fauzy/Data-Science/main/Iris-Flower-ML/Iris.csv
Resolving raw.githubusercontent.com (raw.githubusercontent.com)... 151.101.0.133, 151.101.64.133, 151.101.128.133, ...
Connecting to raw.githubusercontent.com (raw.githubusercontent.com)|151.101.0.133|:443... connected.
HTTP request sent, awaiting response... 200 OK
Length: 5107 (5.0K) [text/plain]
Saving to: ‘Iris.csv’

Iris.csv            100%[===================>]   4.99K  --.-KB/s    in 0s      

2021-01-26 15:42:41 (64.8 MB/s) - ‘Iris.csv’ saved [5107/5107]

In [2]:

import pandas as pd

In [3]:

iris = pd.read_csv('Iris.csv')
iris.head()

Out[3]:
	Id 	SepalLengthCm 	SepalWidthCm 	PetalLengthCm 	PetalWidthCm 	Species
0 	1 	5.1 	3.5 	1.4 	0.2 	Iris-setosa
1 	2 	4.9 	3.0 	1.4 	0.2 	Iris-setosa
2 	3 	4.7 	3.2 	1.3 	0.2 	Iris-setosa
3 	4 	4.6 	3.1 	1.5 	0.2 	Iris-setosa
4 	5 	5.0 	3.6 	1.4 	0.2 	Iris-setosa

Checking Data
In [4]:

iris['Species'].value_counts()

Out[4]:

Iris-setosa        50
Iris-virginica     50
Iris-versicolor    50
Name: Species, dtype: int64

In [5]:

print(iris.dtypes,'\n')
missing_data = iris.isnull()

for column in missing_data.columns.values:
    print(missing_data[column].value_counts(), "\n")

Id                 int64
SepalLengthCm    float64
SepalWidthCm     float64
PetalLengthCm    float64
PetalWidthCm     float64
Species           object
dtype: object 

False    150
Name: Id, dtype: int64 

False    150
Name: SepalLengthCm, dtype: int64 

False    150
Name: SepalWidthCm, dtype: int64 

False    150
Name: PetalLengthCm, dtype: int64 

False    150
Name: PetalWidthCm, dtype: int64 

False    150
Name: Species, dtype: int64 

Check Korelasi data secara visual
In [6]:

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
plt.style.use('ggplot') #visual figure style
#iris.plot(kind ='scatter', x='SepalLengthCm', y='SepalWidthCm')
#plt.show()

#sns.set_style("darkgrid") #seaborn figure style
color = {'color': ['g', 'r', 'b']}
iris_fig1 = sns.FacetGrid(iris, hue="Species", hue_kws = color, height=6) #config data and figure for plot (data, columns value to coloring, specifiy the color, figure height)
iris_plot1 = iris_fig1.map(plt.scatter, "SepalLengthCm", "SepalWidthCm").add_legend() #plot the data to scatter plot and add legend
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title('Sepal Length vs Sepal Width')
plt.show()

In [7]:

iris_fig2 = sns.FacetGrid(iris, hue="Species", hue_kws = color, height=6) #config data and figure for plot (data, columns value to coloring, specifiy the color, figure height)
iris_plot2 = iris_fig2.map(plt.scatter, "SepalLengthCm", "PetalLengthCm").add_legend() #plot the data to scatter plot and add legend
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.title('Sepal Length vs Petal Length')
plt.show()

In [8]:

iris_fig3 = sns.FacetGrid(iris, hue="Species", hue_kws = color, height=6) #config data and figure for plot (data, columns value to coloring, specifiy the color, figure height)
iris_plot3 = iris_fig3.map(plt.scatter, "SepalLengthCm", "PetalWidthCm").add_legend() #plot the data to scatter plot and add legend
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.title('Sepal Length vs Petal Width')
plt.show()

In [9]:

iris_fig4 = sns.FacetGrid(iris, hue="Species", hue_kws = color, height=6) #config data and figure for plot (data, columns value to coloring, specifiy the color, figure height)
iris_plot4 = iris_fig4.map(plt.scatter, "SepalWidthCm", "PetalWidthCm").add_legend() #plot the data to scatter plot and add legend
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Petal Width (cm)')
plt.title('Sepal Width vs Petal Width')
plt.show()

In [10]:

iris_fig5 = sns.FacetGrid(iris, hue="Species", hue_kws = color, height=6) #config data and figure for plot (data, columns value to coloring, specifiy the color, figure height)
iris_plot5 = iris_fig5.map(plt.scatter, "SepalWidthCm", "PetalLengthCm").add_legend() #plot the data to scatter plot and add legend
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Petal Length (cm)')
plt.title('Sepal Width vs Petal Length')
plt.show()

In [11]:

iris_fig6 = sns.FacetGrid(iris, hue="Species", hue_kws = color, height=6) #config data and figure for plot (data, columns value to coloring, specifiy the color, figure height)
iris_plot6 = iris_fig6.map(plt.scatter, "PetalLengthCm", "PetalWidthCm").add_legend() #plot the data to scatter plot and add legend
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.title('Petal Length vs Petal Width')
plt.show()

In [12]:

#relasi untuk setiap variable / kolom (nb: diagonal adalah distribusi tiap data pada kolom)
#iris_drop_Id = iris.drop('Id', axis=1)
#sns.set_style("whitegrid");
#sns.pairplot(iris_drop_Id, hue="Species", hue_order = ['Iris-virginica','Iris-versicolor','Iris-setosa'], height=3);
#plt.show()

Heatmap Pearson Correlation
In [13]:

import seaborn as sns
iris_drop_Id = iris.drop('Id', axis=1)
plt.figure(figsize=(9,6)) 
heatmap = sns.heatmap(iris_drop_Id.corr(),annot=True,cmap='Greys')
heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0)  
plt.show()

Persiapkan variable untuk model development dan evaluation
In [14]:

x = iris.drop(['Id', 'Species'], axis=1)
y = iris['Species']

x.head()
#check dimensi (baris, kolom)
#print(x.shape) 
#print(y.shape)
#y.iloc[:]

Out[14]:
	SepalLengthCm 	SepalWidthCm 	PetalLengthCm 	PetalWidthCm
0 	5.1 	3.5 	1.4 	0.2
1 	4.9 	3.0 	1.4 	0.2
2 	4.7 	3.2 	1.3 	0.2
3 	4.6 	3.1 	1.5 	0.2
4 	5.0 	3.6 	1.4 	0.2
In [15]:

from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

ubah data categorical pada kolom species menjadi numeric
In [16]:

from sklearn.preprocessing import LabelEncoder

#y_dummy = pd.get_dummies(y)

#ubah data categorical pada kolom species menjadi numeric
LE = LabelEncoder()
iris['Category'] = LE.fit_transform(y) #buat kolom baru -> input dan transform y ke bentuk numeric
y_numerical = iris['Category']

#check iris species that equal to iris versicolor
#iris[iris['Species']=='Iris-versicolor']

check apakah berhasil
In [17]:

#check apakah berhasil
iris.head()

Out[17]:
	Id 	SepalLengthCm 	SepalWidthCm 	PetalLengthCm 	PetalWidthCm 	Species 	Category
0 	1 	5.1 	3.5 	1.4 	0.2 	Iris-setosa 	0
1 	2 	4.9 	3.0 	1.4 	0.2 	Iris-setosa 	0
2 	3 	4.7 	3.2 	1.3 	0.2 	Iris-setosa 	0
3 	4 	4.6 	3.1 	1.5 	0.2 	Iris-setosa 	0
4 	5 	5.0 	3.6 	1.4 	0.2 	Iris-setosa 	0
In [18]:

iris.iloc[71:76,:]

Out[18]:
	Id 	SepalLengthCm 	SepalWidthCm 	PetalLengthCm 	PetalWidthCm 	Species 	Category
71 	72 	6.1 	2.8 	4.0 	1.3 	Iris-versicolor 	1
72 	73 	6.3 	2.5 	4.9 	1.5 	Iris-versicolor 	1
73 	74 	6.1 	2.8 	4.7 	1.2 	Iris-versicolor 	1
74 	75 	6.4 	2.9 	4.3 	1.3 	Iris-versicolor 	1
75 	76 	6.6 	3.0 	4.4 	1.4 	Iris-versicolor 	1
In [19]:

iris.tail()

Out[19]:
	Id 	SepalLengthCm 	SepalWidthCm 	PetalLengthCm 	PetalWidthCm 	Species 	Category
145 	146 	6.7 	3.0 	5.2 	2.3 	Iris-virginica 	2
146 	147 	6.3 	2.5 	5.0 	1.9 	Iris-virginica 	2
147 	148 	6.5 	3.0 	5.2 	2.0 	Iris-virginica 	2
148 	149 	6.2 	3.4 	5.4 	2.3 	Iris-virginica 	2
149 	150 	5.9 	3.0 	5.1 	1.8 	Iris-virginica 	2

Ubah nilai pada datafram menjadi array x dan y agar bisa di plot pada grafik mesh
In [20]:

import numpy as np

x_array = np.array(x.values) #ubah nilai pada kolom sepal - petal pada dataframe sebagai array
y_numerical_array = np.array(y_numerical.values) #ubah kolom category pada dataframe sebagai array

print(x_array.shape)
print(y_numerical_array)

(150, 4)
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 2 2]

Plot grafik mesh berdasarkan hasil KNN
In [21]:

from matplotlib.colors import ListedColormap

X = x_array[:, 0:2] #hanya terdiri dari 150 baris dan 2 kolom (sepal length dan sepal widht)
Y = y_numerical_array #terdiri dari 150 baris 1 kolom berisi numerical value dari species (0 = iris setosa, etc)

print(X.shape)
print(Y.shape)

h = .02  # step size in the mesh(semakin smooth boundary nya)

# Create color maps
cmap_light = ListedColormap(['orange', 'cyan', 'gray']) #warna untuk color meshgrid
cmap_bold = ListedColormap(['green', 'red', 'blue']) #warna untuk scatter plot

for k in [1, 30]:
    #instance of Neighbours Classifier and fit the data.
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X, Y)

    # Plot the decision boundary (batas-batas grafik mesh)
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(10,5))
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=cmap_bold,
                edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title('3-Class classification (k:' + str(k) + '}' )
    plt.xlabel('Sepal Length (cm)')
    plt.ylabel('Sepal Width (cm)')

plt.show()

(150, 2)
(150,)

In [22]:

X = x_array[:, 0:3:2] #hanya terdiri dari 150 baris dan 2 kolom (sepal length dan petal length)

h = .02  # step size in the mesh(semakin smooth boundary nya)

# Create color maps
cmap_light = ListedColormap(['orange', 'cyan', 'gray']) #warna untuk color meshgrid
cmap_bold = ListedColormap(['green', 'red', 'blue']) #warna untuk scatter plot

for k in [1, 30]:
    #instance of Neighbours Classifier and fit the data.
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X, Y)

    # Plot the decision boundary (batas-batas grafik mesh)
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(10,5))
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=cmap_bold,
                edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title('3-Class classification (k:' + str(k) + '}' )
    plt.xlabel('Sepal Length (cm)')
    plt.ylabel('Petal Length (cm)')

plt.show()

In [23]:

X = x_array[:, 0:4:3] #hanya terdiri dari 150 baris dan 2 kolom (sepal length dan petal widht)

h = .02  # step size in the mesh(semakin smooth boundary nya)

# Create color maps
cmap_light = ListedColormap(['orange', 'cyan', 'gray']) #warna untuk color meshgrid
cmap_bold = ListedColormap(['green', 'red', 'blue']) #warna untuk scatter plot

for k in [1, 30]:
    #instance of Neighbours Classifier and fit the data.
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X, Y)

    # Plot the decision boundary (batas-batas grafik mesh)
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(10,5))
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=cmap_bold,
                edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title('3-Class classification (k:' + str(k) + '}' )
    plt.xlabel('Sepal Length (cm)')
    plt.ylabel('Petal Width (cm)')

plt.show()

In [24]:

X = x_array[:, 1:3] #hanya terdiri dari 150 baris dan 2 kolom (sepal width dan petal petal length)

h = .02  # step size in the mesh(semakin smooth boundary nya)

# Create color maps
cmap_light = ListedColormap(['orange', 'cyan', 'gray']) #warna untuk color meshgrid
cmap_bold = ListedColormap(['green', 'red', 'blue']) #warna untuk scatter plot

for k in [1, 30]:
    #instance of Neighbours Classifier and fit the data.
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X, Y)

    # Plot the decision boundary (batas-batas grafik mesh)
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(10,5))
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=cmap_bold,
                edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title('3-Class classification (k:' + str(k) + '}' )
    plt.xlabel('Sepal Width (cm)')
    plt.ylabel('Petal Length (cm)')

plt.show()

In [25]:

X = x_array[:, 1:4:2] #hanya terdiri dari 150 baris dan 2 kolom (sepal width dan petal petal width)

h = .02  # step size in the mesh(semakin smooth boundary nya)

# Create color maps
cmap_light = ListedColormap(['orange', 'cyan', 'gray']) #warna untuk color meshgrid
cmap_bold = ListedColormap(['green', 'red', 'blue']) #warna untuk scatter plot

for k in [1, 30]:
    #instance of Neighbours Classifier and fit the data.
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X, Y)

    # Plot the decision boundary (batas-batas grafik mesh)
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(10,5))
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=cmap_bold,
                edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title('3-Class classification (k:' + str(k) + '}' )
    plt.xlabel('Sepal Width (cm)')
    plt.ylabel('Petal Width (cm)')

plt.show()

In [26]:

X = x_array[:, 2:4] #hanya terdiri dari 150 baris dan 2 kolom (petal length dan petal widht)

print(X.shape)
print(Y.shape)

h = .02  # step size in the mesh(semakin smooth boundary nya)

# Create color maps
cmap_light = ListedColormap(['orange', 'cyan', 'gray']) #warna untuk color meshgrid
cmap_bold = ListedColormap(['green', 'red', 'blue']) #warna untuk scatter plot

for k in [1, 30]:
    #instance of Neighbours Classifier and fit the data.
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X, Y)

    # Plot the decision boundary (batas-batas grafik mesh)
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(10,5))
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=cmap_bold,
                edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title('3-Class classification (k:' + str(k) + '}' )
    plt.xlabel('Petal Length (cm)')
    plt.ylabel('Petal Width (cm)')

plt.show()

(150, 2)
(150,)

In [27]:

iris['Id'].count()

Out[27]:

150

Latih model menggunakan semua feature
In [28]:

#k = jumlah tetangga terdekat
k_range = list(range(1,31))
acc_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x, y) #fit features(sepal & petal length and width) dengan attribute (species)
    y_pred = knn.predict(x) #y prediction (flower species prediction)
    acc_scores.append(metrics.accuracy_score(y, y_pred)) #accuracy classification score(y actual, y prediction)
    
#print(y_pred)
plt.plot(k_range, acc_scores, linestyle = 'dashed', marker = 'o', markerfacecolor = 'blue')
plt.xlabel('Value of k')
plt.ylabel('Accuracy Score')
plt.title('Accuracy Scores of k-Nearest-Neighbors')
plt.show()

k_data1 = pd.DataFrame({'K Value':k_range,'Accuracy':acc_scores})
k_data1[k_data1['Accuracy'] == 1]

Out[28]:
	K Value 	Accuracy
0 	1 	1.0

Latih model menggunakan semua feature dengan metode train and test split
In [29]:

#split the data to train set and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.4, random_state=5)

#60% for train
print(x_train.shape)
print(y_train.shape)

#40% for test
print(x_test.shape)
print(y_test.shape)

#plt.plot(x_train['SepalLengthCm'],y_train)
#iris_drop_Id = iris.drop('Id', axis=1)
#sns.set_style("whitegrid");
#sns.pairplot(iris_drop_Id, hue="Species", hue_order = ['Iris-virginica','Iris-versicolor','Iris-setosa'], height=3);
#plt.show()

(90, 4)
(90,)
(60, 4)
(60,)

In [30]:

#k = jumlah tetangga terdekat
k_range = list(range(1,31))
acc_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train) #fit data with train set
    y_pred = knn.predict(x_test) #y prediction menggunakan x_test (flower species prediction)
    acc_scores.append(metrics.accuracy_score(y_test, y_pred)) #accuracy classification score(y actual, y prediction)
    
#print(y_pred)
plt.plot(k_range, acc_scores, linestyle = 'dashed', marker = 'o', markerfacecolor = 'blue')
plt.xlabel('Value of k')
plt.ylabel('Accuracy Score')
plt.title('Test Accuracy Scores of k-Nearest-Neighbors with Split Dataset')
plt.show()

k_data2 = pd.DataFrame({'K Value':k_range,'Accuracy':acc_scores})
k_data2[k_data2['Accuracy'] == 1]

Out[30]:
	K Value 	Accuracy
8 	9 	1.0
10 	11 	1.0
12 	13 	1.0
14 	15 	1.0

Test Model
In [45]:

def predict_iris(min, max):
  ok = False
  #Eksekusi jika ok = False. Berhenti eksekusi jika ok = True
  while not ok:
    try:
      sepal_length = float(input("Enter a Sepal Length from 0.1 to 10: "))
      sepal_width = float(input("Enter a Sepal Width from 0.1 to 10: "))
      petal_length = float(input("Enter a Petal Length from 0.1 to 10: "))
      petal_width = float(input("Enter a Petal Width from 0.1 to 10: "))
      ok = True
    except ValueError:
      print("Error: wrong input")
    
    #Check apakah nilai berada di antara range min dan max
    if ok: #jika ok = True maka eksekusi kode dibawah
      ok =  (sepal_length >= min and sepal_length <= max and \
            sepal_width >= min and sepal_width <= max and \
            petal_length >= min and petal_length <= max and \
            petal_width >= min and petal_width <= max)
      predicted_iris = knn.predict([[sepal_length,sepal_width, petal_length, petal_width]])
    
    #Eksekusi jika nilai tidak sesuai min dan max
    if not ok: #jika ok = False maka eksekusi kode dibawah
      print("Error: the value is not within permitted range (" + str(min) + ".." + str(max) + ")")
  
  return predicted_iris

knn = KNeighborsClassifier(n_neighbors=9)
knn.fit(x_train, y_train)

#value for testing prediction
print(x_test.head(), '\n')
print(y_test.head())
print('\n')

# in-sample prediction
prediction = knn.predict([[5.8, 2.7, 3.9, 1.2]])
print('The Flower is:', prediction[0])

# out-of-sample prediction
#prediction = predict_iris(0.1, 10)
#print('The Flower is:', prediction)

     SepalLengthCm  SepalWidthCm  PetalLengthCm  PetalWidthCm
82             5.8           2.7            3.9           1.2
134            6.1           2.6            5.6           1.4
114            5.8           2.8            5.1           2.4
42             4.4           3.2            1.3           0.2
109            7.2           3.6            6.1           2.5 

82     Iris-versicolor
134     Iris-virginica
114     Iris-virginica
42         Iris-setosa
109     Iris-virginica
Name: Species, dtype: object


The Flower is: Iris-versicolor

In [32]:

#prediction_test = knn.predict(x_test)
#prediction_df = pd.DataFrame({'Predicted Species'})
#iris['predicted species'] = prediction_test


