import matplotlib.pyplot as plt 
import seaborn as sns; sns.set()
import numpy as np 
import pandas as pd 
from sklearn.datasets.samples_generator import make_blobs
from matplotlib.animation import FuncAnimation
from sklearn.cluster import KMeans

# X,y_true = make_blobs(n_samples = 200, 
#                       cluster_std=0.75, 
#                       random_state=0,
#                       n_features=14,
#                       centers=4)
# X = X[:, ::-1]

data = pd.read_csv('xclara.csv')
data = data[:100]
X=data.values

# ------------------------------------
# scikit method
# ------------------------------------
print(X[0][0])
kmeans = KMeans(init='random',n_clusters=4)
kmeans.fit(X)
labels = kmeans.predict(X)
print(labels)
centers = kmeans.cluster_centers_
print(centers)
plt.scatter(X[:,0],X[:,1],c=labels,cmap='viridis')
plt.scatter(centers[:,0],centers[:,1],s=40,marker='*')
plt.show()

from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split
y=labels
X_train, X_test, Y_train, Y_test = train_test_split(X,y, test_size=0.4)
c=20
models = (SVC(kernel='linear', C=c),
          LinearSVC(C=c),
          SVC(kernel='rbf',gamma=0.9, C=c),
          SVC(kernel='poly', degree=6,C=c)
         )
#merubah gamma biar lebih fit, cut off parameter untuk hyperplane (effects decision boundary)
models = (classifier.fit(X_train, Y_train) for classifier in models)
judul = ('SVC with linear kernel',
        'Linear SVC with linear kernel',
        'SVC with rbf kernel',
        'SVC with polynomial 3rd degree kernel')

svc = SVC(kernel='rbf',C=1, gamma=1)
svc.fit(X_train, Y_train)
y_pred = svc.predict(X_test)
from sklearn.metrics import accuracy_score 
print(accuracy_score(Y_test,y_pred))

def make_meshgrid(x,y,h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out
fig, sub = plt.subplots(2,2,figsize=(15,15))
plt.subplots_adjust(wspace=.4, hspace=.4)

X0, X1 = X[:,0],X[:,1]
xx,yy = make_meshgrid(X0,X1)
for classifier, title, ax in zip(models, judul, sub.flatten()):
    plot_contours(ax,classifier,xx,yy,cmap='viridis',alpha=0.8)
    ax.scatter(X0,X1,c=y, cmap='viridis',s=20,edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_ylabel('y')
    ax.set_xlabel('x')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)

plt.show()

# colors = ("red", "green", "blue",'orange')
# for i in range(1,41): 
#     kmeans = KMeans(init='random',n_clusters=6,max_iter=i,random_state=0)
#     labels = kmeans.fit(X)
#     labels = labels.predict(X)
#     # print(labels)
#     from pprint import pprint 
#     # pprint(X)

#     centers = kmeans.cluster_centers_
#     iters_ = kmeans.n_iter_
#     print(iters_)
#     print('label:',len(labels))
#     plt.scatter(X[:,0],X[:,1], c= labels, s=7, cmap='viridis')
#     plt.scatter(centers[:,0],centers[:,1],s=40,marker='*')
#     plt.savefig("gambar/{}.jpg".format(i),quality=50 )

# ---------------------------

# from datetime import datetime
# from matplotlib import pyplot
# from matplotlib.animation import FuncAnimation
# from random import randrange

# x_data, y_data = [], []

# figure = pyplot.figure()
# line, = pyplot.plot_date(x_data, y_data, '-')

# def update(frame):
#     x_data.append(datetime.now())
#     y_data.append(randrange(0, 100))
#     line.set_data(x_data, y_data)
#     figure.gca().relim()
#     figure.gca().autoscale_view()
#     return line,

# animation = FuncAnimation(figure, update, interval=1)
# pyplot.show()

from sklearn.cluster import AgglomerativeClustering


# ------------------------------------
# MANUAL
# ------------------------------------

""" f1 = data['V1'].values
f2 = data['V2'].values
X = np.array(list(zip(f1, f2)))
plt.scatter(f1, f2, c='black', s=7)
plt.show()


# cari jarak euclidean
def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)

# berapa centroid
k = 3
# koordinat x centroid random
C_x = np.random.randint(0, np.max(X), size=k)
# koordinat y centroid random
C_y = np.random.randint(0, np.max(X), size=k)
C = np.array(list(zip(C_x, C_y)), dtype=np.float32)
print(C)

plt.scatter(f1, f2, c='#050505', s=7)
plt.scatter(C_x, C_y, marker='*', s=200, c='g')
plt.show()

from copy import deepcopy

# tampung koordinat centroid baru
C_old = np.zeros(C.shape)
# label dari clusternya, kalau 3 berarti (0,1,2)
clusters = np.zeros(len(X))
# Error = jarak centroid baru dengan centroid lama
error = dist(C, C_old, None)
# looping sampe centroid nya tidak bergerak lagi (old == new : true)
i_=1
while error != 0:
    # tiap koordinat di hitung jarak dengan random centroid
    for i in range(len(X)):
        distances = dist(X[i], C)
        cluster = np.argmin(distances)
        clusters[i] = cluster

    # simpan koordinat centroid lama
    C_old = deepcopy(C)
    # cari centroid baru dengan cari rata2 dari nilai koordinat X
    for i in range(k):
        points = [X[j] for j in range(len(X)) if clusters[j] == i]
        C[i] = np.mean(points, axis=0)
    i_+=1
    error = dist(C, C_old, None)
    print('iterasi ke: ',i_,'\nkoordinat: \n',C)
    print(error)

colors = ['r', 'g', 'b', 'y', 'c', 'm']
fig, ax = plt.subplots()
for i in range(k):
        points = np.array([X[j] for j in range(len(X)) if clusters[j] == i])
        ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])
ax.scatter(C[:, 0], C[:, 1], marker='*', s=200, c='#050505')
plt.show()
 """

