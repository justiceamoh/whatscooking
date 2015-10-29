# Description: Python script to train a variety of classifiers on whatscooking project data
# Dependencies: DataLoader

from DataLoader import DataLoader

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


## Load Data
loader = DataLoader()
x_train,x_valid,y_train,y_valid = loader.get_data()
labels = loader.classes


## Train some classifiers
#KNN
knn = KNeighborsClassifier(n_neighbors=15)
knn.fit(x_train,y_train)

y_pred= knn.predict(x_valid)
acc = accuracy_score(y_valid,y_pred)
print(classification_report(y_valid,y_pred, labels=np.unique(y_valid), target_names=labels ))
print 'Total Accuracy: {0:2.4}%'.format(acc*100)

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    classes=labels[np.unique(y_valid)]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cm = confusion_matrix(y_valid, y_pred)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
np.set_printoptions(precision=2)
print('Confusion matrix, without normalization')
print(cm)
print('Normalized Confusion matrix')
print(cm_norm)
plt.figure()
plot_confusion_matrix(cm)