# Description: Python script to train a variety of classifiers on whatscooking project data
# Dependencies: DataInterface

from DataInterface import DataInterface

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


## Functions 
def plot_confusion_matrix(y_true, y_pred):
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    np.set_printoptions(precision=2)
    print('Confusion matrix, without normalization')
    print(cm)
    print('Normalized Confusion matrix')
    print(cm_norm)

    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    classes=labels[np.unique(y_true)]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


## Load Data
dface  = DataInterface()
x_train,x_valid,y_train,y_valid = dface.get_traindata()
x_test = dface.get_testdata()
labels = dface.classes

## Train some classifiers
#KNN
knn = KNeighborsClassifier(n_neighbors=15)
knn.fit(x_train,y_train)

y_pred= knn.predict(x_valid)
acc = accuracy_score(y_valid,y_pred)
print(classification_report(y_valid,y_pred, labels=np.unique(y_valid), target_names=labels ))
print 'Total Accuracy: {0:2.4}%'.format(acc*100)


plot_confusion_matrix(y_valid, y_pred)


## Create Submission file
clf = knn
predictions = clf.predict(x_test)
dface.make_submission(y_pred)