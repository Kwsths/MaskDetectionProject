import os
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle
import numpy as np
from PIL import Image
from sklearn import svm
from joblib import dump, load
from skimage.feature import hog
import pandas as pd
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------------------------
# useful piece of code

# custom class for image transformation. this will be used in the pipeline
# here we do two things: 1. make feature extraction from images using HOG in order to reduce image complexity
# 2. flatten arrays
class HOG_Transformer(BaseEstimator, TransformerMixin):
  def __init__(self):
    # print('\n>>>>>>>init() called.\n')
    return None

  def fit(self, X, y = None):
    # print('\n>>>>>>>fit() called.\n')
    return self

  def transform(self, X, y = None):
    # print('\n>>>>>>>transform() called.\n')
    img_processed = []
    for obs in X:
        fd = hog(obs, orientations=9, pixels_per_cell=(8, 8),
                            cells_per_block=(2, 2), visualize=False, multichannel=False, feature_vector=True)
        img_processed.append(fd)
    return img_processed

# function to read the images
# data have been constructed on three different folders for train, validation and test sets
def readimages(path, folders):
    img_array = []
    labels_array = []
    i = 0
    y = 0

    for folder in folders:
        if os.path.isdir(path + folder):
            for root, dirs, files in os.walk(path + folder):
                print(root)
                if root == (path + folder + '/with_mask'):
                    for name in files:
                        if name.endswith((".jpg")) or name.endswith((".png")):
                            print(os.path.join(root, name))
                            i += 1
                            img_array.append(np.array(Image.open(os.path.join(root, name)).convert('L').resize((64, 128))))
                            labels_array.append(0)      #class 0 is with mask
                elif root == (path + folder + '/without_mask'):
                    for name in files:
                        if name.endswith((".jpg")) or name.endswith((".png")):
                            print(os.path.join(root, name))
                            y += 1
                            img_array.append(np.array(Image.open(os.path.join(root, name)).convert('L').resize((64, 128))))
                            labels_array.append(1)      #class 1 is without mask
                else:
                    print("images on this folder are not loaded")

    return img_array, labels_array


# ---------------------------------------------------------------------------------------------
# start by importing our train and validation datasets

train_img = []
train_class = []

# we join train and validation data to a single dataset
# this is because we will later use a method for 10-CV on which data will be divided automatically
train_img, train_class = readimages('/Users/panoskosmidis/Desktop/MaskDetection', ['/training_data','/validation_data'])

# shuffle data because they were in order on folders
train_img, train_class = shuffle(train_img, train_class, random_state = 45)

# ---------------------------------------------------------------------------------------------
# create a txt file to keep some notes from the process
# we will keep several results

notes = open(r"SVM_notes.txt", "w")
notes.write("SVM Model Selection through Grid Search\n")
notes.write("\n")
# ---------------------------------------------------------------------------------------------
# non-linear SVM

# create a pipeline which includes two steps:
# HOG transformation: reduce image complexity and flatten image arrays (HOG_Transformer is a custom function see above)
# Non-linear SVM classifier
clf_svm_rbf = Pipeline([('hog_tf', HOG_Transformer()),
                        ('clf-svm-rbf', svm.SVC(kernel = 'rbf')),], verbose = True)

# set several parameters
parameters_svm_rbf = {'clf-svm-rbf__C': (0.1, 1, 5, 10),
                      'clf-svm-rbf__gamma': (0.01, 0.001),}

# construct grid search
gs_clf_svm_rbf = GridSearchCV(clf_svm_rbf, parameters_svm_rbf, n_jobs=1, cv = 10)

# fit the model
gs_clf_svm_rbf = gs_clf_svm_rbf.fit(train_img, train_class)

# save the model
dump(gs_clf_svm_rbf, 'nonlinear_svm_model.joblib')

print(gs_clf_svm_rbf.best_score_)
print(gs_clf_svm_rbf.best_params_)

# save best try
notes.write(">>>>Non-Linear SVM Model<<<<\n")
notes.write("Accuracy: " + str(gs_clf_svm_rbf.best_score_) + "\n")
notes.write("Best parameters: " + str(gs_clf_svm_rbf.best_params_) + "\n")
notes.write("\n")
# ---------------------------------------------------------------------------------------------
# linear SVM

# correspondingly, the same process for linear SVM model
clf_svm_ln = Pipeline([('hog_tf', HOG_Transformer()),
                        ('clf-svm-lin', svm.SVC(kernel = 'linear')),], verbose = True)

parameters_svm_ln = {'clf-svm-lin__C': (0.1, 1, 5, 10),}

gs_clf_svm_ln = GridSearchCV(clf_svm_ln, parameters_svm_ln, n_jobs=1, cv = 10)

gs_clf_svm_ln = gs_clf_svm_ln.fit(train_img, train_class)

dump(gs_clf_svm_ln, 'linear_svm_model.joblib')

print(gs_clf_svm_ln.best_score_)
print(gs_clf_svm_ln.best_params_)

# save best try
notes.write(">>>>Linear SVM Model<<<<\n")
notes.write("Accuracy: " + str(gs_clf_svm_ln.best_score_) + "\n")
notes.write("Best parameters: " + str(gs_clf_svm_ln.best_params_) + "\n")
notes.write("\n")
notes.write("--------------------------------------------------------------------------------\n")
notes.write("\n")

# ---------------------------------------------------------------------------------------------
# best model-> kernel: rbf, C: 5, gamma: 0.001

# explore best model and create some metrics
cv = StratifiedKFold(n_splits=10)

best_clf = Pipeline([('hog_tf', HOG_Transformer()),
                       ('clf', svm.SVC(kernel = 'rbf', C=5, gamma=0.001)),], verbose = True)

scores = pd.DataFrame(columns=['accuracy', 'precision', 'recall', 'f1'])

for i, (train, test) in enumerate(cv.split(train_img, train_class)):

    X_train = []
    y_train = []
    X_val = []
    y_val = []
    for index in train:
        X_train.append(train_img[index])
        y_train.append(train_class[index])
    for index in test:
        X_val.append(train_img[index])
        y_val.append(train_class[index])

    best_clf = best_clf.fit(X_train, y_train)

    y_train_pred = best_clf.predict(X_val)

    scores = scores.append(pd.DataFrame([[round(metrics.accuracy_score(y_val, y_train_pred), 3),
                                          round(metrics.precision_score(y_val, y_train_pred, average='macro'), 3),
                                          round(metrics.recall_score(y_val, y_train_pred, average='macro'), 3),
                                          round(metrics.f1_score(y_val, y_train_pred, average='macro'), 3)
                                          ]],
                                        columns=['accuracy', 'precision', 'recall', 'f1']), ignore_index=True)

print(scores.mean())
print(scores)

# save details for the best model
notes.write(">>>>Explore best model<<<<\n")
notes.write("Create best SVM model with kernel: rbf, C: 5, gamma: 0.001\n")
notes.write("\n")
notes.write("Details for each iteration:\n")
notes.write(str(scores) + "\n")
notes.write("Mean values:\n")
notes.write(str(scores.mean()) + "\n")
notes.write("\n")
notes.write("--------------------------------------------------------------------------------\n")
notes.write("\n")

# ---------------------------------------------------------------------------------------------
# load test data

test_img = []
test_class = []

test_img, test_class = readimages('/Users/panoskosmidis/Desktop/MaskDetection', ['/test_data'])

test_img, test_class = shuffle(test_img, test_class, random_state = 98)

# ---------------------------------------------------------------------------------------------
# check model on test data
# find metrics for test data

y_test_pred = best_clf.predict(test_img)

print('Accuracy of the best model on validation dataset:', round(metrics.accuracy_score(test_class, y_test_pred), 4))
print('Confusion matrix:\n%s' % metrics.confusion_matrix(test_class, y_test_pred))


notes.write(">>>>Check model on test data<<<<\n")
notes.write("\n")
notes.write("Accuracy: " + str(round(metrics.accuracy_score(test_class, y_test_pred), 4)) + "\n")
notes.write("Confusion matrix:\n")
notes.write(str(metrics.confusion_matrix(test_class, y_test_pred)) + "\n")
notes.close()

# ---------------------------------------------------------------------------------------------
# plot ROC curve

# compute the probability of one observation belonging to a specific class
y_test_decision_prob = best_clf.decision_function(test_img)

# compute false positives, true positives and area under the curve for our classifier and then plot it
fpr_best_model = dict()
tpr_best_model = dict()
roc_auc_best_model = dict()

fpr_best_model, tpr_best_model, _ = roc_curve(test_class, y_test_decision_prob)
roc_auc_best_model = auc(fpr_best_model, tpr_best_model)


plt.figure()
plt.plot(fpr_best_model, tpr_best_model,label='Best Model ROC curve (area = %0.2f)' % roc_auc_best_model,color='red')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.01])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC graph')
plt.legend(loc=4, prop={'size': 8})
plt.savefig('SVM_ROC.jpg')
