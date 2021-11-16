
import numpy as np
import math
import pickle
import pandas as pd
import cv2 as cv
import scipy.misc
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import tensorflow as tf
import keras
import keras.layers as layers
from keras.models import Sequential
from scipy import interp
from itertools import cycle
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import roc_curve, auc

# ______________________ load pickle Fashion MNIST __________________________

file=open('MNIST/train_image','rb')
train_images=pickle.load(file)
file.close()

file=open('MNIST/train_labels','rb')
train_labels=pickle.load(file)
file.close()

file=open('MNIST/test_image','rb')
test_images=pickle.load(file)
file.close()

file=open('MNIST/test_labels','rb')
test_labels=pickle.load(file)
file.close()


resized_train=np.zeros((len(train_images),32,32,1))

for i in range(len(train_images)):
    resized_train[i,:,:,0]=cv.resize(train_images[i,:,:],(32,32),interpolation=cv.INTER_AREA)
    

resized_test=np.zeros((len(test_images),32,32,1))

for i in range(len(test_images)):
    resized_test[i,:,:,0]=cv.resize(test_images[i,:,:],(32,32),interpolation=cv.INTER_AREA)

print('Resized and ready to train the model')
# ____________________________ model ____________________________

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

model = keras.Sequential()

# C1 Convolutional Layer
model.add(layers.Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=(32,32,1)))

# S2 Pooling Layer
model.add(layers.AveragePooling2D())

# C3 Convolutional Layer
model.add(layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))

# S4 Pooling Layer
model.add(layers.AveragePooling2D())

#Flatten the CNN output so that we can connect it with fully connected layers
model.add(layers.Flatten())

# FC6 Fully Connected Layer
model.add(layers.Dense(units=120, activation='relu'))
model.add(layers.Dense(units=84, activation='relu'))

#Output Layer with softmax activation
model.add(layers.Dense(units=10, activation = 'softmax'))

model.summary()

model.compile(optimizer='Adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# _______________________ fit model __________________________

model.fit(resized_train, train_labels, epochs=3,batch_size=128)

# ______________________ test accuracy ____________________________

test_loss, test_acc = model.evaluate(resized_test,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)

# _______________________ predict __________________________

prediction = model.predict(resized_test)

y_pred=np.zeros(len(prediction))
for i in range(len(prediction)):
    y_pred[i]=np.argmax(prediction[i,:])
    

# ______________________ performance measures ____________________
    
# _____________________ CONFUSION MATRIX _____________________

confusionmatrix=np.zeros((len(class_names),len(class_names)))

confusionmatrix=confusion_matrix(test_labels, y_pred)

plt.figure()
plt.title('LeNet5')
sns.heatmap(confusionmatrix,annot=True,fmt="d",square=True,xticklabels=class_names,yticklabels=class_names,)
b, t = plt.ylim() # discover the values for bottom and top
b += 0.5
t -= 0.5
plt.ylim(b,t)
plt.show()

# __________________ divide classes_______________

n_classes=len(class_names)
y_class=np.zeros((len(test_labels),len(class_names)))

for i in range(len(class_names)):
    y_class[:,i]=(test_labels==i)

# _______________ PRECISION AND RECALL ________________

# For each class
precision = dict()
recall = dict()
average_precision = dict()
average_recall = dict()
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(y_class[:, i],
                                                        prediction[:, i])
    average_precision[i] = average_precision_score(y_class[:, i], prediction[:, i])
    # average_recall[i] = recall_score(y_class[:, i], prediction[:, i])
    
plt.figure(figsize=(14, 10))  
labels = []  
colors = cycle(['orange', 'red', 'blue','lime','green','cyan','purple','magenta','grey','black'],)
    
for i, color in zip(range(n_classes), colors):
    l, = plt.plot(recall[i], precision[i], color=color, lw=2)
    labels.append('Precision-recall for class {0} (precision = {1:0.2f})'
                  ''.format(class_names[i], average_precision[i]))
    
fig = plt.gcf()
fig.subplots_adjust(bottom=0.25)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve for each class')
plt.legend(labels, loc=(0, -.38), prop=dict(size=14))
plt.show()  

# __________________ micro averaging precision and recall _____________________


precision["micro"], recall["micro"], _ = precision_recall_curve(y_class.ravel(), prediction.ravel())
average_precision["micro"] = average_precision_score(y_class, prediction,average="micro")

lw = 2
plt.figure(figsize=(14, 10))

plt.plot(precision['micro'], recall['micro'], color='red', lw=lw,label='ROC curve of class {0} (precision = {1:0.2f})'.format('micro', average_precision['micro']))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Micro precision and recall curve')
plt.legend(loc="lower right",prop=dict(size=14))
plt.show()


# ____________ ROC SCORE __________________

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    
    fpr[i], tpr[i], _ = roc_curve(y_class[:,i], prediction[:,i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    
# Plot all ROC curves

lw = 2
plt.figure(figsize=(14, 10))
colors = cycle(['orange', 'red', 'blue','lime','green','cyan','purple','magenta','grey','black'],)
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(class_names[i], roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multilabel ROC curve')
plt.legend(loc="lower right",prop=dict(size=14))
plt.show()

# ___________________ MACRO ROC curves _________________

# First aggregate all false positive rates

all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# # Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# # Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot macro ROC curve

lw = 2
plt.figure(figsize=(14, 10))

plt.plot(fpr['macro'], tpr['macro'], color='red', lw=lw,label='ROC curve of class {0} (area = {1:0.2f})'.format('macro', roc_auc['macro']))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Macro ROC curve')
plt.legend(loc="lower right",prop=dict(size=14))
plt.show()
    
# _______________ PLOT RESULTS _________________

for i in range(2):
    number=5
    selection=(len(test_labels)*np.random.rand(number,)).astype(int)
    bars = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')
    
    plt.figure(figsize=(16,12))
    for i in range(1,number+1):
        plt.subplot(math.ceil(number),2,i*2-1)
        plt.title(bars[test_labels[selection[i-1]]],fontsize=6)
        plt.imshow(resized_test[selection[i-1]],cmap=plt.cm.binary)
        plt.xticks([])
        plt.yticks([])
        plt.subplot(math.ceil(number),2,i*2)
        plt.bar(range(10), prediction[selection[i-1]])
        plt.xticks(range(10), bars, fontsize=6)
    plt.show()















