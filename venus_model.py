import pandas as pd
import matplotlib.pyplot as plot
import seaborn as sb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from time import time
from sklearn.metrics import classification_report

#LOAD DATA

TRAINING_PATH = "data/Volcanoes_train/"
TESTING_PATH = "data/Volcanoes_test/"
training_images = pd.read_csv(TRAINING_PATH + 'train_images.csv', header=None)
training_labels = pd.read_csv(TRAINING_PATH + 'train_labels.csv')
print("training loaded")

testing_images = pd.read_csv(TESTING_PATH + 'test_images.csv', header=None)
testing_labels = pd.read_csv(TESTING_PATH + 'test_labels.csv')
print("testing loaded")

#SHOW DATA
training_counter = training_labels['Volcano?'].value_counts()
testing_counter = testing_labels['Volcano?'].value_counts()

plot.figure(figsize = (8,4))
plot.subplot(121)
sb.barplot(training_counter.index, training_counter.values)
plot.title('training')
plot.subplot(122)
sb.barplot(testing_counter.index, testing_counter.values)
plot.title('testing')
plot.show()


#SHOW EXAMPLES
volocano = training_images[training_labels['Volcano?'] == 1].sample(5)
no_volcano = training_images[training_labels['Volcano?'] == 0].sample(5)
print("sampled")

plot.subplots(figsize=(15, 6))
for i in range(5):
    plot.subplot(2, 5, i+1)
    plot.imshow(volocano.iloc[i, :].values.reshape((110, 110)), cmap='icefire')
    if i == 0:
        plot.ylabel('volcano')
plot.show()
for i in range(5):
    plot.subplot(2, 5, i+6)
    if i == 0:
        plot.ylabel('no volcano')
    plot.imshow(no_volcano.iloc[i, :].values.reshape((110, 110)), cmap='icefire')
plot.show()

#NORMALISE
Xtrain_raw = training_images/256
ytrain_raw = training_labels['Volcano?']
Xtest_raw = testing_images/256
ytest_raw = testing_labels['Volcano?']

Xtrain, Xval, ytrain, yval = train_test_split(Xtrain_raw, ytrain_raw)
Xtest, ytest = Xtest_raw, ytest_raw
model = LogisticRegression()

#LEARN
TrainingStart = time()
model.fit(Xtrain, ytrain)
TrainingEnd = time()
print('Trained in: {} mins.'.format((TrainingEnd-TrainingStart)/60))

#PREDICT AND TEST
predictedValidation = model.predict(Xval)
predictedTest = model.predict(Xtest)
print('validation report:','\n', classification_report(yval, predictedValidation))
print('testing report:', '\n', classification_report(ytest, predictedTest))