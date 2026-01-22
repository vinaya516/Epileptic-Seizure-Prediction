from tkinter import *
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pickle
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import os
from sklearn import svm
import imutils
import numpy as np
import pandas as pd
from keras.utils.np_utils import to_categorical
from sklearn.metrics import accuracy_score
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
import pickle
from keras.callbacks import ModelCheckpoint 
from keras.models import Model
from sklearn.preprocessing import StandardScaler

main = tkinter.Tk()
main.title("Epileptic Seizures Prediction Using Deep Learning Techniques")
main.geometry("1200x1200")

global X_train, X_test, y_train, y_test, scaler
global cnn_model, svm_model, dataset
global X, Y

def uploadDataset():
    global filename, dataset, textdata, labels
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="Dataset")
    text.insert(END,str(filename)+" Dataset Loaded\n\n")
    pathlabel.config(text=str(filename)+" Dataset Loaded")

    dataset = pd.read_csv(filename)
    text.insert(END,str(dataset.head()))
    target = dataset['labels']
    unique, count = np.unique(target, return_counts = True)
    print(unique)
    print(count)
    height = count
    bars = ('Interictal State (Normal)', 'Preictal State (Seizure)')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.title("Interictal State (Normal) & Preictal State (Seizure) Graph")
    plt.xlabel("Label Type")
    plt.ylabel("Count")
    plt.show()
    

def preprocessDataset():
    text.delete('1.0', END)
    global X, Y, dataset, scaler
    global X_train, X_test, y_train, y_test
    dataset.fillna(0, inplace = True)
    dataset = dataset.values
    X = dataset[:,0:dataset.shape[1]-1]
    Y = dataset[:,dataset.shape[1]-1]
    eeg = X[0]
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    Y1 = to_categorical(Y)
    X = np.reshape(X, (X.shape[0], 20, 20, 3))
    X_train, X_test, y_train, y_test = train_test_split(X, Y1, test_size=0.2) #split dataset into train and test
    text.insert(END,"Normalized Dataset Values\n\n")
    text.insert(END,"Dataset Size : "+str(X.shape[0])+"\n")
    text.insert(END,"80% dataset size used to train CNN + SVM : "+str(X_train.shape[0])+"\n")
    text.insert(END,"20% dataset size used to test CNN + SVM  : "+str(X_test.shape[0])+"\n")
    text.update_idletasks()
    plt.plot(eeg)
    plt.title("EEG Signal Graphs")
    plt.xlabel("Time")
    plt.ylabel("Signals")
    plt.show()
    
def calculateMetrics(algorithm, predict, target):
    acc = accuracy_score(target,predict)*100
    p = precision_score(target,predict,average='macro') * 100
    r = recall_score(target,predict,average='macro') * 100
    f = f1_score(target,predict,average='macro') * 100
    text.insert(END,algorithm+" Precision  : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall     : "+str(r)+"\n")
    text.insert(END,algorithm+" F1-Score   : "+str(f)+"\n")
    text.insert(END,algorithm+" Accuracy   : "+str(acc)+"\n\n")
    text.update_idletasks()
    LABELS = ['Interictal State (Normal)', 'Preictal State (Seizure)']
    conf_matrix = confusion_matrix(target, predict) 
    plt.figure(figsize =(6, 6)) 
    ax = sns.heatmap(conf_matrix, xticklabels = LABELS, yticklabels = LABELS, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,2])
    plt.title(algorithm+" Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()


def runCNNSVM():
    text.delete('1.0', END)
    global cnn_model, svm_model, X, Y
    global X_train, X_test, y_train, y_test

    cnn_model = Sequential()
    cnn_model.add(Convolution2D(32, (3,3), input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation = 'relu'))
    cnn_model.add(MaxPooling2D(pool_size = (2, 2)))
    cnn_model.add(Convolution2D(32, (3, 3), activation = 'relu'))
    cnn_model.add(MaxPooling2D(pool_size = (2, 2)))
    cnn_model.add(Flatten())
    cnn_model.add(Dense(units = 256, activation = 'relu'))
    cnn_model.add(Dense(units = y_train.shape[1], activation = 'softmax'))
    cnn_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    if os.path.exists("model/model_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/model_weights.hdf5', verbose = 1, save_best_only = True)
        hist = cnn_model.fit(X_train, y_train, batch_size = 16, epochs = 30, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
        f = open('model/history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:
        cnn_model.load_weights("model/model_weights.hdf5")
    cnn_model = Model(cnn_model.inputs, cnn_model.layers[-2].output)#creating cnn model
    cnn_features = cnn_model.predict(X)  #extracting cnn features from test data
    cnn_X_train, cnn_X_test, cnn_y_train, cnn_y_test = train_test_split(cnn_features, Y, test_size=0.2) #split extracted CNN features into train and test

    svm_model = svm.SVC()
    svm_model.fit(cnn_X_train, cnn_y_train)#now training CNN extracted features using SVM algorithm
    predict = svm_model.predict(cnn_X_test) #now SVM predicting on CNN test features    
    calculateMetrics("CNN + SVM Classification Algorithm", predict, cnn_y_test)    

def predict():
    labels = ['Interictal State (Normal)', 'Preictal State (Seizure)']
    text.delete('1.0', END)
    global cnn_model, svm_model, scaler
    filename = filedialog.askopenfilename(initialdir="Dataset")
    dataset = pd.read_csv(filename)
    dataset = dataset.values
    testData = scaler.transform(dataset)
    testData = np.reshape(testData, (testData.shape[0], 20, 20, 3))
    predict = cnn_model.predict(testData)
    print(predict.shape)
    predict = svm_model.predict(predict)
    print(predict)
    for i in range(len(predict)):
        text.insert(END,"Test Data : "+str(dataset[i])+" =====> Predicted as : "+str(labels[int(predict[i])])+"\n\n")

def trainingGraph():
    f = open('model/history.pckl', 'rb')
    graph = pickle.load(f)
    f.close()
    accuracy = graph['accuracy']
    error = graph['loss']

    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('EPOCH')
    plt.ylabel('Accuracy/Loss')
    plt.plot(accuracy, 'ro-', color = 'green')
    plt.plot(error, 'ro-', color = 'red')
    plt.legend(['CNN Accuracy', 'CNN Loss'], loc='upper left')
    plt.title('CNN Training Accuracy & Loss Graph')
    plt.show()

    
def close():
    main.destroy()

font = ('times', 14, 'bold')
title = Label(main, text='Epileptic Seizures Prediction Using Deep Learning Techniques')
title.config(bg='DarkGoldenrod1', fg='black')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=5,y=5)

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload CHB-MIT Epilepsy Dataset", command=uploadDataset)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='brown', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=500,y=100)

preprocessButton = Button(main, text="Dataset Preprocessing", command=preprocessDataset)
preprocessButton.place(x=50,y=150)
preprocessButton.config(font=font1)

cnnsvmButton = Button(main, text="Run CNN + SVM Classification Algorithm", command=runCNNSVM)
cnnsvmButton.place(x=50,y=200)
cnnsvmButton.config(font=font1)

graphButton = Button(main, text="CNN Training Graph", command=trainingGraph)
graphButton.place(x=50,y=250)
graphButton.config(font=font1)

predictButton = Button(main, text="Epilepsy Prediction from Test Data", command=predict)
predictButton.place(x=50,y=300)
predictButton.config(font=font1)

exitButton = Button(main, text="Exit", command=close)
exitButton.place(x=50,y=350)
exitButton.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(main,height=25,width=90)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=450,y=150)
text.config(font=font1)


main.config(bg='LightSteelBlue1')
main.mainloop()
