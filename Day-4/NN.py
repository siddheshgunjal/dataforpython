import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score


from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense

# Importing dataset
dataset = pd.read_csv("heart.csv")

# train_test_split
x = dataset.drop("target",axis=1)
y = dataset["target"]

X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.20,random_state=0)

# Neural Network
model = Sequential()
model.add(Dense(11,activation='relu',input_dim=13))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(X_train,Y_train,epochs=20)

# saving accuracy score
Y_pred_nn = model.predict(X_test)

#rounding the score value
rounded = [round(x[0]) for x in Y_pred_nn]
Y_pred_nn = rounded

#printing accuracy score
score_nn = round(accuracy_score(Y_pred_nn,Y_test)*100,2)
print("The accuracy score achieved using Neural Network is: "+str(score_nn)+" %")