#--------------------------------------------------------
#Deep learning Pipeline
#--------------------------------------------------------
# 1. Read the data from the csv file    use pandas to read the data from the csv file
# 2.Data Analysis  (EDA)                use pandas to analyze the data and check for missing values, data types, and basic statistics  
# 3.Preprocessing                        use pandas to preprocess the data, handle missing values, and encode categorical variables
# 4. Train test split                    use sklearn to split the data into training and testing sets
# 5. Feature Scaling                     use sklearn to scale the features using StandardScaler or MinMaxScaler
# 6. FNN model training                  use keras to build and train a feedforward neural network model
# 7. Model evaluation                     use sklearn to evaluate the model performance using metrics such as accuracy, precision, recall, and F1-score  
# 8. Graphical representation of the model
# 9. Model preservation
# 10. Model Loading and preserve
# 11. Test Unseen data
#--------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import  accuracy_score,confusion_matrix
#--------------------------------------------------------
# 1. Read the data from the csv file
#--------------------------------------------------------

print("Reading the data from the csv file")
data = pd.read_csv("placement_data.csv")
print("complete dataset successfully")
print(data)

#--------------------------------------------------------
#2 . Data Analysis (EDA)
#--------------------------------------------------------

print(" step 2 :Data Analysis (EDA)")
print("first 5 rows :")
print(data.head())

print("Columns names :")
print(data.columns)

print("Shape of dataset")
print(data.shape)

print("Statistical summary")
print(data.describe())


#--------------------------------------------------------
# step 3 : Preprocessing  
#--------------------------------------------------------

print("step 3 : Preprocessing")
X = data[['Aptitude', 'Coding', 'Communication', 'Academics', 'Internship']]
Y = data['Placed']

print("Target :")
print(Y.head())


#--------------------------------------------------------
# step 4 : Train test split
#--------------------------------------------------------

print("step 4 : Train test split")
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=42)

print("Traning Input Shape :",X_train.shape)
print("Testing Input Shape :",X_test.shape)
print("Training output Shape :",Y_train.shape)
print("Testing output Shape :",Y_test.shape)


#--------------------------------------------------------
# step 5 : Feature Scalling
#--------------------------------------------------------

print("Step 5 : Feature scalling")

scalar = StandardScaler()

X_train_scaled = scalar.fit_transform(X_train) 
X_test_scaled = scalar.fit_transform(X_test)

print("Scaled Tranning Data :")
print(X_train_scaled[:5])

#--------------------------------------------------------
# step 6 : FNN Model Tranning                                   #imp
#--------------------------------------------------------

print("step 6 : FNN Model Tranning ")

model = MLPClassifier(
    hidden_layer_sizes=(8,4),
    activation='relu',
    solver="adam",
    max_iter=1000,
    random_state=42
)

print(model)

print("Train the model ")
model.fit(X_train_scaled,Y_train)

print("Model tranning completed")

#--------------------------------------------------------
#step 7: Model evaluation 
#--------------------------------------------------------

print("step 7: Model evaluation ")

Y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(Y_test,Y_pred)
print("Accuracy is :",accuracy)

cm = confusion_matrix(Y_test,Y_pred)

print("Confusion Matrix",cm)
print("Predict the probability :")

Y_prob = model.predict_proba(X_test_scaled)

print(Y_prob[:5])

#--------------------------------------------------------
#step 9: Model preserve
#--------------------------------------------------------

print("step 9: Model preserve")

joblib.dump(model,"placement_fnn_model.pkl")
joblib.dump(scalar,"placement_scalar.pkl")
print("Model and scalar gets dump sucessfully")

#--------------------------------------------------------
#step 10: Model Loading and Preserve
#--------------------------------------------------------

print("step 10: Model Loading and Preserve")

loaded_model = joblib.load("placement_fnn_model.pkl")
loaded_scalar = joblib.load("placement_scalar.pkl")

print("Model gets loaded successfully")

#--------------------------------------------------------
#Step 11. Test Unseen data
#Aptitude          :   70
#coding            :   75
#Communication     : 80
#Acadamics         : 85
#Internship        : 1
#--------------------------------------------------------

new_student = pd.DataFrame([[70,75,80,85,1]],columns=['Aptitude', 'Coding', 'Communication', 'Academics', 'Internship'])

new_student_scaled = loaded_scalar.transform(new_student)

new_prediction = loaded_model.predict(new_student_scaled)

new_probablity = loaded_model.predict_proba(new_student_scaled)

print("New student Data:")
print(new_student)

print("Prediction Probability :",new_probablity)

if new_prediction[0]==1:
    print("Prediction : placed")

else:
    print("Prediction : Not Placed")

