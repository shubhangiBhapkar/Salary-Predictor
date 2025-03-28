import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

#read data
salary=pd.read_csv("Salary_Data.csv")
print(salary.head())
#null value check
print(salary.isna().sum())

#feature and target Selection
X =salary[['YearsExperience']]
y =salary[['Salary']]

#splitting data set into trainig and testing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,y,train_size=0.7,random_state= 42)

#Choosing the model to fetch training data
from sklearn.linear_model import LinearRegression
model= LinearRegression()
model.fit(x_train,y_train) #training

#model prediction
y_pred = model.predict(x_test)
print(y_pred)

#Evaluation
from sklearn.metrics import r2_score
results = r2_score(y_pred,y_test)
print(results)

#plot
plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,model.predict(x_train),color='blue')

#save 
import joblib as jb 
jb.dump(model,"Sal_predictor.pkl")

# using model
print('Welcome to the salary precdictor ML model..')
exp = float(input("Enter your experience in years: "))
new_data = pd.DataFrame([[exp]], columns=['YearsExperience'])
sal_model = jb.load('Sal_predictor.pkl')
prediction = sal_model.predict(new_data)

predicted_salary = float(prediction[0])
print(f"\U0001F4B0 Predicted salary for experience {exp} years is: ₹{predicted_salary:,.2f} INR")
