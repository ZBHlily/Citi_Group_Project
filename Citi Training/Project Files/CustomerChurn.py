import pandas as pd

#packages for analysis/modeling 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# Load the dataset
data = pd.read_csv("Databel - Data.csv")

#Prepare the data by separating the features and the target variable:
features = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
target = 'Exited'
X = data[features]
y = data[target]

# Perform one-hot encoding for categorical features
X = pd.get_dummies(X, drop_first=True)  # Converts categorical variables to numeric using one-hot encoding

#Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest classifier
rf = RandomForestClassifier(random_state=42)

# Train the model
rf.fit(X_train, y_train)

# FEature Importance ranks
importances = rf.feature_importances_
feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
"""
              Feature  Importance
1                 Age    0.236922
7     EstimatedSalary    0.147558
0         CreditScore    0.143338
3             Balance    0.141612
4       NumOfProducts    0.131486
2              Tenure    0.082080
6      IsActiveMember    0.040725
8   Geography_Germany    0.026190
5           HasCrCard    0.018454
10        Gender_Male    0.018421
9     Geography_Spain    0.013214
"""

# function that takes user inputs and predicts customer churn
def predict_churn(credit_score=None, geography=None, gender=None, age=None, tenure=None, balance=None, num_of_products=None, has_cr_card=None, is_active_member=None, estimated_salary=None):
    # Create a DataFrame with the user inputs
    user_data = pd.DataFrame([[credit_score, geography, gender, age, tenure, balance, num_of_products, has_cr_card, is_active_member, estimated_salary]],
                             columns=features)

    # Convert categorical variables to numeric using one-hot encoding
    user_data_encoded = pd.get_dummies(user_data, drop_first=True)

    # Align the user input DataFrame with the training DataFrame to ensure matching columns
    user_data_aligned = user_data_encoded.reindex(columns=X.columns, fill_value=0)

    # Make predictions using the trained classifier
    churn_prediction = rf.predict(user_data_aligned)

    return churn_prediction[0]

# Test the accuracy of the Random Forest classifier on the test set
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
# Accuracy: 0.8665

prediction = predict_churn(age=40, balance=5000, is_active_member=0, estimated_salary=50000)
print(f"Churn Prediction:", prediction)
if prediction > 0:
    print("lost customer.")
else:
    print("retained customer!")