import streamlit as st
import pandas as pd
import joblib
import os

# Load the model
model = joblib.load('titanic_model.pkl')  

# Title and description
st.title('Titanic Survival Prediction App')
st.write('This app predicts the survival of passengers on the Titanic.')
st.markdown("""
## About the Titanic Survival Prediction Model
This model predicts whether a passenger on the Titanic would have survived the sinking. It is trained on historical data from the Titanic disaster in 1912, using features such as age, sex, passenger class, and others. The model utilizes a machine learning algorithm to analyze these characteristics and estimate survival outcomes.
""")

# Collecting user input features into a dataframe
input_features = {
    'Sex': st.radio('Sex', options=['male', 'female']),
    'Age': st.slider('Age', min_value=0, max_value=100, value=20),
    'SibSp': st.slider('Siblings/Spouses Aboard', min_value=0, max_value=10, value=0, help="The number of siblings or spouses the passenger has aboard the Titanic."),
    'Parch': st.slider('Parents/Children Aboard', min_value=0, max_value=10, value=0, help="The number of parents or children the passenger has aboard the Titanic."),
    'Fare': st.number_input('Fare', min_value=0.0, max_value=1000.0, value=50.0),
    'Embarked': st.selectbox('Embarked', options=['C', 'Q', 'S']),
    'Pclass': st.selectbox('Passenger Class', options=[1, 2, 3], index=0),
}

# Convert input features into a dataframe
input_df = pd.DataFrame([input_features])

# Predict the output
if st.button('Predict'):

    # Process input to match training data
    input_df['FamilySize'] = input_df['SibSp'] + input_df['Parch'] + 1
    input_df['IsAlone'] = (input_df['FamilySize'] == 1).astype(int)
    input_df['Sex'] = (input_df['Sex'] == 'female').astype(int)
    
    # One-hot encoding for 'Embarked'
    embarked_one_hot = pd.get_dummies(input_df['Embarked'], prefix='Embarked')
    input_df = pd.concat([input_df.drop('Embarked', axis=1), embarked_one_hot], axis=1)

    # One-hot encoding for 'Pclass'
    pclass_one_hot = pd.get_dummies(input_df['Pclass'], prefix='Passenger Class')
    input_df = pd.concat([input_df.drop('Pclass', axis=1), pclass_one_hot], axis=1)

    # Ensure all expected columns are present
    expected_columns = ['Sex', 'Age', 'Fare', 'FamilySize', 'IsAlone', 'Embarked_C', 'Embarked_Q', 'Embarked_S', 'Passenger Class_1', 'Passenger Class_2', 'Passenger Class_3']
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0  # Add missing columns as zeros
    
    # Select and reorder the feature columns to match the training data
    input_df = input_df[expected_columns]

    # Make predictions
    prediction = model.predict(input_df)
    st.write(f'Prediction: {"Survived" if prediction[0] else "Did Not Survive"}')
    
    # Adding the prediction explanation
    if prediction[0]:
        st.success('The model predicts survival. Factors like being female, young age, higher ticket class, or fewer family members aboard might have contributed to this prediction.')
    else:
        st.error('The model predicts non-survival. Factors like being male, older age, lower ticket class, or more family members aboard might have contributed to this prediction.')
    
