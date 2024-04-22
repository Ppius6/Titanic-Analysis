# Titanic Survival Prediction Project

## Project Overview
This project utilizes machine learning techniques to predict the survival likelihood of passengers aboard the Titanic, based on historical data. It covers data preprocessing, exploratory data analysis, feature engineering, model building, evaluation, and deployment using Streamlit.

## Dataset
The dataset used in this project is available on Kaggle, and it is divided into two parts: training and test sets. These datasets include detailed passenger information, which is used to train the models to predict survival outcomes.

[Access the Titanic Dataset on Kaggle](https://www.kaggle.com/c/titanic/data)

### Features
- **PassengerId** - Unique identifier for each passenger
- **Survived** - Survival (0 = No, 1 = Yes)
- **Pclass** (Passenger Class) - Socio-economic status (1 = Upper class, 2 = Middle class, 3 = Lower class)
- **Name** - Passenger's name
- **Sex** - Passenger's sex
- **Age** - Passenger's age
- **SibSp** (Siblings/Spouses Aboard) - The number of siblings or spouses aboard the Titanic
- **Parch** (Parents/Children Aboard) - The number of parents or children aboard the Titanic
- **Ticket** - Ticket number
- **Fare** - Passenger fare
- **Cabin** - Cabin number
- **Embarked** - Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)

## Prerequisites
Before running this project, you need to have the following installed:
- Python 3.8 or higher
- Pandas
- Numpy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Streamlit

You can install these packages using pip:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib streamlit
```

## File Descriptions

`train.csv` - Training set containing details of a subset of the passengers aboard the Titanic.

`test.csv` - Test set containing details of a different subset of the passengers, used to evaluate the models.

`app.py` - Streamlit application for interacting with the trained models.

## Running the Application

To run the Streamlit application, navigate to the project directory in your terminal and run:

```
streamlit run app.py
```

This will start the Streamlit server, and you can interact with the application by going to localhost:8501 in your web browser.

## Model Deployment
The best-performing model is saved and deployed using Streamlit. Users can input passenger details into the web application to receive predictions on survival.

## Contributors
Pius Mutuma 

## License
This project is licensed under the MIT License - see the LICENSE.md file for details.

## Acknowledgments
Inspiration from the historical Titanic dataset. [Access the Titanic Dataset on Kaggle](https://www.kaggle.com/c/titanic/data)

Thanks to the community for various insights and discussions.