import pandas as pd
import seaborn as sns
import plotly.express as px

#for classification tasks
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# To preprocess the data
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler

# machine learning
from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# ignore warnings
import warnings
warnings.filterwarnings('ignore')


# load the data from csv file placed locally in our pc
df = pd.read_csv('creditcard.csv')

from sklearn.preprocessing import Normalizer

normalize = Normalizer(norm='l2')
print(normalize.fit_transform(df))



# separating the data for analysis
trusted = df[df['Class'] == 0]
fraud = df[df['Class'] == 1]


# compare the values for both transactions
df.groupby('Class').mean()

# compare the values for both transactions
df.groupby('Class').mean()

# Under-Sampling (building sample dataset containing similar distribution of normal transactions and Fraudulent Transactions)
trusted_sample = trusted.sample(n=492)
# Concatenating two DataFrames
new_df = pd.concat([trusted_sample, fraud], axis=0)
# Print first 5 rows of the new dataset
new_df.head()


# Check Missing Values
df.isnull().sum().sort_values(ascending = False)


# Splitting the data into Features & Targets
X = new_df.drop(columns='Class', axis=1)
y = new_df['Class']


# Splitting the data into Training data & Testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Check whether the data is splitted in 80:20 ratio
print(X.shape, X_train.shape, X_test.shape)

# Call the Model
model = RandomForestClassifier(random_state=42)

# import pipeline
from sklearn.pipeline import Pipeline


# Create a pipeline for each model
pipeline = Pipeline([
# save the best model
import pickle
pickle.dump(best_model, open('fraud_pickle_file', 'wb'))


model_columns2=list(df.columns)
