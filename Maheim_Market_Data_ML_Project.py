
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, precision_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

def evaluate_classifiers(classifiers, X_train, y_train, X_test, y_test):
    for classifier_name, classifier in classifiers.items():
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)

        print(f'{classifier_name}:')
        print(f'  Accuracy: {accuracy:.2f}')
        print(f'  Confusion Matrix:\n{cm}')
        print(f'  F1-score: {f1:.2f}')
        print(f'  Recall: {recall:.2f}')
        print(f'  Precision: {precision:.2f}')
        print('-' * 40)

# Load your cleaned data here
data = pd.read_excel(r'MarketCombined.xlsx')

data = data.dropna()

# Convert Timestamp values to strings
datetime_columns = [col for col in data.columns if isinstance(data[col].iloc[0], pd.Timestamp)]
for col in datetime_columns:
    data[col] = data[col].astype(str)

# Calculate the median price
median_price = data['SellingPrice'].median()

# Create a new column 'AboveMedian' with 1 if the selling price is above the median, 0 otherwise
data['AboveMedian'] = (data['SellingPrice'] > median_price).astype(int)

# Remove 'SellingPrice' from the feature list
features = ['Region', 'Year', 'Make', 'Model', 'Trim', 'Body', 'Transmission',
                             'Vin', 'State', 'Condition', 'Odometer', 'Color', 'Interior', 'Seller',
                             'MMR', 'DealType', 'SaleDate', 'Market', 'DayWeekSold',
                             'MonthYearSold', 'DayMonthSold', 'YearSold', 'SaleDate_short',
                             'DateDiff', 'TimeDaySold', 'TimeZone']
features.remove('SellingPrice')

# Preprocess categorical features
le = LabelEncoder()
for feature in features:
    if data[feature].dtype == 'object':
        data[feature] = le.fit_transform(data[feature].astype(str))

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data[features], data['AboveMedian'],
                                                    test_size=0.3, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create a dictionary of classifiers to test
classifiers = {
    'RandomForestClassifier': RandomForestClassifier(random_state=42),
    'LogisticRegression': LogisticRegression(random_state=42),
    'KNeighborsClassifier': KNeighborsClassifier(),
    'SVC': SVC(random_state=42),
    'DecisionTreeClassifier': DecisionTreeClassifier(random_state=42)
}

# Call the function to evaluate classifiers
evaluate_classifiers(classifiers, X_train, y_train, X_test, y_test)
