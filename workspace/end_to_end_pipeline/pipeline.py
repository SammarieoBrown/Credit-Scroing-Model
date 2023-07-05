import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from joblib import dump, load


class CreditScoringPipeline:

    def __init__(self, train_file_path, test_file_path):
        self.train_file_path = train_file_path
        self.test_file_path = test_file_path
        self.model = XGBClassifier(tree_method='exact')
        self.scaler = StandardScaler()
        self.selector = SelectKBest(score_func=f_classif)
        self.imputer = SimpleImputer(strategy='median')
        self.features = None
        self.label = None

    def load_data(self):
        data = pd.read_excel(self.train_file_path,sheet_name = "cs-training")
        data.drop('Unnamed: 0', axis=1, inplace=True)
        return data

    def preprocess(self, data):
        data['NumberOfDependents'].fillna(0, inplace=True)
        data['MonthlyIncome'].fillna(data['MonthlyIncome'].median(), inplace=True)

        # Iterate over columns to remove outliers
        for column in data.columns:
            if column != 'SeriousDlqin2yrs':  # Skip the 'SeriousDlqin2yrs' column as specified
                Q1 = data[column].quantile(0.25)
                Q3 = data[column].quantile(0.75)
                IQR = Q3 - Q1
                data = data[~((data[column] < (Q1 - 1.5 * IQR)) | (data[column] > (Q3 + 1.5 * IQR)))]

        data.drop_duplicates(inplace=True)
        return data

    def split_data(self, data):
        self.features = data.drop(['SeriousDlqin2yrs'], axis=1)
        self.label = data['SeriousDlqin2yrs']

    def feature_engineering(self):
        self.features = self.scaler.fit_transform(self.features)
        self.features = self.selector.fit_transform(self.features, self.label)

    def train(self):

        self.model.fit(self.features, self.label.values.ravel())
        y_pred = self.model.predict(self.features)
        print("Model Accuracy: ", accuracy_score(self.label, y_pred))
        print("Confusion Matrix: \n", confusion_matrix(self.label, y_pred))
        print("Classification Report: \n", classification_report(self.label, y_pred))

    def save_model(self, file_name):
        dump(self.model, file_name)
        print("Model saved successfully.")

    def load_model(self, file_name):
        self.model = load(file_name)
        print("Model loaded successfully.")

    def run_pipeline(self):
        # 1. data collection
        data = self.load_data()

        # 2. data processing and cleaning
        data = self.preprocess(data)

        # 3. split data into features and labels
        self.split_data(data)

        # 4. feature engineering
        self.feature_engineering()

        # 5. model building and training
        self.train()

        # 6. model deployment (saving the model)
        self.save_model('credit_scoring_model.joblib')


if __name__ == '__main__':
    # Instantiate and run the pipeline
    train_data = r'C:\Users\samma\PycharmProjects\jn_ta\data\train\cs-training.xlsx'
    test_data = r'C:\Users\samma\PycharmProjects\jn_ta\data\test\cs-test.xlsx'
    pipeline = CreditScoringPipeline(train_data, test_data)
    pipeline.run_pipeline()
    # print(pipeline.load_data().columns)
