import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss, recall_score, precision_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier
from scipy import stats
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import StratifiedKFold



class CreditScoringPipeline:

    def __init__(self, train_file_path, test_file_path, model_path):
        self.train_file_path = train_file_path
        self.test_file_path = test_file_path
        self.model_path = model_path
        self.model = XGBClassifier(random_state=42, eval_metric='logloss')
        self.scaler = StandardScaler()
        self.selector = SelectFromModel(estimator=RandomForestClassifier())
        self.features = ['RevolvingUtilizationOfUnsecuredLines',
                         'NumberOfTime30-59DaysPastDueNotWorse',
                         'DebtRatio', 'MonthlyIncome',
                         'NumberOfOpenCreditLinesAndLoans',
                         'NumberOfTimes90DaysLate',
                         'NumberOfTime60-89DaysPastDueNotWorse',
                         'NumberOfDependents']
        self.label = 'SeriousDlqin2yrs'

    def load_training_data(self):
        training_data = pd.read_excel(self.train_file_path, sheet_name="cs-training")
        training_data.drop('Unnamed: 0', axis=1, inplace=True)
        return training_data

    def load_test_data(self):
        test_data = pd.read_excel(self.test_file_path, sheet_name="cs-test")
        test_data.drop('Unnamed: 0', axis=1, inplace=True)
        return test_data

    def analyze_skewness(self, data):
        # extract categorical features
        num_df = data.select_dtypes(include=[np.number])
        skew_values = num_df.skew()

        analysis = []
        for column, skewness in skew_values.items():
            if abs(skewness) < 0.5:
                skew_type = 'approximately symmetric'
            elif skewness > 0:
                skew_type = 'right-skewed (positive skewness)'
            else:
                skew_type = 'left-skewed (negative skewness)'

            if abs(skewness) > 5:
                high_skew = 'Yes'
            else:
                high_skew = 'No'
            analysis.append([column, skewness, skew_type, high_skew])

        result_df = pd.DataFrame(analysis, columns=['Variables', 'Skewness', 'Analysis', 'Highly Skewed'])
        return result_df

    # remove top percentile

    def remove_top_percentile(self, data, columns, percentile=99):
        for col in columns:
            threshold = np.percentile(data[col], percentile)
            data = data[data[col] <= threshold]
        return data

    # cap extreme values

    def cap_extreme_values(self, data, columns, threshold=20):
        for col in columns:
            data.loc[data[col] > threshold, col] = threshold
        return data

    # box-cox transformation for skewed features
    def box_cox_transform(self, data):
        # identify skewed features
        skewed_df = self.analyze_skewness(data)
        highly_skewed_columns = skewed_df[skewed_df['Highly Skewed'] == 'Yes']['Variables'].tolist()

        # apply box-cox transformation
        for column in highly_skewed_columns:
            if data[column].min() > 0 and data[column].notnull().all():
                data[column], _ = stats.boxcox(data[column])
        return data

    def preprocess(self, data):
        data['NumberOfDependents'].fillna(0, inplace=True)
        data['MonthlyIncome'].fillna(data['MonthlyIncome'].mean(), inplace=True)

        # remove duplicate rows
        data.drop_duplicates(inplace=True)

        # drop records with age < 18
        data = data.drop(data[data['age'] < 18].index)

        # age bucketing
        bins = [18, 30, 50, 70, float('inf')]
        labels = ['18-29', '30-49', '50-69', '70 and above']
        data['age_bin'] = pd.cut(data['age'], bins=bins, labels=labels)
        age_bin_mapping = {'18-29': 1, '30-49': 2, '50-69': 3, '70 and above': 4}
        data['age_bin_encoded'] = data['age_bin'].map(age_bin_mapping)

        # remove top 99 percentile
        data = self.remove_top_percentile(data, ['RevolvingUtilizationOfUnsecuredLines', 'DebtRatio', 'MonthlyIncome'])

        # cap extreme values
        data = self.cap_extreme_values(data, ['NumberOfTime30-59DaysPastDueNotWorse',
                                              'NumberOfTimes90DaysLate',
                                              'NumberOfTime60-89DaysPastDueNotWorse'], threshold=20)

        # box-cox transformation
        data = self.box_cox_transform(data)

        return data

    # select features
    def select_features(self, data):
        X = data[self.features]
        y = data[self.label]

        # split data into train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)

        # scale features
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        return X_train, X_test, y_train, y_test

    # build model
    def build_model(self, data):

        X_train, X_test, y_train, y_test = self.select_features(data)

        param_grid = {
            'n_estimators': [100, 200, 300, 400, 500],
            'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
            'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2]
        }
        stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # run grid search

        grid_search = RandomizedSearchCV(self.model,
                                         param_grid,
                                         cv=stratified_kfold,
                                         scoring='roc_auc',
                                         n_iter=10)
        grid_search.fit(X_train, y_train)

        # run model on test data
        y_pred = grid_search.predict(X_test)
        y_pred_proba = grid_search.predict_proba(X_test)[:, 1]

        # evaluate model
        print('Accuracy: ', accuracy_score(y_test, y_pred))
        print('Precision: ', precision_score(y_test, y_pred))
        print('Recall: ', recall_score(y_test, y_pred))
        print('F1 Score: ', f1_score(y_test, y_pred))
        print('ROC AUC: ', roc_auc_score(y_test, y_pred_proba))

        # save model
        joblib.dump(grid_search, self.model_path)
        print('Model saved successfully')

        return grid_search

    def make_predictions(self, data):
        # Load the model
        loaded_model = joblib.load(self.model_path)

        # Preprocess the data
        data = self.preprocess(data)

        # Select the features
        X = data[self.features]

        # Check if the scaler has been fitted
        try:
            check_is_fitted(self.scaler)
        except NotFittedError:
            # If not, fit the scaler on the test data
            self.scaler.fit(X)

        # Scale the features
        X_scaled = self.scaler.transform(X)

        # Make the predictions
        predictions = loaded_model.predict(X_scaled)
        probabilities = loaded_model.predict_proba(X_scaled)[:, 1]

        # Add the predictions and probabilities to the dataframe
        data['SeriousDlqin2yrs'] = predictions
        data['Default_Probability'] = probabilities

        return data


if __name__ == "__main__":
    # Initialize the pipeline with file paths
    train_data = r'C:\Users\samma\PycharmProjects\jn_ta\data\train\cs-training.xlsx'
    test_data = r'C:\Users\samma\PycharmProjects\jn_ta\data\test\cs-test.xlsx'
    model_path = r'C:\Users\samma\PycharmProjects\jn_ta\workspace\end_to_end_pipeline\models\model.joblib'
    pipeline = CreditScoringPipeline(train_data, test_data, model_path)

    # Load and preprocess train data
    train_data = pipeline.load_training_data()
    train_data = pipeline.preprocess(train_data)

    # Build model
    pipeline.build_model(train_data)

    # Load and preprocess test data
    test_data = pipeline.load_test_data()
    test_data = pipeline.preprocess(test_data)

    # Make predictions on the test data
    test_data_with_predictions = pipeline.make_predictions(test_data)

    print(test_data_with_predictions.head())

    # Save the predictions
    test_data_with_predictions.to_csv(
        r'C:\Users\samma\PycharmProjects\jn_ta\workspace\predictions\predictions.csv', index=False)
