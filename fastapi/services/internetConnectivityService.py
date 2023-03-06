from collections import OrderedDict
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from typing import Optional, Dict, Tuple, List, Any

import pandas as pd

from xgboost import XGBClassifier

class StudentCountEstimator(BaseEstimator, TransformerMixin):
    
    # Columns needed to estimate student count
    LOCALITY_COLUMNS = [
        # 'district_code',
        'municipality_code', 
        # 'microregion_code',
        # 'mesoregion_code', 
        'state_code', 
        # 'region_code',
    ]

    BY_LOCALITY_REGION_TYPE_KEY = 'by=loc+reg+type'
    BY_LOCALITY_REGION_KEY = 'by=loc+reg'
    BY_LOCALITY_KEY = 'by=loc'

    def _generate_counter_map(self,
                              X: pd.DataFrame,
                              column: Optional[str] = None
                             ):
        groupby_columns = [column, 'school_region', 'school_type']

        counter_map = {}
        # School count by locality, school region and school_type
        
        counter_map[self.BY_LOCALITY_REGION_TYPE_KEY] = \
            X.groupby(groupby_columns)['student_count'].median().to_dict()

        # School count by locality and school region
        counter_map[self.BY_LOCALITY_REGION_KEY] = \
            X.groupby(groupby_columns[:2])['student_count'].median().to_dict()

        # School count by locatily
        counter_map[self.BY_LOCALITY_KEY] = \
            X.groupby(groupby_columns[:1])['student_count'].median().to_dict()

        return counter_map

    def fit(self, X: pd.DataFrame, y: pd.Series=None):
        for column in self.LOCALITY_COLUMNS + ['student_count']:
            assert column in X.columns, \
                f"DataFrame does not contain column '{column}'"

        self.locality_counter_maps_ = OrderedDict()
        for column in self.LOCALITY_COLUMNS:
            self.locality_counter_maps_[column] = \
                self._generate_counter_map(X, column)

        return self

    def _get_count(self, row: pd.Series) -> int:
        # Get the "best" approximation possible given school data
        for column, counter_map in self.locality_counter_maps_.items():
            key_values = [row[column],
                          row['school_region'], 
                          row['school_type']]

            key = tuple(key_values)
            if key in counter_map[self.BY_LOCALITY_REGION_TYPE_KEY]:
                return counter_map[self.BY_LOCALITY_REGION_TYPE_KEY][key]

            key = tuple(key_values[:2])
            if key in counter_map[self.BY_LOCALITY_REGION_KEY]:
                return counter_map[self.BY_LOCALITY_REGION_KEY][key]

            key = tuple(key_values[:1])
            if key in counter_map[self.BY_LOCALITY_KEY]:
                return counter_map[self.BY_LOCALITY_KEY][key]

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # Compute the student count for each school
        X.loc[:, 'student_count'] = X.apply(self._get_count, axis=1)
        return X


class InternetConnectivityDataLoader:

    def __init__(self,
                 localities: pd.DataFrame,
                 schools: pd.DataFrame,
                 merge_key: str = 'municipality_code',
                 ) -> None:
        self.localities = localities
        self.schools = schools
        self.merge_key = merge_key

        # TODO: For now, hardcoded variables
        # After an extensive experimentation, these were the variables choosen
        self.input_columns = ['latitude',  'longitude', 'student_count',
                              'school_type', 'school_region', 
                              'state_code', 'state_name',
                              'municipality_code', 'municipality_name']
        self.output_column = 'internet_availability'

    def validate(self) -> None:
        pass

    def setup(self) -> bool:
        # TODO: Validate data
        raw_data = pd.merge(self.schools, self.localities, on=self.merge_key)

        # Columns needed
        data_columns = self.input_columns + [self.output_column]
        dataset = raw_data[data_columns]

        # Since we're applying supervised learning algorithm, we discard 
        # all rows with N/A output values
        output_data = dataset[self.output_column]
        is_test_rows = output_data.isna()

        self._train_dataset = dataset[~is_test_rows].reset_index(drop=True)
        self._test_dataset = dataset[is_test_rows].reset_index(drop=True)

    @property
    def train_dataset(self) -> pd.DataFrame:
        return self._train_dataset

    @property
    def test_dataset(self) -> pd.DataFrame:
        return self._test_dataset


class InternetConnectivityModel:

    def __init__(self) -> None:
        # TODO: For now, hardcoded variables
        # After an extensive experimentation, these were the variables choosen
        self.continuous_input_columns = ['latitude',  'longitude', 
                                         'student_count']
        self.categorical_input_columns = ['school_type', 'school_region', 
                                          # 'region_name', 'mesoregion_name',
                                          'state_name']
        self.output_column = 'internet_availability'

        self.model = None


    def prepare_data(self, 
                     data: pd.DataFrame, 
                     for_train: bool=True
                     ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Input: dataset with information for predicting internet availaibility
                in Brazilian schools.
        Returns the independent variables used to evaluate the model, as well
                as the dependent variable to be predicted in boolean form.
        """

        input_columns = list(data.columns)
        input_columns.remove(self.output_column)
        input_data = data[input_columns]

        # TODO: Move to preprocessing step
        #
        # Fix noise values
        # Since we will have to estimate the student_count, it make sense to use 
        # a small count since most school there are few students
        if 'student_count' in input_data.columns:
            input_data.loc[:, 'student_count'] = \
                input_data['student_count'].clip(upper=200)
        
        # Variables used in the models
        # Discriteze categorical variables
        X = input_data
        y = data[self.output_column].astype(bool) if for_train else None
        return X, y


    def get_classifier_and_param_grid(self, classifier_name):
        if classifier_name == 'decision_tree':
            param_grid = {
                'classifier__max_depth': [1, 2, 4, 8, 16, 32],
                'classifier__min_samples_leaf': [5]
            }
            classifier = DecisionTreeClassifier(random_state=0)        
        elif classifier_name == 'random_forest':
            param_grid = {
                'classifier__max_depth': [1, 2, 4, 8, 16, 32],
                'classifier__min_samples_leaf': [5]
            }
            classifier = RandomForestClassifier(random_state=0)
        elif classifier_name == 'xgboost':
            param_grid = {
                'classifier__learning_rate': [0.1, 0.2, 0.3],
                'classifier__n_estimators': [100],
                'classifier__max_depth': [2, 3, 5]
            }
            classifier = XGBClassifier(random_state=0)
        return classifier, param_grid


    def fit(self, 
            data: pd.DataFrame, 
            classifier_name: str = 'random_forest'
           ) -> Dict[str, Any]:
        X, y = self.prepare_data(data)
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

        num_folds = 10
        cv_folds = StratifiedKFold(num_folds).split(X_train, y_train)
        cv_folds = list(cv_folds)

        classifier, param_grid = self.get_classifier_and_param_grid(classifier_name)

        # Normalize the data
        model = Pipeline(steps=[
            ('estimator', StudentCountEstimator()),
            ('selector', ColumnTransformer([
                ('selector', 'passthrough', self.continuous_input_columns),
                ('encoder', OneHotEncoder(sparse=False), self.categorical_input_columns)
            ], remainder="drop")),
            ('scaler', MinMaxScaler()), 
            ('classifier', classifier)
        ], verbose=False)

        clf_gs = GridSearchCV(estimator=model,
                              param_grid=param_grid,
                              cv=cv_folds,
                              n_jobs=1,
                              verbose=1,
                              scoring=['accuracy'],
                              refit='accuracy',
                              return_train_score=True)

        clf_gs.fit(X_train, y_train)

        cv_results = clf_gs.cv_results_
        best_params = clf_gs.best_params_
        best_index = clf_gs.best_index_

        experiment_result = {
            'classifier_name': classifier_name,
            'num_folds': num_folds,

            'train_accuracies': [],
            'mean_train_accuracy': cv_results["mean_train_accuracy"][best_index],
            'std_train_accuracy': cv_results["std_train_accuracy"][best_index],

            'valid_accuracies': [],
            'mean_valid_accuracy': cv_results["mean_test_accuracy"][best_index],
            'std_valid_accuracy': cv_results["std_test_accuracy"][best_index],
        }

        for fold in range(num_folds):
            train_acc = cv_results[f"split{fold}_train_accuracy"][best_index]
            experiment_result['train_accuracies'].append(train_acc)

            valid_acc = cv_results[f"split{fold}_test_accuracy"][best_index]
            experiment_result['valid_accuracies'].append(valid_acc)
        
        best_model = clf_gs.best_estimator_

        y_pred = best_model.predict(X_test)
        experiment_result['test_accuracy'] = accuracy_score(y_test, y_pred)
        
        # Finally, retrain the model using all labelled data
        model.set_params(**best_params)
        self.model = model.fit(X, y)

        return experiment_result


    def predict(self, data: pd.DataFrame) -> List[bool]:
        if self.model is None:
            raise NotFittedError("This InternetConnectivityModel instance is"
                                 " not fitted yet. Call 'fit' with appropriate"
                                 " arguments before using this estimator.")

        X, _ = self.prepare_data(data, for_train=False)    
        assert X.shape[0] == data.shape[0], \
            f"Number of test entries is different! X: {X.shape} | data: {data.shape}"

        # Predict
        return self.model.predict(X)


if __name__ == '__main__':
    import sys
    args = sys.argv

    if len(args) != 3:
        print('python3 script.py <localities file> <schools file>')

    # Arguments
    localities_filepath = args[1]
    schools_filepath = args[2]

    # Files
    localities_df = pd.read_csv(localities_filepath, sep=',', encoding='utf-8')
    schools_df = pd.read_csv(schools_filepath, sep=',', encoding='utf-8')

    # Transform the data
    connectivity_dl = InternetConnectivityDataLoader(localities_df, schools_df)
    connectivity_dl.setup()

    # Train the model
    model = InternetConnectivityModel()
    result = model.fit(connectivity_dl.train_dataset)
    import json
    print(json.dumps(result, indent=4))

    # Test
    #predictions = model.predict(connectivity_dl.test_dataset)
    #print(sum(predictions) / len(predictions))
      

