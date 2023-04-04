import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from typing import Optional, Union, Tuple, Dict, List, Set, Any
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
from xgboost import XGBClassifier

import copy
import pandas as pd
import pandera as pa


# Reference: https://stackoverflow.com/questions/55562696/how-to-replace-missing-values-with-group-mode-in-pandas
def fast_mode(df: pd.DataFrame, key_columns: List[str], target_column: str):
    """
    Calculate a column mode, by group, ignoring null values.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame over which to calcualate the mode.
    key_columns : list of str
        Columns to groupby for calculation of mode.
    target_column : str
        Column for which to calculate the mode.

    Returns
    -------
    pandas.DataFrame
        One row for the mode of value_col per key_cols group. If ties,
        returns the one which is sorted first.
    """
    return (df.groupby(key_columns + [target_column]).size()
              .to_frame('counts').reset_index()
              .sort_values('counts', ascending=False)
              .drop_duplicates(subset=key_columns)).drop(columns='counts')


class ProcessedTable(object):
    """ Class to hold processed table information

    Parameters
    ----------
    is_ok : bool
        Whether table was processed successfully
    initial_df : pandas.DataFrame
        Table before processing
    final_df : pandas.DataFrame
        Table after processing
    failure_cases : pandas.DataFrame
        Failure cases
    failure_rows : pandas.DataFrame
        Failure rows
    """

    is_ok: bool
    initial_df: pd.DataFrame
    final_df: pd.DataFrame
    failure_cases: pd.DataFrame
    failure_rows: pd.DataFrame

    def __init__(self,
                 is_ok: bool,
                 initial_df: pd.DataFrame,
                 final_df: pd.DataFrame,
                 failure_cases: pd.DataFrame,
                 failure_rows: pd.DataFrame
                ) -> None:
        self.is_ok = is_ok
        self.initial_df = initial_df
        self.final_df = final_df
        self.failure_cases = failure_cases
        self.failure_rows = failure_rows


class LocalityTableProcessor:
    """ Class to process localities table

    Parameters
    ----------
    schema : pandera.DataFrameSchema
        Pandas schema to validate table

    """

    schema: pa.DataFrameSchema

    def __init__(self) -> None:
        self.schema = pa.DataFrameSchema({
            'country_code':      pa.Column(str, unique=False, nullable=False),
            'country_name':      pa.Column(str, unique=False, nullable=False),
            'state_code':        pa.Column(str, unique=False, nullable=False),
            'state_name':        pa.Column(str, unique=False, nullable=False),
            'municipality_code': pa.Column(str, unique=True, nullable=False),
            'municipality_name': pa.Column(str, unique=False, nullable=False),
        }, coerce=True, strict=True)


    def process(self, initial_df: pd.DataFrame) -> ProcessedTable:
        """ Process table. It will clean data, remove noise, and validate it.

        Parameters
        ----------
        initial_df : pandas.DataFrame
            Table to process. It must have the following columns:
            country_code, country_name, state_code, state_name,
            municipality_code, municipality_code.

        Returns
        -------
        ProcessedTable
            Processed table information
        """
        df = copy.deepcopy(initial_df)
        df = self._clean_data(df)
        df = self._standardize_values(df)

        try:
            self.schema.validate(df, lazy=True)
            df = self._convert_dtypes(df)
            is_ok = True
            failure_cases = None
            failure_rows = None
        except (pa.errors.SchemaError, pa.errors.SchemaErrors) as err:
            is_ok = False
            failure_cases = err.failure_cases
            error_indices = err.failure_cases['index'].unique()
            failure_rows = err.data.iloc[error_indices]
            err.failure_cases['index'] += 2 # Add 2 to skip header and 0-indexs

        return ProcessedTable(
            is_ok         = is_ok,
            initial_df    = initial_df,
            final_df      = df,
            failure_cases = failure_cases,
            failure_rows  = failure_rows,
        )


    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        # It will throwing an error during the validation step
        if 'municipality_code' not in df.columns:
            return df

        # Remove unnecessary columns
        df = df[df.columns.intersection(self.schema.columns)]

        # Replace empty strings by N/A
        df = df.replace(r'^\s*$', pd.NA, regex=True)

        # All localities must be municipalities
        df = df[~df['municipality_code'].isna()]

        # Remove redundant data
        df = df.drop_duplicates(subset=['municipality_code'])

        return df


    def _standardize_values(self, df: pd.DataFrame) -> pd.DataFrame:
        # It will throwing an error during the validation step
        columns_required = {'country_code', 'country_name',
                            'state_code', 'state_name'}
        if not columns_required.issubset(set(df.columns)):
            return df

        # Get correct country name
        country_map = fast_mode(df, ['country_code'], 'country_name')\
            .set_index('country_code')['country_name']
        df['country_name'] = df['country_code'].map(country_map)

        # Get correct state name
        state_map = fast_mode(df, ['country_code', 'state_code'], 'state_name')\
            .set_index(['country_code', 'state_code'])['state_name']
        df['state_name'] = df.set_index(['country_code', 'state_code'])\
            .index.map(state_map)

        return df


    def _convert_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.astype('string')


#TODO: Check how many columns are null
#TODO: Add stats per column
#TODO: Add estimators for: school type, school region, latitude, longitude
class SchoolTableProcessor:
    """ Class to process schools table

    Parameters
    ----------
    municipality_codes : set or list
        Set of valid municipality codes
    schema : pandera.DataFrameSchema
        Pandas schema to validate tables
    """

    def __init__(self, municipality_codes: Union[Set, List]) -> None:

        isin_municipality_fn = None
        if isinstance(municipality_codes, (set, list)) and len(municipality_codes) > 0:
            isin_municipality_fn = pa.Check.isin(municipality_codes,
                                                 error='is_valid_municipality_code')

        self.schema = pa.DataFrameSchema({
            'school_code':           pa.Column(str, unique=True, nullable=False),
            'school_name':           pa.Column(str, unique=False, nullable=False),
            'school_type':           pa.Column(str, unique=False, nullable=False),
            'school_region':         pa.Column(str, unique=False, nullable=False),
            'student_count':         pa.Column(int, unique=False, nullable=True),
            'latitude':              pa.Column(float, unique=False, nullable=False),
            'longitude':             pa.Column(float, unique=False, nullable=False),
            'municipality_code':     pa.Column(str, unique=False, nullable=False,
                                               checks=isin_municipality_fn),
            'internet_availability': pa.Column(bool, unique=False, nullable=True)
        }, coerce=True, strict=True)


    def process(self, initial_df: pd.DataFrame) -> ProcessedTable:
        """ Process table. It will clean data, remove noise, and validate it.

        Parameters
        ----------
        initial_df : pandas.DataFrame
            Table to process. It must have the following columns:
            school_code, school_name, school_type, school_region,
            student_count, latitude, longitude, municipality_code,
            internet_availability.

        Returns
        -------
        ProcessedTable
            Processed table information
        """
        df = copy.deepcopy(initial_df)
        df = self._clean_data(df)

        try:
            self.schema.validate(df, lazy=True)
            df = self._convert_dtypes(df)
            is_ok = True
            failure_cases = None
            failure_rows = None
        except (pa.errors.SchemaError, pa.errors.SchemaErrors) as err:
            is_ok = False
            failure_cases = err.failure_cases
            error_indices = err.failure_cases['index'].unique()
            failure_rows = err.data.iloc[error_indices]
            err.failure_cases['index'] += 2 # Add 2 to skip header and 0-indexs

        return ProcessedTable(
            is_ok         = is_ok,
            initial_df    = initial_df,
            final_df      = df,
            failure_cases = failure_cases,
            failure_rows  = failure_rows,
        )


    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        # It will throwing an error during the validation step
        columns_required = {'municipality_code', 'school_code',
                            'internet_availability'}
        if not columns_required.issubset(set(df.columns)):
            return df

        # Remove unnecessary columns
        df = df[df.columns.intersection(self.schema.columns)]

        # Replace empty strings by N/A
        df = df.replace(r'^\s*$', pd.NA, regex=True)

        # All localities must have a municipality code
        df = df[~df['municipality_code'].isna()]

        # Remove duplicated schools
        df = df[~df['school_code'].isna()]

        # Remove redundant data
        df = df.drop_duplicates(subset=['school_code'])

        # Convert string to bolean
        def parse_boolean(value: Any) -> bool:
            if pd.isna(value):
                return None
            value = str(value).lower().strip()
            if value in ['true', 'yes', 'y', '1']:
                return True
            elif value in ['false', 'no', 'n', '0']:
                return False
            else:
                try:
                    return bool(float(value))
                except ValueError:
                    return value
        df['internet_availability'] = df['internet_availability'].apply(parse_boolean)
        return df


    def _convert_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.astype({
            'school_code':           'string',
            'school_name':           'string',
            'school_type':           'string',
            'school_region':         'string',
            'student_count':         'Int32',
            'latitude':              'Float32',
            'longitude':             'Float32',
            'municipality_code':     'string',
            'internet_availability': 'boolean'
        })


class StudentCountEstimator(BaseEstimator, TransformerMixin):

    # Columns needed to estimate student count
    LOCALITY_COLUMNS = [
        'municipality_code',
        'state_code',
        'country_code'
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
        # X.loc[:, 'student_count'] = X.apply(self._get_count, axis=1)
        X['student_count'] = X.apply(self._get_count, axis=1)
        return X


class InternetConnectivityDataLoader:

    def __init__(self,
                 locality_df: pd.DataFrame,
                 school_df: pd.DataFrame,
                 merge_key: Optional[str] = 'municipality_code',
                 ) -> None:
        self.locality_df = locality_df
        self.school_df = school_df
        self.merge_key = merge_key

        # After an extensive experimentation, these were the variables choosen
        self.input_columns = ['latitude',  'longitude', 'student_count',
                              'school_code', 'school_type', 'school_region',
                              'country_code', 'country_name',
                              'state_code', 'state_name',
                              'municipality_code', 'municipality_name']
        self.output_column = 'internet_availability'


    def setup(self) -> bool:
        # TODO: Validate data
        raw_data = pd.merge(self.school_df, self.locality_df, on=self.merge_key)

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


#TODO: Estimate latitude, longitude, school_region, school_type using Mode by Municipality
class InternetConnectivityModel:
    """ Model for predicting internet availability in schools.

    Parameters
    ----------
    continuous_input_columns : list of str
        List of columns that are continuous variables.
    categorical_input_columns : list of str
        List of columns that are categorical variables.
    output_column : str
        Name of the column that contains the output variable.
    model : sklearn model
        Model used to predict the output variable.
    """

    def __init__(self) -> None:
        # TODO: For now, hardcoded variables
        # After an extensive experimentation, these were the variables choosen
        self.continuous_input_columns = ['latitude',  'longitude',
                                         'student_count']
        self.categorical_input_columns = ['school_type', 'school_region',
                                          'state_name']
        self.output_column = 'internet_availability'

        self.model = None


    def prepare_data(self,
                     data: pd.DataFrame,
                     for_train: bool=True
                     ) -> Tuple[pd.DataFrame, pd.Series]:
        """

        Parameters
        ----------
        data : pd.DataFrame
            Dataset with information for predicting internet availaibility
            in schools.
        for_train : bool
            If True, the output variable will be returned. Otherwise, it will
            be discarded.

        Returns
        -------
        X : pd.DataFrame
            Independent variables used to evaluate the model.
        y : pd.Series
            Dependent variable to be predicted. If for_train is False, this
            will be None.
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
            input_data['student_count'] = input_data['student_count'].clip(upper=200)

        # Variables used in the models
        # Discriteze categorical variables
        X = input_data
        y = data[self.output_column].astype(bool) if for_train else None
        return X, y


    def get_classifier_and_param_grid(self, classifier_name):
        """ Get the classifier and the parameter grid for the classifier.

        Parameters
        ----------
        classifier_name: str
            Name of the classifier to be used.

        Returns
        -------
        classifier : sklearn classifier
            Classifier to be used.
        param_grid : dict
            Parameter grid to be used in the classifier.
        """
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
        """ Fit the model.

        Parameters
        ----------
        data : pd.DataFrame
            Dataset with information for predicting internet availaibility
            in schools.
        classifier_name : str
            Name of the classifier to be used.

        Returns
        -------
        results : dict
            Dictionary with the results of the model.
        """
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
                ('encoder', OneHotEncoder(sparse_output=False), self.categorical_input_columns)
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
        """ Predict the internet connectivity in schools.

        Parameters
        ----------
        data : pd.DataFrame
            Dataset with information for predicting internet availaibility
            in schools.

        Returns
        -------
        predictions : list of booleans
            List of internet availability predictions for each school.
        """
        if self.model is None:
            raise NotFittedError("This InternetConnectivityModel instance is"
                                 " not fitted yet. Call 'fit' with appropriate"
                                 " arguments before using this estimator.")

        X, _ = self.prepare_data(data, for_train=False)
        assert X.shape[0] == data.shape[0], \
            f"Number of test entries is different! X: {X.shape} | data: {data.shape}"

        # Predict
        return self.model.predict(X)


class InternetConnectivitySummarizer:
    """ Class for summarizing the results of the InternetConnectivityModel. """

    def compute_stats_by_columns(self,
                                 df: pd.DataFrame,
                                 groupby_columns: List[str]
                                ) -> pd.DataFrame:
        """ Compute statistics by grouping a Pandas DataFrame by one or more columns.

        Parameters
        ----------
        df : pd.DataFrame
            A Pandas DataFrame containing the data to group and aggregate.
        groupby_columns : list of str
            A list of column names to group the data by.

        Returns
        -------
        stats_df : pd.DataFrame
            A Pandas DataFrame containing the aggregated statistics..

        Examples
        --------
            >>> df = pd.DataFrame({
            ...     'state_code': ['CA', 'CA', 'NY', 'NY', 'NY'],
            ...     'municipality_code': ['LA', 'LA', 'NYC', 'NYC', 'BUF'],
            ...     'school_code': ['1', '2', '3', '4', '5'],
            ...     'school_type': ['State', 'Local', 'Federal', 'State', 'State'],
            ...     'school_region': ['Urban', 'Rural', 'Urban', 'Urban', 'Urban'],
            ...     'student_count': [100, 200, 150, 300, 250],
            ...     'internet_availability': ['Yes', 'No', 'Yes', 'NA', 'NA'],
            ...     'internet_availability_prediction': ['Yes', 'No', 'Yes', 'No', 'No'],
            ... })
            >>> stats_df = compute_stats_by_columns(df, ['state_code', 'municipality_code'])
            >>> print(stats_df.to_dict('records'))
            [
                {
                    'state_code': 'CA',
                    'municipality_code': 'LA',
                    'state_count': 1,
                    'municipality_count': 1,
                    'school_count': 2
                    'student_count': 300,
                    'internet_availability_by_value': {
                        'Yes': 0, 'No': 0,'NA': 0
                    },
                    'internet_availability_by_school_region': {
                        'Urban': { 'Yes': 0, 'No': 0, 'NA': 0 },
                        ...
                    }
                    'internet_availability_by_school_type': {
                        'State': { 'Yes': 0, 'No': 0, 'NA': 0 },
                        ...
                    },
                    'internet_availability_prediction_by_value': {
                        'Yes': 0, 'No': 0,
                    },
                    'internet_availability_prediction_by_school_region': {
                        'Urban': { 'Yes': 0, 'No': 0 },
                        ...
                    }
                    'internet_availability_prediction_by_school_type': {
                        'State': { 'Yes': 0, 'No': 0 },
                        ...
                    }
                },
                ...
            ]

        Notes:
            This function computes various statistics based on the data in the input DataFrame `df`,
            grouped by the columns specified in `groupby_columns`. The function uses the `groupby` and
            `agg` methods of a Pandas DataFrame to perform the grouping and aggregation.

            The function computes the following statistics for each group:
            - Number of unique states.
            - Number of unique municipalities.
            - Number of unique schools.
            - Total number of students enrolled.
            - Distribution of internet availability values.
            - Distribution of internet availability values by school region.
            - Distribution of internet availability values by school types
            - Distribution of internet availability prediction values
              (Only schools where 'internet_availability' == 'NA').
            - Distribution of internet availability prediction values by school region
              (Only schools where 'internet_availability' == 'NA').
            - Distribution of internet availability prediction values by school types
              (Only schools where 'internet_availability' == 'NA').
        """

        def parse_bool(value: bool):
            if value is None or pd.isna(value):
                return 'NA'
            return 'Yes' if value else 'No'
        df = copy.deepcopy(df)
        df['internet_availability'] = df['internet_availability'].apply(parse_bool)
        df['internet_availability_prediction'] = df['internet_availability_prediction'].apply(parse_bool)

        # Statistics given Internet Availability

        group_df = df.groupby(groupby_columns)
        stats_df = group_df.agg(
            state_count        = ('state_code', lambda x: len(set(x))),
            municipality_count = ('municipality_code', lambda x: len(set(x))),
            school_count       = ('school_code', lambda x: len(set(x))),
            student_count      = ('student_count', 'sum'),
        )

        stats_df['internet_availability_by_value'] = group_df.apply(
            lambda x: x.groupby(['internet_availability'])
                .size().to_dict()
        )
        stats_df['internet_availability_by_school_region'] = group_df.apply(
            lambda x: x.groupby(['school_region', 'internet_availability'])
                .size().unstack(fill_value=0).to_dict('index')
        )
        stats_df['internet_availability_by_school_type'] = group_df.apply(
            lambda x: x.groupby(['school_type', 'internet_availability'])
                .size().unstack(fill_value=0).to_dict('index')
        )


        # Statistics given Internet Availability Prediction
        # (Only for schools without Internet Availability info)

        stats_df['internet_availability_prediction_by_value'] = group_df.apply(
            lambda x: x[x['internet_availability'] == 'NA']
                .groupby('internet_availability_prediction')
                .size().to_dict()
        )
        stats_df['internet_availability_prediction_by_school_region'] = \
            group_df.apply(
                lambda x: x[x['internet_availability'] == 'NA']
                    .groupby(['school_region', 'internet_availability_prediction'])
                    .size()
                    .unstack(fill_value=0).to_dict('index')
            )
        stats_df['internet_availability_prediction_by_school_type'] = \
            group_df.apply(
                lambda x: x[x['internet_availability'] == 'NA']
                    .groupby(['school_type', 'internet_availability_prediction'])
                    .size()
                    .unstack(fill_value=0).to_dict('index')
            )

        return stats_df


    def compute_statistics_by_locality(self, df: pd.DataFrame) -> Dict:
        """Computes statistics for the given DataFrame `df` grouped by localities.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing the data to compute statistics for.

        Returns
        -------
        Dict
            Dictionary containing the computed statistics.
        """
        groupby_columns = [
            'country_name', 'country_code',
            'state_name', 'state_code',
            'municipality_name', 'municipality_code'
        ]
        country_df = self.compute_stats_by_columns(df, groupby_columns[:2]).reset_index()
        state_df = self.compute_stats_by_columns(df, groupby_columns[:4]).reset_index()
        municipality_df = self.compute_stats_by_columns(df, groupby_columns).reset_index()
        stats_df = pd.concat([country_df, municipality_df, state_df])

        # From DataFrame -> Json
        stats_df[groupby_columns] = stats_df[groupby_columns].fillna('')
        return stats_df.to_dict('records')


if __name__ == '__main__':
    import sys
    args = sys.argv

    if len(args) != 3:
        print('python3 script.py <localities file> <schools file>')

    # Arguments
    locality_filepath = args[1]
    school_filepath = args[2]

    # Files
    locality_df = pd.read_csv(locality_filepath, sep=',', encoding='utf-8', dtype=object)
    school_df = pd.read_csv(school_filepath, sep=',', encoding='utf-8', dtype=object)

    locality_processor = LocalityTableProcessor()
    processed_locality = locality_processor.process(locality_df)

    municipality_codes = set(processed_locality.final_df['municipality_code'].values)
    print(len(municipality_codes))
    school_processor = SchoolTableProcessor(municipality_codes)
    processed_school = school_processor.process(school_df)

    import json
    print(processed_locality.initial_df.shape)
    print(processed_locality.final_df.shape)
    print(processed_school.initial_df.shape)
    print(processed_school.final_df.shape)

    if not processed_locality.is_ok:
        print("locations")
        locality_error = {
            'is_ok': processed_locality.is_ok,
            'failure_cases': processed_locality.failure_cases.to_dict(orient='records'),
        }
        with open('locality_error.json', 'w') as f:
            f.write(json.dumps(locality_error, indent=4))

    if not processed_school.is_ok:
        print("schools")
        school_error = {
            'is_ok': processed_school.is_ok,
            'failure_cases': processed_school.failure_cases.to_dict(orient='records'),
        }
        with open('school_error.json', 'w') as f:
            f.write(json.dumps(school_error, indent=4))

    if not processed_locality.is_ok or not processed_school.is_ok:
        sys.exit(1)

    # Transform the data
    connectivity_dl = InternetConnectivityDataLoader(processed_locality.final_df,
                                                     processed_school.final_df)
    connectivity_dl.setup()

    # Train the model
    model = InternetConnectivityModel()
    model_metrics = model.fit(connectivity_dl.train_dataset)
    print(json.dumps(model_metrics, indent=4))

    # Test
    full_dataset = pd.concat([connectivity_dl.train_dataset,
                              connectivity_dl.test_dataset])
    predictions = model.predict(full_dataset)
    print(sum(predictions) / len(predictions))

    # Connectivity summary
    full_dataset['internet_availability_prediction'] = predictions
    summarizer = InternetConnectivitySummarizer()
    result_summary = summarizer.compute_statistics_by_locality(full_dataset)
    print(json.dumps(result_summary, indent=4))

    response_filepath = 'internet_connectivity_response.json'
    with open(response_filepath, mode='w', encoding='utf-8') as fp:
        response = {
            'model_metrics': model_metrics,
            'result_summary': result_summary
        }
        json.dump(response, fp, indent=4)
