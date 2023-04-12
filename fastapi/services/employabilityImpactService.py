from typing import Optional, Union, List, Tuple, Set, Dict, Any
from ast import literal_eval
from collections import Counter
from services.utils import fast_mode, convert_to_list, parse_int, parse_boolean

from scipy.stats import ks_2samp

import copy
import numpy as np
import pandas as pd
import pandera as pa
import re

pd.set_option('display.float_format', '{:.3f}'.format)


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
                ) -> None:
        self.is_ok = is_ok
        self.initial_df = initial_df
        self.final_df = final_df
        self.failure_cases = failure_cases


class SchoolHistoryTableProcessor:
    """ Class to process school history table

    Parameters
    ----------
    schema : pandera.DataFrameSchema
        Pandas schema to validate table

    """

    schema: pa.DataFrameSchema

    def __init__(self, municipality_codes: Union[Set, List]) -> None:        
        self.municipality_codes = None

        isin_municipality_fn = None
        if isinstance(municipality_codes, (set, list)) and len(municipality_codes) > 0:
            self.municipality_codes = municipality_codes
            isin_municipality_fn = pa.Check.isin(municipality_codes,
                                                 error='is_valid_municipality_code')

        is_list_fn = pa.Check(
            lambda x: isinstance(x, (list, tuple, set)),
            error='is_list_type', element_wise=True
        )
        is_same_length_fn = pa.Check(
            lambda row:  len(row['years']) == len(row['internet_availability']),
            error='is_same_length', element_wise=True
        )

        self.schema = pa.DataFrameSchema({
            'school_code':           pa.Column(str, unique=True, nullable=False),
            'school_name':           pa.Column(str, unique=False, nullable=False),
            'municipality_code':     pa.Column(str, unique=False, nullable=False,
                                               checks=isin_municipality_fn),
            'years':                 pa.Column(object, unique=False, nullable=False,
                                               checks=is_list_fn),
            'internet_availability': pa.Column(object, unique=False, nullable=False,
                                               checks=is_list_fn),
        },  checks=is_same_length_fn, coerce=True, strict=True)


    def process(self, initial_df: pd.DataFrame) -> ProcessedTable:
        """ Process table. It will clean data, remove noise, and validate it.

        Parameters
        ----------
        initial_df : pandas.DataFrame
            Table to process. It must have the following columns:
            school_code, school_name, municipality_code, years, internet_availability

        Returns
        -------
        ProcessedTable
            Processed table information
        """
        df = copy.deepcopy(initial_df)
        df = self.__preprocess(df)
        df = self.__clean_data(df)

        try:
            self.schema.validate(df, lazy=True)
            df = self.__convert_dtypes(df)
            is_ok = True
            failure_cases = None
        except (pa.errors.SchemaError, pa.errors.SchemaErrors) as err:
            is_ok = False
            failure_cases = err.failure_cases
            err.failure_cases['index'] += 2 # Add 2 to skip header and 0-indexs

        return ProcessedTable(
            is_ok         = is_ok,
            initial_df    = initial_df,
            final_df      = df,
            failure_cases = failure_cases,
        )


    def __preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        # It will throwing an error during the validation step
        columns_required = {'years', 'internet_availability'}
        if not columns_required.issubset(set(df.columns)):
            return df

        df['years'] = df['years']\
            .apply(lambda x: convert_to_list(x, parse_int))
        df['internet_availability'] = df['internet_availability']\
            .apply(lambda x: convert_to_list(x, parse_boolean))
        return df


    def __clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        # It will throwing an error during the validation step
        columns_required = {'municipality_code', 'years', 'internet_availability'}
        if not columns_required.issubset(set(df.columns)):
            return df

        # Remove unnecessary columns
        df = df[df.columns.intersection(self.schema.columns)]

        # Replace empty strings by N/A
        df = df.replace(r'^\s*$', pd.NA, regex=True)

        # All localities must be municipalities and have connectivity data
        df = df[~df['school_code'].isna()]
        df = df[~df['municipality_code'].isna()]
        df = df[~df['years'].isna()]
        df = df[~df['internet_availability'].isna()]
        df = df[~df['internet_availability'].apply(lambda x: None in x)]

        # Must have data for all years
        years = df['years'].apply(set).values
        years = set.union(*years)
        df = df[df['years'].apply(lambda x: set(x) == years)]

        # Get only valid cities
        if self.municipality_codes is not None:
            df = df[df['municipality_code'].isin(self.municipality_codes)]

        # Remove redundant data
        df = df.drop_duplicates(subset=['school_code'])

        return df


    def __convert_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.astype({
            'school_code':           'string',
            'school_name':           'string',
            'municipality_code':     'string',
            'years':                 'object',
            'internet_availability': 'object'
        })


class EmployabilityHistoryTableProcessor:
    """ Class to process employability history table

    Parameters
    ----------
    schema : pandera.DataFrameSchema
        Pandas schema to validate table

    """

    schema: pa.DataFrameSchema

    def __init__(self) -> None:
        is_list_fn = pa.Check(
            lambda x: isinstance(x, (list, tuple, set)),
            error='is_list_type', element_wise=True
        )
        is_same_length_fn = pa.Check(
            lambda row:  len(row['years']) == len(row['employability_rate']),
            error='is_same_length', element_wise=True
        )

        self.schema = pa.DataFrameSchema({
            'country_code':       pa.Column(str, unique=False, nullable=False),
            'country_name':       pa.Column(str, unique=False, nullable=False),
            'state_code':         pa.Column(str, unique=False, nullable=False),
            'state_name':         pa.Column(str, unique=False, nullable=False),
            'municipality_code':  pa.Column(str, unique=True, nullable=False),
            'municipality_name':  pa.Column(str, unique=False, nullable=False),
            'hdi':                pa.Column(float, unique=False, nullable=False),
            'population_size':    pa.Column(float, unique=False, nullable=False),
            'years':              pa.Column(object, unique=False, nullable=False,
                                            checks=is_list_fn),
            'employability_rate': pa.Column(object, unique=False, nullable=False,
                                            checks=is_list_fn),
        },  checks=is_same_length_fn, coerce=True, strict=True)


    def process(self, initial_df: pd.DataFrame) -> ProcessedTable:
        """ Process table. It will clean data, remove noise, and validate it.

        Parameters
        ----------
        initial_df : pandas.DataFrame
            Table to process. It must have the following columns:
            country_code, country_name, state_code, state_name,
            municipality_code, municipality_name, hdi, population_size,
            years, employability_rate

        Returns
        -------
        ProcessedTable
            Processed table information
        """
        df = copy.deepcopy(initial_df)
        df = self.__preprocess(df)
        df = self.__clean_data(df)
        df = self.__standardize_values(df)

        try:
            self.schema.validate(df, lazy=True)
            df = self.__convert_dtypes(df)
            is_ok = True
            failure_cases = None
        except (pa.errors.SchemaError, pa.errors.SchemaErrors) as err:
            is_ok = False
            failure_cases = err.failure_cases
            err.failure_cases['index'] += 2 # Add 2 to skip header and 0-indexs

        return ProcessedTable(
            is_ok         = is_ok,
            initial_df    = initial_df,
            final_df      = df,
            failure_cases = failure_cases,
        )


    def __preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        # It will throwing an error during the validation step
        columns_required = {'years', 'employability_rate'}
        if not columns_required.issubset(set(df.columns)):
            return df

        df['years'] = df['years']\
            .apply(lambda x: convert_to_list(x, parse_int))
        df['employability_rate'] = df['employability_rate']\
            .apply(lambda x: convert_to_list(x, parse_int))
        return df


    def __clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        # It will throwing an error during the validation step
        columns_required = {'municipality_code', 'hdi', 'population_size',
                            'years', 'employability_rate'}
        if not columns_required.issubset(set(df.columns)):
            return df

        # Remove unnecessary columns
        df = df[df.columns.intersection(self.schema.columns)]

        # Replace empty strings by N/A
        df = df.replace(r'^\s*$', pd.NA, regex=True)

        # All localities must be municipalities and have their social statistics
        df = df[~df['municipality_code'].isna()]
        df = df[~df['hdi'].isna()]
        df = df[~df['population_size'].isna()]
        df = df[~df['years'].isna()]
        df = df[~df['employability_rate'].isna()]
        df = df[~df['employability_rate'].apply(lambda x: None in x)]

        # Must have data for all years
        years = df['years'].apply(set).values
        years = set.union(*years)
        df = df[df['years'].apply(lambda x: set(x) == years)]

        # Remove redundant data
        df = df.drop_duplicates(subset=['municipality_code'])

        return df


    def __standardize_values(self, df: pd.DataFrame) -> pd.DataFrame:
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


    def __convert_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.astype({
            'country_code':       'string',
            'country_name':       'string',
            'state_code':         'string',
            'state_name':         'string',
            'municipality_code':  'string',
            'municipality_name':  'string',
            'hdi':                'float',
            'population_size':    'int',
            'years':              'object',
            'employability_rate': 'object'
        })


class EmployabilityImpactDataLoader:

    def __init__(self,
                 employability_history_df: pd.DataFrame,
                 school_history_df: pd.DataFrame,
                 merge_key: Optional[str] = 'municipality_code',
                 ) -> None:
        self.employability_history_df = employability_history_df
        self.school_history_df = school_history_df
        self.merge_key = merge_key

        # After an extensive experimentation, these were the variables choosen
        self.school_history_columns = [
            'school_code', 'municipality_code', 'years',
            'internet_availability'
        ]

        self.connectivity_history_columns = [
            'connectivity_year', 'connectivity_rate', 'school_count'
        ]
        self.employability_history_columns = [
            'country_code', 'country_name', 'state_code', 'state_name',
            'municipality_code', 'municipality_name', 'hdi', 'population_size',
            'employability_year', 'employability_rate'
        ]


    def _get_connectivity_history(self) -> pd.DataFrame:
        # Join history columns
        df = copy.deepcopy(self.school_history_df)

        df['internet_availability_dict'] = self.school_history_df.apply(
            lambda x: dict(zip(x['years'], x['internet_availability'])),
            axis=1
        )

        def compute_internet_availability_stats_per_year(group: pd.Series) -> dict:
            list_of_dicts = group.values

            # Get all years available
            years = {year for d in list_of_dicts for year in d.keys()}
            # Compute the counter for each year
            count_per_year = { year: Counter() for year in years }
            for d in list_of_dicts:
                for year, value in d.items():
                    count_per_year[year][value] += 1

            # Compute the ratio
            result = {}
            for year, counter in count_per_year.items():
                result[year] = counter[True] / (counter[True] + counter[False])
            return result

        group_df = df.groupby('municipality_code')
        connectivity_history_df = group_df.agg({
            'internet_availability_dict': compute_internet_availability_stats_per_year
        }).rename(columns={'internet_availability_dict': 'connectivity'})
        connectivity_history_df['connectivity_year'] = \
            connectivity_history_df['connectivity'].apply(lambda x: list(x.keys()))
        connectivity_history_df['connectivity_rate'] = \
            connectivity_history_df['connectivity'].apply(lambda x: list(x.values()))
        connectivity_history_df = connectivity_history_df.drop(columns=['connectivity'])

        connectivity_history_df['school_count'] = group_df.size()
        connectivity_history_df = connectivity_history_df.reset_index()
        return connectivity_history_df


    def setup(self) -> None:
        connecivity_history_df = self._get_connectivity_history()

        df = pd.merge(self.employability_history_df, connecivity_history_df,
                      on=self.merge_key, validate='one_to_one')
        df.rename(columns={'years': 'employability_year'}, inplace=True)

        # Guarantee that the columns are in the correct format
        df['connectivity_year'] = df['connectivity_year'].apply(np.array)
        df['connectivity_rate'] = df['connectivity_rate'].apply(np.array)
        df['employability_year'] = df['employability_year'].apply(np.array)
        df['employability_rate'] = df['employability_rate'].apply(np.array)

        # Filters
        df = df[df['school_count'].apply(lambda x: x >= 10)]
        df = df[df['employability_rate'].apply(lambda x: (x >= 10).all())]
        th = df['hdi'].describe([0.75])['75%']
        df = df[df['hdi'] <= th]

        # Columns needed
        data_columns = self.employability_history_columns + self.connectivity_history_columns
        self._dataset = df[data_columns]


    @property
    def dataset(self) -> pd.DataFrame:
        return self._dataset


class Setting:


    def __init__(self,
                 df: pd.DataFrame,
                 connectivity_range: Tuple[int, int],
                 employability_range: Tuple[int, int],
                 connectivity_threshold_A: float,
                 connectivity_threshold_B: float,
                 connectivity_col: str,
                 employability_col: str,
                 filter_A: str,
                 filter_B: str,
                 min_n_cities_test: int = 100,
                 significance_test: bool = False
                ) -> None:
        self.connectivity_range = connectivity_range
        self.employability_range = employability_range
        self.connectivity_threshold_A = connectivity_threshold_A
        self.connectivity_threshold_B = connectivity_threshold_B
        self.connectivity_col = connectivity_col
        self.employability_col = employability_col
        self.filter_A = filter_A
        self.filter_B = filter_B

        A, B = self.get_sets(df)
        self.__set_statistics(A, B, employability_col)

        self.p_value_ks_greater, self.p_value_ks_less = (np.nan, np.nan)
        if (significance_test
            and self.n_cities_A >= min_n_cities_test
            and self.n_cities_B >= min_n_cities_test):
            self.p_value_ks_greater, self.p_value_ks_less = \
                self.__get_significance_test(A, B, employability_col)


    def get_infos(self):
        infos = [self.connectivity_range[0], self.connectivity_range[1],
                 self.employability_range[0], self.employability_range[1],
                 self.connectivity_threshold_A, self.connectivity_threshold_B,
                 self.connectivity_col, self.employability_col,
                 self.filter_A, self.filter_B,
                 self.n_cities_A, self.n_cities_B,
                 self.employability_mean_A, self.employability_mean_B,
                 self.employability_max_A, self.employability_max_B,
                 self.employability_std_A, self.employability_std_B,
                 self.employability_ratio_A_B,
                 self.HDI_mean_A, self.HDI_mean_B,
                 self.HDI_std_A, self.HDI_std_B,
                 self.p_value_ks_greater,  self.p_value_ks_less]

        head  =  ['connectivity_year_start', 'connectivity_year_end',
                  'employability_year_start', 'employability_year_end',
                  'connectivity_threshold_A', 'connectivity_threshold_B',
                  'connectivity', 'employability',
                  'threshold_A', 'threshold_B',
                  'n_cities_A', 'n_cities_B',
                  'employability_mean_A', 'employability_mean_B',
                  'employability_max_A', 'employability_max_B',
                  'employability_std_A', 'employability_std_B',
                  'employability_ratio_A_B',
                  'HDI_mean_A', 'HDI_mean_B',
                  'HDI_std_A', 'HDI_std_B',
                  'pval_ks_greater', 'pval_ks_less']

        return (head, infos)


    def get_sets(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        A = df.query(self.filter_A)
        B = df.query(self.filter_B)

        threshold_A = A[self.employability_col].describe([0.98])['98%']
        threshold_B = B[self.employability_col].describe([0.98])['98%']

        A = A[A[self.employability_col] <= threshold_A]
        B = B[B[self.employability_col] <= threshold_B]

        return A, B


    def __set_statistics(self, A: pd.DataFrame, B: pd.DataFrame,
                         employability_col: str) -> None:
        self.n_cities_A = A.shape[0]
        self.n_cities_B = B.shape[0]
        self.employability_mean_A = A[employability_col].mean()
        self.employability_mean_B = B[employability_col].mean()
        self.employability_max_A = A[employability_col].max()
        self.employability_max_B = B[employability_col].max()
        self.employability_std_A = A[employability_col].std()
        self.employability_std_B = B[employability_col].std()
        self.employability_ratio_A_B = \
            self.employability_mean_A / self.employability_mean_B

        self.HDI_mean_A = A['hdi'].mean()
        self.HDI_mean_B = B['hdi'].mean()
        self.HDI_std_A = A['hdi'].std()
        self.HDI_std_B = B['hdi'].std()


    def __get_significance_test(self, A, B, employability_col):
        #test F_B > F_A -> A > B
        _, p_value_ks_g = ks_2samp(B[employability_col],
                                   A[employability_col],
                                   alternative='greater')
        p_value_ks_greater = np.round(p_value_ks_g,4)

        #test F_B < F_A -> A < B
        _, p_value_ks_l = ks_2samp(B[employability_col],
                                   A[employability_col],
                                   alternative='less')
        p_value_ks_less = np.round(p_value_ks_l,4)

        return p_value_ks_greater, p_value_ks_less


class EmployabilityImpactTemporalAnalisys:

    settings: List[Setting]
    df: pd.DataFrame


    def __init__(self, df: pd.DataFrame) -> None:
        self.settings = []
        self.df = df.copy()
        self.__create_temporal_features('connectivity_year', 'connectivity_rate',
                                        'connectivity')
        self.__create_temporal_features('employability_year', 'employability_rate',
                                        'employability')


    def __get_new_feature(self, row: pd.Series, year_column: str, rate_column: str,
                          start_year: int, end_year: int):
        years = row[year_column]
        rates = row[rate_column]

        # Assuming the lists are numpy arrays
        start_indices = np.where(years == start_year)[0]
        end_indices = np.where(years == end_year)[0]
        if start_indices.size == 0 or end_indices.size == 0:
            return np.nan

        start_idx = start_indices[0]
        end_idx = end_indices[0]
        if np.isclose(rates[start_idx], 0):
            return np.nan
        return rates[end_idx] / rates[start_idx]


    def __create_temporal_features(self,
                                   year_column: str,
                                   rate_column: str,
                                   prefix: str) -> None:
        years = self.df.iloc[0][year_column]
        years = np.sort(years)
        for i in range(years.size-1):
            start_year = years[i]
            for j in range(i+1, years.size):
                end_year = years[j]
                new_column = f'{prefix}_{start_year}_{end_year}'
                self.df[new_column] = self.df.apply(
                    self.__get_new_feature, axis=1,
                    args=(year_column, rate_column, start_year, end_year))


    def __parse_interval_column(self, col: str) -> Tuple[int, int]:
        temp = col.split('_')
        end_year = int(temp[-1])
        start_year = int(temp[-2])
        return (start_year, end_year)


    def __is_valid_range(self, con_time: Tuple[int, int], emp_time: Tuple[int, int]) -> bool:
        return con_time[0] <= emp_time[0] and con_time[1] <= emp_time[1]


    def generate_settings(self, thresholds_A_B, replace: bool=True) -> None:
        if replace:
            self.settings = []

        # Assuming no other columns with connectivity_dddd_dddd
        # and employability_dddd_dddd prefixes
        connectivity_cols = [col for col in self.df.columns
                             if re.search(r'connectivity_\d{4}_\d{4}', col)]
        employability_cols = [col for col in self.df.columns
                              if re.search(r'employability_\d{4}_\d{4}', col)]

        # Testing all combinations of range periods
        for con_col in connectivity_cols:
            for emp_col in employability_cols:
                con_range = self.__parse_interval_column(con_col)
                emp_range = self.__parse_interval_column(emp_col)
                if not self.__is_valid_range(con_range, emp_range):
                    continue

                for thA, thB in thresholds_A_B:
                    filter_A = f'{con_col}>={thA}'
                    filter_B = f'{con_col}<={thB}'
                    self.settings.append(Setting(self.df, 
                                                 con_range, emp_range,
                                                 thA, thB,
                                                 con_col, emp_col,
                                                 filter_A, filter_B,
                                                 min_n_cities_test=100,
                                                 significance_test=True))


    def get_best_setting(self, significance_test: bool=True) -> Setting:
        best_setting = None
        best_ratio = 0
        for setting in self.settings:
            if significance_test and (np.isnan(setting.p_value_ks_greater)
                or setting.p_value_ks_greater > 0.05):
                continue
            if setting.employability_ratio_A_B > best_ratio:
                best_setting = setting
                best_ratio = setting.employability_ratio_A_B
        return best_setting


    def get_result_summary(self) -> pd.DataFrame:
        # assuming the settings have already been generated
        columns = self.settings[0].get_infos()[0]
        data_df = [setting.get_infos()[1] for setting in self.settings]
        return pd.DataFrame(data_df, columns=columns)


class EmployabilityImpactOutputter:


    def get_output(self,
                   temporal_analisys_df: pd.DataFrame,
                   setting_df: pd.DataFrame,
                   best_setting: Setting,
                   significance_threshold: float = 0.05
                  ) -> Dict:
        scenario_distribution_output = \
            self.get_scenario_distribution_output(setting_df, significance_threshold)

        best_setting_output = \
            self.get_best_scenario_output(temporal_analisys_df, best_setting)

        return {
            'all_scenarios': scenario_distribution_output,
            'best_scenario': best_setting_output,
        }


    def get_scenario_distribution_output(self,
                                         setting_df: pd.DataFrame, 
                                         significance_threshold: float = 0.05
                                         ) -> Dict[str, Any]:
        greater_filter = f'(employability_ratio_A_B>1.00001 and pval_ks_greater<{significance_threshold})'
        less_filter = f'(employability_ratio_A_B<0.9999 and pval_ks_less<{significance_threshold})'

        greater_count = len(setting_df.query(greater_filter))
        less_count = len(setting_df.query(less_filter))
        not_computed_count = len(setting_df[
            (setting_df['pval_ks_less'].isna()) &
            (setting_df['pval_ks_greater'].isna())
        ])
        equal_count = len(setting_df) - greater_count - less_count - not_computed_count

        connectivity_range = [setting_df['connectivity_year_start'].min(),
                              setting_df['connectivity_year_end'].max()]
        employability_range = [setting_df['employability_year_start'].min(),
                                 setting_df['employability_year_end'].max()]

        connectivity_thresholds_A_B = setting_df[['connectivity_threshold_A',
                                                  'connectivity_threshold_B']].values.tolist()
        connectivity_thresholds_A_B = set([tuple(x) for x in connectivity_thresholds_A_B])        
        connectivity_thresholds_A_B = [f'({x[0]}, {x[1]})' for x in connectivity_thresholds_A_B]

        employability_mean_A = 100 * (setting_df['employability_mean_A'] - 1)
        employability_mean_B = 100 * (setting_df['employability_mean_B'] - 1)
        return {
            'num_scenarios': len(setting_df),
            'connectivity_range': connectivity_range,
            'employability_range': employability_range,
            'connectivity_thresholds_A_B': connectivity_thresholds_A_B,
            'employability_rate': {
                'mean_by_scenario': {
                    'A': employability_mean_A.round(2).tolist(),
                    'B': employability_mean_B.round(2).tolist(),
                },
                'is_A_greater_than_B_by_scenario': {
                    'yes': greater_count,
                    'equal': equal_count,
                    'no': less_count,
                    'not_computed': not_computed_count,
                }
            }
        }


    def __get_set_output(self,
                         set_df: pd.DataFrame,
                         connectivity_col: str,
                         employability_col: str,
                         connectivity_threshold: float,
                         ) -> Dict[str, Any]:
        connectivity_values = 100 * (set_df[connectivity_col] - 1)
        employability_values = 100 * (set_df[employability_col] - 1)
        return {
            'connectivity_threshold': connectivity_threshold,
            'num_municipalities': len(set_df),
            'municipality_name': set_df['municipality_name'].tolist(),
            'state_name': set_df['state_name'].tolist(),
            'hdi': set_df['hdi'].tolist(),
            'population_size': set_df['population_size'].tolist(),
            'connectivity_rate': connectivity_values.round(2).tolist(),
            'employability_rate': employability_values.round(2).tolist(),
        }


    def get_best_scenario_output(self,
                                 temporal_analisys_df: pd.DataFrame,
                                 best_setting: Setting
                                ) -> Dict[str, Any]:
        A, B = best_setting.get_sets(temporal_analisys_df)
        con_col = best_setting.connectivity_col
        emp_col = best_setting.employability_col
        return {
            'connectivity_range': list(best_setting.connectivity_range),
            'employability_range': list(best_setting.employability_range),
            'A': self.__get_set_output(A, con_col, emp_col, best_setting.connectivity_threshold_A),
            'B': self.__get_set_output(B, con_col, emp_col, best_setting.connectivity_threshold_B),
        }


if __name__ == '__main__':
    import sys
    args = sys.argv

    if len(args) != 3:
        print('python3 script.py <localities file> <schools file>')

    # Arguments
    employability_history_filepath = args[1]
    school_history_filepath = args[2]

    # Files
    employability_history_df = pd.read_csv(employability_history_filepath, sep=',',
                                           encoding='utf-8', dtype=object)
    school_history_df = pd.read_csv(school_history_filepath, sep=',',
                                    encoding='utf-8', dtype=object)

    # Process localities table
    employability_processor = EmployabilityHistoryTableProcessor()
    processed_employability = employability_processor.process(employability_history_df)

    # Process schools table
    processed_employability_df = processed_employability.final_df

    municipality_codes = None
    if (processed_employability_df is not None
        and 'municipality_code' in processed_employability_df.columns):
        municipality_codes = processed_employability_df['municipality_code']
        municipality_codes = set(municipality_codes.values)

    school_processor = SchoolHistoryTableProcessor(municipality_codes)
    processed_school = school_processor.process(school_history_df)

    import json
    print(processed_employability.initial_df.shape)
    print(processed_employability.final_df.shape)
    print(processed_employability.final_df.head())
    print()
    print(processed_school.initial_df.shape)
    print(processed_school.final_df.shape)
    print(processed_school.final_df.head())
    print()

    if not processed_employability.is_ok:
        print("locations")
        locality_error = {
            'is_ok': processed_employability.is_ok,
            'failure_cases': processed_employability.failure_cases.to_dict(orient='records'),
        }
        with open('employability_error.json', 'w') as f:
            f.write(json.dumps(locality_error, indent=4))

    if not processed_school.is_ok:
        print("schools")
        school_error = {
            'is_ok': processed_school.is_ok,
            'failure_cases': processed_school.failure_cases.to_dict(orient='records'),
        }
        with open('school_error.json', 'w') as f:
            f.write(json.dumps(school_error, indent=4))

    if not processed_employability.is_ok or not processed_school.is_ok:
        sys.exit(1)

    impact_dl = EmployabilityImpactDataLoader(processed_employability.final_df,
                                              processed_school.final_df)
    impact_dl.setup()

    valid_states = [
        "Alagoas",
        "Ceará",
        "Bahia",
        "Maranhão",
        "Pará" ,
        "Paraíba",
        "Pernambuco",
        "Piauí",
        "Rio Grande do Norte",
        "Sergipe",
    ]

    print(impact_dl.dataset.shape)
    states_query = ' or '.join([f'state_name == "{state}"' for state in valid_states])
    impact_dl.dataset.query(states_query, inplace=True)
    print(impact_dl.dataset.shape)

    temporal_analisys = EmployabilityImpactTemporalAnalisys(impact_dl.dataset)
    temporal_analisys.generate_settings(thresholds_A_B=[(2, 1)])

    setting_df = temporal_analisys.get_result_summary()
    best_setting = temporal_analisys.get_best_setting()

    outputter = EmployabilityImpactOutputter()
    output = outputter.get_output(temporal_analisys.df, setting_df,
                                  best_setting, 0.05)

    import json
    print(json.dumps(output['all_scenarios']))
    print()
    print(json.dumps(output['best_scenario']))

    # setting_df.query('n_cities_A>=200 and n_cities_B>=200', inplace=True)
    # setting_df.sort_values(by=['employability_ratio_A_B'], ascending=False, inplace=True)
    # with open('temporal_analisys.csv', 'w') as f:
    #     f.write(setting_df.to_csv(index=False))

    # query = '(employability_ratio_A_B>1.00001 and pval_ks_greater<0.05)'
    # query += ' or '
    # query += '(employability_ratio_A_B<0.9999 and pval_ks_less<0.05)'
    # setting_df = setting_df.query(query)

    # employability_mean_A = setting_df.employability_mean_A
    # employability_mean_B = setting_df.employability_mean_B
    # num_settings = len(setting_df)
    # improved_count = (employability_mean_A > employability_mean_B).sum()
    # print(improved_count, num_settings)
    # print(np.round((improved_count * 100) / num_settings, 4))
