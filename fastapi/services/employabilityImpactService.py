from typing import Optional, Union, List, Tuple, Set, Dict, Any
from collections import Counter
from services.utils import fast_mode, convert_to_list, parse_int, parse_boolean

from scipy.stats import ks_2samp
from sklearn.preprocessing import KBinsDiscretizer

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
        df = self._preprocess(df)
        df = self._clean_data(df)

        try:
            self.schema.validate(df, lazy=True)
            df = self._convert_dtypes(df)
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


    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        # It will throwing an error during the validation step
        columns_required = {'years', 'internet_availability'}
        if not columns_required.issubset(set(df.columns)):
            return df

        df['years'] = df['years']\
            .apply(lambda x: convert_to_list(x, parse_int))
        df['internet_availability'] = df['internet_availability']\
            .apply(lambda x: convert_to_list(x, parse_boolean))
        return df


    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
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
        years = set.union(*years) - {None, pd.NA, ''}
        df = df[df['years'].apply(lambda x: set(x) == years)]

        # Get only valid cities
        if self.municipality_codes is not None:
            df = df[df['municipality_code'].isin(self.municipality_codes)]

        # Remove redundant data
        df = df.drop_duplicates(subset=['school_code'])

        return df


    def _convert_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
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
        df = self._preprocess(df)
        df = self._clean_data(df)
        df = self._standardize_values(df)

        try:
            self.schema.validate(df, lazy=True)
            df = self._convert_dtypes(df)
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


    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        # It will throwing an error during the validation step
        columns_required = {'years', 'employability_rate'}
        if not columns_required.issubset(set(df.columns)):
            return df

        df['years'] = df['years']\
            .apply(lambda x: convert_to_list(x, parse_int))
        df['employability_rate'] = df['employability_rate']\
            .apply(lambda x: convert_to_list(x, parse_int))
        return df


    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
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
        years = set.union(*years) - {None, pd.NA, ''}
        df = df[df['years'].apply(lambda x: set(x) == years)]

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


    def setup(self, filter_data: bool = True) -> None:
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
        if filter_data:
            df = df[df['school_count'].apply(lambda x: x >= 10)]
            df = df[df['employability_rate'].apply(lambda x: (x >= 10).all())]

        # Columns needed
        data_columns = self.employability_history_columns + self.connectivity_history_columns
        self._dataset = df[data_columns]


    @property
    def dataset(self) -> pd.DataFrame:
        return self._dataset


class Homogenizer:


    def __init__(self,
                 A: pd.DataFrame,
                 B: pd.DataFrame,
                 continuous_features: List[str] = [],
                 categorical_features: List[str] =[],
                 min_size_A: int =50,
                 min_size_B: int=50,
                 n_bins=10
                ) -> None:

        self.A = A.copy()
        self.B = B.copy()

        self.A.reset_index(drop=True, inplace=True)
        self.B.reset_index(drop=True, inplace=True)

        self.min_size_A = min_size_A
        self.min_size_B = min_size_B
        self.n_bins = n_bins

        self.features =  [feature for feature in continuous_features]
        self.features += [feature for feature in categorical_features]
        self.features = np.array(self.features)

        #quando removemos estados com < 10 cidades, mexemos na distribuição de features
        # pode ser interessante discretizar só após isso
        if len(continuous_features) > 0:
            discretized_A, discretized_B = self.get_discrete_features(A, B, continuous_features)
            self.A[continuous_features] = discretized_A
            self.B[continuous_features] = discretized_B


    def get_discrete_features(self, A, B, features):
        discretizer = KBinsDiscretizer(n_bins=self.n_bins, encode='ordinal', strategy='uniform')
        discretizer.fit(np.concatenate((A[features], B[features]), axis=0))
        discretized_A = discretizer.transform(A[features].values).astype(int)
        discretized_B = discretizer.transform(B[features].values).astype(int)
        return discretized_A, discretized_B


    def get_frequence(self, values):
        unique, counts = np.unique(values, return_counts=True)
        frequence = {unique[i]:freq for i,freq in enumerate(counts)}
        return frequence


    def kl_divergence(self, p: Dict[Any, float], q: Dict[Any, float],
                      sum_p: float, sum_q: float) -> float:
        divergence = 0
        for key in p:
            assert key in q
            if p[key]==0 or q[key]==0:
                divergence += 0
            else:
                p_perc = (p[key]/sum_p)
                q_perc = (q[key]/sum_q)
                divergence += p_perc * np.log2(p_perc / q_perc)
        return divergence


    def JS_divergence(self, A_freq: Dict[Any, float], B_freq: Dict[Any, float]) -> float:
        # compute distributions of attribute for A and B
        keys = np.unique([list(A_freq.keys()) + list(B_freq.keys())])
        sum_A = sum(A_freq.values())
        sum_B = sum(B_freq.values())

        M = {}
        for key in keys:
            if key not in A_freq:
                A_freq[key] = 0
            if key not in B_freq:
                B_freq[key] = 0
            M[key] = 0.5 * (sum_B*A_freq[key] + sum_A*B_freq[key])

        # compute JS divergence
        A_div = self.kl_divergence(A_freq, M, sum_A, sum_A*sum_B)
        B_div = self.kl_divergence(B_freq, M, sum_B, sum_A*sum_B)
        JS = 0.5 * (A_div + B_div)
        assert JS <= 1.0001

        return JS


    def get_divergence(self, A_frequence, B_frequence):
        return sum(self.JS_divergence(A_frequence[feature], B_frequence[feature])
                   for feature in self.features)


    def remove_states(self):
        A_states, A_counts = np.unique(self.A['state_name'].values, return_counts=True)
        B_states, B_counts = np.unique(self.B['state_name'].values, return_counts=True)

        mask_A = A_counts >= 10
        mask_B = B_counts >= 10
        states = list(set(A_states[mask_A]) & set(B_states[mask_B]))

        return ~self.A['state_name'].isin(states).values, \
               ~self.B['state_name'].isin(states).values


    def build_hash_series(self, combination_values):
        self.hash_map = {tuple(value): i for i, value in enumerate(combination_values)}

        def create_hash(row: pd.Series) -> int:
            tuple_key = tuple([row[col] for col in self.features])
            return self.hash_map.get(tuple_key, -1)
        self.A['hash'] = self.A.apply(create_hash, axis=1)
        self.B['hash'] = self.B.apply(create_hash, axis=1)

        self.hash_to_cities_A = self.A['hash'].reset_index().groupby('hash')['index'].agg(list)
        self.hash_to_cities_B = self.B['hash'].reset_index().groupby('hash')['index'].agg(list)


    def get_divergence_city(self, frequence, min_size, row_values, count_remainder, hash_to_cities):
        divergence = np.infty
        city = -1
        flag = ((count_remainder > min_size) and
                (self.hash_map[tuple(row_values)] in hash_to_cities.index) and
                (len(hash_to_cities[self.hash_map[tuple(row_values)]]) > 0))

        if flag:
            for i, feature in enumerate(self.features):
                frequence[feature][row_values[i]] -= 1

            divergence = self.get_divergence(self.A_frequence, self.B_frequence)
            #city = hash_to_cities[hash(tuple(row_values))][0]
            city = hash_to_cities[self.hash_map[tuple(row_values)]][0]

            for i, feature in enumerate(self.features):
                frequence[feature][row_values[i]] += 1

        return [divergence, city]



    def get_homogenized_sets(self):
        removed_A, removed_B = self.remove_states()

        count_remainder_A = (~removed_A).sum()
        count_remainder_B = (~removed_B).sum()

        self.A_frequence = {feature: self.get_frequence(self.A[~removed_A][feature].values)
                            for feature in self.features}
        self.B_frequence = {feature: self.get_frequence(self.B[~removed_B][feature].values)
                            for feature in self.features}

        combination_values = pd.concat([self.A.loc[~removed_A, self.features],
                                        self.B.loc[~removed_B, self.features]])\
                                .drop_duplicates().values

        self.build_hash_series(combination_values)

        current_div = self.get_divergence(self.A_frequence, self.B_frequence)

        while (count_remainder_A > self.min_size_A)  or  (count_remainder_B > self.min_size_B):
            A_divergences = np.array([
                self.get_divergence_city(self.A_frequence, self.min_size_A, row_values,
                                         count_remainder_A, self.hash_to_cities_A)
                for row_values in combination_values
            ])

            B_divergences = np.array([
                self.get_divergence_city(self.B_frequence, self.min_size_B, row_values,
                                         count_remainder_B, self.hash_to_cities_B)
                for row_values in combination_values
            ])

            min_ind_A = np.argmin(A_divergences[:,0])
            min_ind_B = np.argmin(B_divergences[:,0])

            min_div_A = A_divergences[min_ind_A][0]
            min_div_B = B_divergences[min_ind_B][0]

            city_A = int(A_divergences[min_ind_A][1])
            city_B = int(B_divergences[min_ind_B][1])

            if current_div <= min(min_div_A, min_div_B)+0.00000001:
                break

            elif min_div_A < min_div_B:
                removed_A[city_A] = True
                count_remainder_A -= 1
                current_div = min_div_A
                row_values = combination_values[min_ind_A]
                for i,feature in enumerate(self.features):
                    self.A_frequence[feature][row_values[i]] -=1
                self.hash_to_cities_A[self.hash_map[tuple(row_values)]].pop(0)
            else:
                removed_B[city_B] = True
                count_remainder_B -= 1
                current_div = min_div_B
                row_values = combination_values[min_ind_B]
                for i,feature in enumerate(self.features):
                    self.B_frequence[feature][row_values[i]] -=1
                self.hash_to_cities_B[self.hash_map[tuple(row_values)]].pop(0)

        return ~removed_A, ~removed_B


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
                 num_cities_threshold: int = 100,
                 significance_test: bool = False,
                 homogenize_sets: bool = True
                ) -> None:
        self.connectivity_range = connectivity_range
        self.employability_range = employability_range
        self.connectivity_threshold_A = connectivity_threshold_A
        self.connectivity_threshold_B = connectivity_threshold_B
        self.connectivity_col = connectivity_col
        self.employability_col = employability_col
        self.filter_A = filter_A
        self.filter_B = filter_B
        self.num_cities_threshold = num_cities_threshold
        self.homogenize_sets = homogenize_sets

        A, B = self.get_sets(df)
        self._set_statistics(A, B, employability_col)

        self.p_value_ks_greater, self.p_value_ks_less = (np.nan, np.nan)
        if (significance_test
            and self.n_cities_A >= num_cities_threshold
            and self.n_cities_B >= num_cities_threshold):
            self.p_value_ks_greater, self.p_value_ks_less = \
                self._get_significance_test(A, B, employability_col)


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

        if (len(A) < self.num_cities_threshold 
            or len(B) < self.num_cities_threshold
            or self.homogenize_sets == False):
            return A, B

        homogenizer = Homogenizer(
            A, B,
            continuous_features = ['hdi', 'population_size'],
            categorical_features = ['state_name'],
            min_size_A = self.num_cities_threshold,
            min_size_B = self.num_cities_threshold,
            n_bins = 5
        )
        ind_A, ind_B = homogenizer.get_homogenized_sets()
        return A[ind_A], B[ind_B]


    def _set_statistics(self, A: pd.DataFrame, B: pd.DataFrame,
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


    def _get_significance_test(self, A, B, employability_col):
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
        self._create_temporal_features('connectivity_year', 'connectivity_rate',
                                        'connectivity')
        self._create_temporal_features('employability_year', 'employability_rate',
                                        'employability')


    def _get_new_feature(self, row: pd.Series, year_column: str, rate_column: str,
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


    def _create_temporal_features(self,
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
                    self._get_new_feature, axis=1,
                    args=(year_column, rate_column, start_year, end_year))


    def _parse_interval_column(self, col: str) -> Tuple[int, int]:
        temp = col.split('_')
        end_year = int(temp[-1])
        start_year = int(temp[-2])
        return (start_year, end_year)


    def _is_valid_range(self, con_time: Tuple[int, int], emp_time: Tuple[int, int]) -> bool:
        return con_time[0] <= emp_time[0] and con_time[1] <= emp_time[1]


    def generate_settings(self,
                          thresholds_A_B: List[Tuple[float, float]],
                          num_cities_threshold: int=100,
                          replace: bool=True) -> None:
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
                con_range = self._parse_interval_column(con_col)
                emp_range = self._parse_interval_column(emp_col)
                if not self._is_valid_range(con_range, emp_range):
                    continue

                for thA, thB in thresholds_A_B:
                    filter_A = f'{con_col}>={thA}'
                    filter_B = f'{con_col}<={thB}'
                    self.settings.append(Setting(
                        self.df,
                        con_range, emp_range,
                        thA, thB,
                        con_col, emp_col,
                        filter_A, filter_B,
                        num_cities_threshold=num_cities_threshold,
                        significance_test=True
                    ))


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

        connectivity_range = [int(setting_df['connectivity_year_start'].min()),
                              int(setting_df['connectivity_year_end'].max())]
        employability_range = [int(setting_df['employability_year_start'].min()),
                               int(setting_df['employability_year_end'].max())]

        connectivity_thresholds_A_B = setting_df[['connectivity_threshold_A',
                                                  'connectivity_threshold_B']].values.tolist()
        connectivity_thresholds_A_B = set([tuple(x) for x in connectivity_thresholds_A_B])
        connectivity_thresholds_A_B = [f'({x[0]}, {x[1]})' for x in connectivity_thresholds_A_B]

        valida_setting_df = setting_df[(~setting_df['pval_ks_less'].isna()) |
                                       (~setting_df['pval_ks_greater'].isna())]
        employability_mean_A = 100 * (valida_setting_df['employability_mean_A'] - 1)
        employability_mean_B = 100 * (valida_setting_df['employability_mean_B'] - 1)
        return {
            'num_scenarios': len(setting_df),
            'connectivity_range': connectivity_range,
            'employability_range': employability_range,
            'connectivity_thresholds_A_B': connectivity_thresholds_A_B,
            'employability_rate': {
                'mean_by_valid_scenario': {
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


    def _get_set_output(self,
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
            'A': self._get_set_output(A, con_col, emp_col, best_setting.connectivity_threshold_A),
            'B': self._get_set_output(B, con_col, emp_col, best_setting.connectivity_threshold_B),
        }
