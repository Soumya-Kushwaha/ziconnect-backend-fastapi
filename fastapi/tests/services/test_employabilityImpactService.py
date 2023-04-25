import copy
import numpy as np
import pandas as pd
import unittest

from services.employabilityImpactService import (
    ProcessedTable,
    EmployabilityHistoryTableProcessor,
    SchoolHistoryTableProcessor,    
    EmployabilityImpactDataLoader,
    Homogenizer,
    Setting,
    EmployabilityImpactTemporalAnalisys,
    EmployabilityImpactOutputter
)


class TestProcessedTable(unittest.TestCase):


    def test_init(self):
        is_ok = True
        initial_df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6],})
        final_df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6],})
        failure_cases_df = None

        processed_table = ProcessedTable(is_ok, initial_df, final_df, failure_cases_df)
        self.assertEqual(processed_table.is_ok, is_ok)
        self.assertTrue(processed_table.initial_df.equals(initial_df))
        self.assertTrue(processed_table.final_df.equals(final_df))
        self.assertIsNone(processed_table.failure_cases)


class TestEmployabilityHistoryTableProcessor(unittest.TestCase):


    def setUp(self) -> None:
        self.table_processor = EmployabilityHistoryTableProcessor()

        localities = [
            ('FR', 'France', 'FR-01', 'Ain', 'FR-01-001', 'Bourg-en-Bresse', 0.7, 10000, [2016,2017], [1000, 2000]),
            ('FR', 'France', 'FR-02', 'Aisne', 'FR-02-001', 'Laon', 0.7, 10000, [2016,2017], [1000, 2000]),
            ('FR', 'France', 'FR-03', 'Allier', 'FR-03-001', 'Moulins', 0.7, 10000, [2016,2017], [1000, 2000]),
            ('US', 'United States', 'US-CA', 'California', 'US-CA-001', 'Los Angeles', 0.7, 10000, [2016,2017], [1000, 2000]),
            ('US', 'United States', 'US-CA', 'California', 'US-CA-001', 'Los Angeles', 0.7, 10000, [2016,2017], [1000, 2000]),

            ('FR', 'France', 'FR-02', 'Aisne', 'FR-02-001', 'Laon', 0.7, 10000, [None], [1000, 2000]),
            ('FR', 'France', 'FR-02', 'Aisne', 'FR-02-001', 'Laon', 0.7, 10000, [2016,2017], None),
            ('FR', 'France', 'FR-02', 'Aisne', 'FR-02-001', 'Laon', 0.7, 10000, [2016], [1000]),
            ('US', 'United States', 'US-CA', 'California', 'US-CA-002', 'San Francisco', 10000, None, [2016,2017], [1000, 2000]),
            ('US', 'United States', 'US-CA', 'California', 'US-CA-002', 'San Francisco', None, 10000, [2016,2017], [1000, 2000]),
            ('US', 'United States', 'US-CA', 'California', None, 'San Diego', 0.7, 10000, [2016,2017], [1000, 2000]),
            ('US', None, 'US-CA', 'California', 'US-CA-004', 'San Jose', 0.7, 10000, [2016,2017], [1000, 2000]),
        ]

        columns = ['country_code', 'country_name', 'state_code', 'state_name',
                   'municipality_code', 'municipality_name', 'hdi', 'population_size',
                   'years', 'employability_rate']
        self.employability_history_df = pd.DataFrame(localities, columns=columns)


    def test_standardize_values(self):
        employability_history_df = copy.deepcopy(self.employability_history_df)
        employability_history_df.iloc[2]['country_name'] = 'Frances'

        standardized_df = self.table_processor._standardize_values(employability_history_df)

        country_codes = pd.Series(['FR', 'FR', 'FR', 'US', 'US',
                                   'FR', 'FR', 'FR', 'US', 'US', 'US', 'US'])
        self.assertTrue(standardized_df['country_code'].equals(country_codes))

        country_names = pd.Series(['France', 'France', 'France', 'United States', 'United States',
                                   'France', 'France', 'France', 'United States', 'United States',
                                   'United States','United States'])
        self.assertTrue(standardized_df['country_name'].equals(country_names))


    def test_clean_data(self):
        employability_history_df = copy.deepcopy(self.employability_history_df)

        cleaned_df = self.table_processor._clean_data(employability_history_df)
        self.assertEqual(len(cleaned_df), 5)


    def test_convert_dtypes(self):
        employability_history_df = copy.deepcopy(self.employability_history_df)
        employability_history_df = employability_history_df.head(5)

        converted_df = self.table_processor._convert_dtypes(employability_history_df)

        column_dtype_dict = converted_df.dtypes.to_dict()
        self.assertEqual(column_dtype_dict['country_code'], 'string')
        self.assertEqual(column_dtype_dict['country_name'], 'string')
        self.assertEqual(column_dtype_dict['state_code'], 'string')
        self.assertEqual(column_dtype_dict['state_name'], 'string')
        self.assertEqual(column_dtype_dict['municipality_code'], 'string')
        self.assertEqual(column_dtype_dict['municipality_name'], 'string')
        self.assertEqual(column_dtype_dict['hdi'], 'float')
        self.assertEqual(column_dtype_dict['population_size'], 'int')
        self.assertEqual(column_dtype_dict['years'], 'object')
        self.assertEqual(column_dtype_dict['employability_rate'], 'object')


    def test_process(self):
        employability_history_df = copy.deepcopy(self.employability_history_df)

        processed_table = self.table_processor.process(employability_history_df)
        self.assertTrue(processed_table.is_ok)
        self.assertTrue(processed_table.initial_df.equals(employability_history_df))
        self.assertEqual(len(processed_table.final_df), 5)
        self.assertIsNone(processed_table.failure_cases)


class TestSchoolHistoryTableProcessor(unittest.TestCase):


    def setUp(self) -> None:
        self.table_processor = SchoolHistoryTableProcessor(None)

        schools = [
            (1,    'School 1', 'FR-01-001', [2016,2017], [True, False]),
            (2,    'School 2', 'FR-01-002', [2016,2017], [False, True]),
            (3,    'School 3', 'FR-01-001', [2016,2017], [True, False]),
            (4,    'School 4', 'FR-01-002', [2016,2017], [False, True]),

            (4,    'School 4', 'FR-01-002', [2016,2017], [False, True]),
            (5,    'School 5',        None, [2016,2017], [False, True]),
            (6,    'School 6', 'FR-01-001', [None], [True, False]),
            (7,    'School 7', 'FR-01-001', [2016,2017], [None, True]),
            (8,    'School 8', 'FR-01-001', [None], [None]),
        ]

        columns = ['school_code', 'school_name', 'municipality_code', 'years', 'internet_availability']
        self.school_history_df = pd.DataFrame(schools, columns=columns)


    def test_clean_data(self):
        school_history_df = copy.deepcopy(self.school_history_df)
        cleaned_df = self.table_processor._clean_data(school_history_df)
        self.assertEqual(len(cleaned_df), 4)


    def test_convert_dtypes(self):
        school_history_df = copy.deepcopy(self.school_history_df)
        school_history_df = school_history_df.head(4)

        converted_df = self.table_processor._convert_dtypes(school_history_df)

        column_dtype_dict = converted_df.dtypes.to_dict()
        self.assertEqual(column_dtype_dict['school_code'], 'string')
        self.assertEqual(column_dtype_dict['school_name'], 'string')
        self.assertEqual(column_dtype_dict['municipality_code'], 'string')
        self.assertEqual(column_dtype_dict['years'], 'object')
        self.assertEqual(column_dtype_dict['internet_availability'], 'object')


    def test_process(self):
        school_history_df = copy.deepcopy(self.school_history_df)

        processed_table = self.table_processor.process(school_history_df)
        self.assertTrue(processed_table.is_ok)
        self.assertTrue(processed_table.initial_df.equals(school_history_df))
        self.assertEqual(len(processed_table.final_df), 4)
        self.assertIsNone(processed_table.failure_cases)


class TestEmployabilityImpactDataLoader(unittest.TestCase):


    def setUp(self) -> None:
        localities = [
            ('FR', 'France', 'FR-01', 'Ain', 'FR-01-001', 'Bourg-en-Bresse', 0.7, 10000, [2016,2017], [1000, 2000]),
            ('FR', 'France', 'FR-01', 'Aisne', 'FR-01-002', 'Laon', 0.7, 10000, [2016,2017], [1000, 2000]),
            ('US', 'United States', 'US-CA', 'California', 'US-CA-001', 'Los Angeles', 0.7, 10000, [2016,2017], [1000, 2000]),
        ]

        columns = ['country_code', 'country_name', 'state_code', 'state_name',
                   'municipality_code', 'municipality_name', 'hdi', 'population_size',
                   'years', 'employability_rate']
        self.employability_history_df = pd.DataFrame(localities, columns=columns)

        schools = [
            (1, 'School 1', 'FR-01-001', [2016,2017], [True, False]),
            (2, 'School 2', 'FR-01-002', [2016,2017], [False, True]),
            (3, 'School 3', 'FR-01-001', [2016,2017], [True, False]),
            (4, 'School 4', 'FR-01-002', [2016,2017], [True, False]),
            (5, 'School 5', 'US-CA-001', [2016,2017], [True, True]),
        ]

        columns = ['school_code', 'school_name', 'municipality_code', 'years', 'internet_availability']
        self.school_history_df = pd.DataFrame(schools, columns=columns)

        self.data_loader = EmployabilityImpactDataLoader(self.employability_history_df,
                                                         self.school_history_df)


    def _search_municipality_in_df(self, df: pd.DataFrame, municipality_code: str) -> dict:
        result = df[df['municipality_code'] == municipality_code]
        if len(result) == 0:
            return {}
        return result.iloc[0].to_dict()


    def test_get_connectivity_history(self):
        connectivity_history_df = self.data_loader._get_connectivity_history()

        self.assertEqual(len(connectivity_history_df), 3)

        row_dict = self._search_municipality_in_df(connectivity_history_df, 'FR-01-001')
        self.assertDictEqual(row_dict, {
            'municipality_code': 'FR-01-001',
            'school_count': 2,
            'connectivity_year': [2016, 2017],
            'connectivity_rate': [1.0, 0.0]
        })

        row_dict = self._search_municipality_in_df(connectivity_history_df, 'FR-01-002')
        self.assertDictEqual(row_dict, {
            'municipality_code': 'FR-01-002',
            'school_count': 2,
            'connectivity_year': [2016, 2017],
            'connectivity_rate': [0.5, 0.5]
        })

        row_dict = self._search_municipality_in_df(connectivity_history_df, 'US-CA-001')
        self.assertDictEqual(row_dict, {
            'municipality_code': 'US-CA-001',
            'school_count': 1,
            'connectivity_year': [2016, 2017],
            'connectivity_rate': [1.0, 1.0]
        })


    def test_setup(self):
        self.data_loader.setup(filter_data=False)
        dataset = self.data_loader.dataset

        self.assertEqual(len(dataset), 3)

        def assertDictEqual(d1: dict, d2: dict):
            self.assertEqual(len(d1), len(d2))
            self.assertEqual(set(d1.keys()), set(d2.keys()))
            for key in d1.keys():
                if isinstance(d1[key], np.ndarray):
                    self.assertTrue(np.all(d1[key] == d2[key]))
                else:
                    self.assertEqual(d1[key], d2[key])

        row_dict = self._search_municipality_in_df(dataset, 'FR-01-001')
        assertDictEqual(row_dict, {
            'country_code': 'FR',
            'country_name': 'France',
            'state_code': 'FR-01',
            'state_name': 'Ain',
            'municipality_code': 'FR-01-001',
            'municipality_name': 'Bourg-en-Bresse',
            'hdi': 0.7,
            'population_size': 10000,
            'employability_year': np.array([2016, 2017]),
            'employability_rate': np.array([1000, 2000]),
            'connectivity_year': np.array([2016, 2017]),
            'connectivity_rate': np.array([1.0, 0.0]),
            'school_count': 2
        })

        row_dict = self._search_municipality_in_df(dataset, 'FR-01-002')
        assertDictEqual(row_dict, {
            'country_code': 'FR',
            'country_name': 'France',
            'state_code': 'FR-01',
            'state_name': 'Aisne',
            'municipality_code': 'FR-01-002',
            'municipality_name': 'Laon',
            'hdi': 0.7,
            'population_size': 10000,
            'employability_year': np.array([2016, 2017]),
            'employability_rate': np.array([1000, 2000]),
            'school_count': 2,
            'connectivity_year': np.array([2016, 2017]),
            'connectivity_rate': np.array([0.5, 0.5])
        })

        row_dict = self._search_municipality_in_df(dataset, 'US-CA-001')
        assertDictEqual(row_dict, {
            'country_code': 'US',
            'country_name': 'United States',
            'state_code': 'US-CA',
            'state_name': 'California',
            'municipality_code': 'US-CA-001',
            'municipality_name': 'Los Angeles',
            'hdi': 0.7,
            'population_size': 10000,
            'employability_year': np.array([2016, 2017]),
            'employability_rate': np.array([1000, 2000]),
            'school_count': 1,
            'connectivity_year': np.array([2016, 2017]),
            'connectivity_rate': np.array([1.0, 1.0])
        })


class TestSetting(unittest.TestCase):
    

    def setUp(self):
        records = [
            (0.50, 2.5, 1.15),
            (0.70, 3.0, 1.10),
            (0.60, 2.3, 1.05),
            (0.80, 2.0, 1.20),

            (0.70, 1.4, 1.15),
            (0.70, 1.2, 1.15),
            (0.70, 1.7, 1.15),

            (0.80, 1.0, 1.05),
            (0.60, 0.8, 1.00),
            (0.70, 0.5, 1.20),
        ]
        columns = ['hdi', 'connectivity_2010_2015', 'employability_2016_2020']
        self.setting_df = pd.DataFrame(records, columns=columns)

        self.connectivity_range = (2010, 2015)
        self.employability_range = (2016, 2020)
        self.connectivity_threshold_A = 2.0
        self.connectivity_threshold_B = 1.0
        self.connectivity_col = 'connectivity_2010_2015'
        self.employability_col = 'employability_2016_2020'
        self.filter_A = 'connectivity_2010_2015>=2.0'
        self.filter_B = 'connectivity_2010_2015<=1.0'
        self.min_n_cities_test = 2
        self.significance_test = True
        self.homogenize_sets = False

        self.setting = Setting(
            self.setting_df,
            self.connectivity_range,
            self.employability_range,
            self.connectivity_threshold_A,
            self.connectivity_threshold_B,
            self.connectivity_col,
            self.employability_col,
            self.filter_A,
            self.filter_B,
            self.min_n_cities_test,
            self.significance_test,
            self.homogenize_sets
        )


    def test_get_sets(self):
        A, B = self.setting.get_sets(self.setting_df)
        self.assertEqual(len(A), 3)
        self.assertEqual(len(B), 2)


    def test_get_infos(self):
        columns, row = self.setting.get_infos()

        self.assertEqual(len(columns), 25)
        self.assertEqual(len(row), 25)

        row_dict = dict(zip(columns, row))

        def assertDictEqual(d1: dict, d2: dict):
            self.assertEqual(len(d1), len(d2))
            self.assertEqual(set(d1.keys()), set(d2.keys()))
            for key in d1.keys():
                if isinstance(d1[key], float):
                    if np.isnan(d1[key]) and np.isnan(d2[key]):
                        continue
                    self.assertAlmostEqual(d1[key], d2[key], places=3)
                else:
                    self.assertEqual(d1[key], d2[key])

        assertDictEqual(row_dict, {
            'connectivity_year_start': self.connectivity_range[0],
            'connectivity_year_end': self.connectivity_range[1],
            'employability_year_start': self.employability_range[0],
            'employability_year_end': self.employability_range[1],
            'connectivity_threshold_A': self.connectivity_threshold_A,
            'connectivity_threshold_B': self.connectivity_threshold_B,
            'connectivity': self.connectivity_col,
            'employability': self.employability_col,
            'threshold_A': self.filter_A,
            'threshold_B': self.filter_B,
            'n_cities_A': 3,
            'n_cities_B': 2,
            'employability_mean_A': 1.10,
            'employability_mean_B': 1.025,
            'employability_max_A': 1.15,
            'employability_max_B': 1.05,
            'employability_std_A': 0.05,
            'employability_std_B': 0.03535,
            'employability_ratio_A_B': 1.10 / 1.025,
            'HDI_mean_A': 0.6,
            'HDI_mean_B': 0.7,
            'HDI_std_A': 0.1,
            'HDI_std_B': 0.1414,
            'pval_ks_greater': 0.3,
            'pval_ks_less': 1.0
        })


class TestEmployabilityImpactTemporalAnalisys(unittest.TestCase):
    

    def setUp(self) -> None:
        connectivity_years = np.array([2007, 2008])
        employability_years = np.array([2014, 2015, 2016])
        records = [
            [connectivity_years, np.array([0.6, 0.9]), employability_years, np.array([1000, 2000, 3000])],
            [connectivity_years, np.array([0.5, 0.5]), employability_years, np.array([2000, 1000, 1500])],
        ]
        columns = ['connectivity_year', 'connectivity_rate', 'employability_year', 'employability_rate']

        self.raw_df = pd.DataFrame(records, columns=columns)
        self.connectivity_years = connectivity_years
        self.employability_years = employability_years


    def test_create_temporal_features(self):
        temporal_analisys = EmployabilityImpactTemporalAnalisys(self.raw_df)

        expected_columns = {
            'connectivity_2007_2008',
            'employability_2014_2015',
            'employability_2015_2016',
            'employability_2014_2016',
        }

        columns = set(temporal_analisys.df.columns.to_list())
        self.assertTrue(expected_columns.issubset(columns))

        self.assertEqual(temporal_analisys.df['connectivity_2007_2008'].to_list(), [1.5, 1.0])
        self.assertEqual(temporal_analisys.df['employability_2014_2015'].to_list(), [2.0, 0.5])
        self.assertEqual(temporal_analisys.df['employability_2015_2016'].to_list(), [1.5, 1.5])
        self.assertEqual(temporal_analisys.df['employability_2014_2016'].to_list(), [3.0, 0.75])


    def test_parse_interval_column(self):
        temporal_analisys = EmployabilityImpactTemporalAnalisys(self.raw_df)

        self.assertEqual(temporal_analisys._parse_interval_column('connectivity_2007_2008'), (2007, 2008))
        self.assertEqual(temporal_analisys._parse_interval_column('employability_2014_2015'), (2014, 2015))
        self.assertEqual(temporal_analisys._parse_interval_column('employability_2015_2016'), (2015, 2016))
        self.assertEqual(temporal_analisys._parse_interval_column('employability_2014_2016'), (2014, 2016))


    def test_is_valid_range(self):
        temporal_analisys = EmployabilityImpactTemporalAnalisys(self.raw_df)

        self.assertTrue(temporal_analisys._is_valid_range((2007, 2008), (2007, 2008)))
        self.assertTrue(temporal_analisys._is_valid_range((2007, 2008), (2007, 2009)))
        self.assertTrue(temporal_analisys._is_valid_range((2007, 2008), (2009, 2010)))
        self.assertFalse(temporal_analisys._is_valid_range((2008, 2010), (2007, 2010)))


class TestEmployabilityImpactOutputter(unittest.TestCase):
    pass


class TestHomogenizer(unittest.TestCase):


    def test_get_homogenized_sets(self):
        #Define set A
        connectivity_years = np.array([2007, 2008, 2009, 2012, 2014, 2016])
        connectivity_rates = np.array([0.6, 0.64, 0.71, 0.75, 0.78, 0.8])
        employability_years = np.array([2014, 2015, 2016])
        employability_rates = np.array([1300, 1400, 1500])

        A_test = [(f'city {i}', 'CE', 0.6, 100000, connectivity_years, connectivity_rates,
                   employability_years, employability_rates) for i in range(10)]

        #Sets set A equals set B
        B_test = [(f'city {i}', 'CE', 0.6, 100000, connectivity_years, connectivity_rates,
                   employability_years, employability_rates) for i in range(10, 20)]

        columns = ['municipality_name', 'state_name','hdi', 'population_size',
                   'connectivity_years', 'connectivity_rate',
                   'employability_years', 'employability_rate']

        A_test = pd.DataFrame(data=A_test, columns=columns)
        B_test = pd.DataFrame(data=B_test, columns=columns)

        H = Homogenizer(A_test, B_test, continuous_features=['hdi'],
                        min_size_A=1, min_size_B=1, n_bins=5)
        idx_A, idx_B = H.get_homogenized_sets()

        # The sets are already homogeneous and no city should be discarded
        self.assertTrue(idx_A.all())
        self.assertTrue(idx_B.all())

        # Adding a city with a different HDI in A
        new_city = ['new_city', 'CE', 0.65, 100000, connectivity_years, connectivity_rates,
                    employability_years, employability_rates]
        A_test.loc[len(A_test)] = new_city

        H = Homogenizer(A_test, B_test, continuous_features=['hdi', 'population_size'],
                        categorical_features=['state_name'], min_size_A=1, min_size_B=1, n_bins=5)
        idx_A, idx_B = H.get_homogenized_sets()

        # Only the new city must have been discarded
        self.assertEqual(idx_A.sum(), len(A_test)-1)
        self.assertTrue(idx_B.all())
        self.assertFalse(idx_A[len(A_test)-1])


    def test_get_discrete_features(self):
        #buckets
        #attr1: [1, 4), [4, 7), [7, 10]
        #attr2: [1, 5), [5, 9), [9, 13]
        A_test = pd.DataFrame({'attr1': [1, 2, 3], 'attr2': [9, 11, 13], 'attr3': [1, 2, 3]})
        B_test = pd.DataFrame({'attr1': [7, 8, 10], 'attr2': [1, 5, 9], 'attr3': [4, 5, 6]})

        H = Homogenizer(A_test, B_test, continuous_features=['attr1', 'attr2'], n_bins=3)

        self.assertListEqual(A_test.attr3.values.tolist(), H.A.attr3.values.tolist())
        self.assertListEqual(H.A.attr1.values.tolist(), [0, 0, 0])
        self.assertListEqual(H.B.attr1.values.tolist(), [2, 2, 2])
        self.assertListEqual(H.A.attr2.values.tolist(), [2, 2, 2])
        self.assertListEqual(H.B.attr2.values.tolist(), [0, 1, 2])


    def test_get_frequence(self):
        df = pd.DataFrame({'attr1': [1, 0, 0], 'attr2': [1, 1, 1]})
        H = Homogenizer(df, df)

        freq = H.get_frequence(df.attr1.values)
        self.assertDictEqual(freq, {0: 2, 1: 1})


    def test_kl_divergence(self):
        H = Homogenizer(pd.DataFrame(), pd.DataFrame())
        p = {'apple': 0.1, 'banana': 0.4, 'orange': 0.5}
        q = {'apple': 0.8, 'banana': 0.15, 'orange': 0.05}
        sum_p = sum(p.values())
        sum_q = sum(q.values())

        kl_div = H.kl_divergence(p, q, sum_p, sum_q)
        self.assertAlmostEqual(kl_div, 1.927, places=3)

        kl_div = H.kl_divergence(q, p, sum_q, sum_p)
        self.assertAlmostEqual(kl_div, 2.022, places=3)


    def test_JS_divergence(self):
        H = Homogenizer(pd.DataFrame(), pd.DataFrame())
        p = {'apple': 0.36, 'banana': 0.48, 'orange': 0.16}
        q = {'apple': 0.3, 'banana': 0.5, 'orange': 0.2}

        js_div = H.JS_divergence(p, q)

        self.assertAlmostEqual(js_div, 0.0037235634420310863, places=10)


if __name__ == '__main__':
    unittest.main()
