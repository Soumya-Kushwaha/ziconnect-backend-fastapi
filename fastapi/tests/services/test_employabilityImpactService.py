import copy
import pandas as pd
import unittest

from services.employabilityImpactService import (
    ProcessedTable,    
    EmployabilityHistoryTableProcessor,
    SchoolHistoryTableProcessor
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
        print(school_history_df.head())
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


if __name__ == '__main__':
    unittest.main()
