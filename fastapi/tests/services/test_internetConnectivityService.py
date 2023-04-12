import copy
import unittest
import pandas as pd

from services.internetConnectivityService import (
    ProcessedTable,
    LocalityTableProcessor,
    SchoolTableProcessor,
    StudentCountEstimator
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


class TestLocalityTableProcesser(unittest.TestCase):


    def setUp(self) -> None:
        self.locality_table_processor = LocalityTableProcessor()

        localities = [
            ('FR', 'France', 'FR-01', 'Ain', 'FR-01-001', 'Bourg-en-Bresse'),
            ('FR', 'France', 'FR-02', 'Aisne', 'FR-02-001', 'Laon'),
            ('FR', 'France', 'FR-03', 'Allier', 'FR-03-001', 'Moulins'),
            ('US', 'United States', 'US-CA', 'California', 'US-CA-001', 'Los Angeles'),
            ('US', 'United States', 'US-CA', 'California', 'US-CA-001', 'Los Angeles'),
            ('US', 'United States', 'US-CA', 'California', 'US-CA-002', 'San Francisco'),
            ('US', 'United States', 'US-CA', 'California', None, 'San Diego'),
            ('US', None, 'US-CA', 'California', 'US-CA-004', 'San Jose'),
        ]

        columns = ['country_code', 'country_name', 'state_code', 'state_name',
                   'municipality_code', 'municipality_name']
        self.locality_df = pd.DataFrame(localities, columns=columns)


    def test_standardize_values(self):
        locality_df = copy.deepcopy(self.locality_df)
        locality_df.iloc[2]['country_name'] = 'Frances'

        standardized_df = self.locality_table_processor._standardize_values(locality_df)

        country_codes = pd.Series(['FR', 'FR', 'FR', 'US', 'US', 'US', 'US', 'US'])
        self.assertTrue(standardized_df['country_code'].equals(country_codes))

        country_names = pd.Series(['France', 'France', 'France', 'United States', 'United States',
                                   'United States', 'United States', 'United States'])
        self.assertTrue(standardized_df['country_name'].equals(country_names))


    def test_clean_data(self):
        locality_df = copy.deepcopy(self.locality_df)

        cleaned_df = self.locality_table_processor._clean_data(locality_df)
        self.assertEqual(len(cleaned_df), 6)


    def test_convert_dtypes(self):
        locality_df = copy.deepcopy(self.locality_df)

        converted_df = self.locality_table_processor._convert_dtypes(locality_df)
        for column_type in converted_df.dtypes:
            self.assertEqual(column_type, 'string')


    def test_process(self):
        locality_df = copy.deepcopy(self.locality_df)

        processed_table = self.locality_table_processor.process(locality_df)
        self.assertTrue(processed_table.is_ok)
        self.assertTrue(processed_table.initial_df.equals(locality_df))
        self.assertEqual(len(processed_table.final_df), 6)
        self.assertIsNone(processed_table.failure_cases)


class TestSchoolTableProcessor(unittest.TestCase):


    def setUp(self):
        self.school_table_processor = SchoolTableProcessor(None)

        schools = [
            (1,    'School 1', 'State',        'Rural',   100, 1.0, 6.0, 'FR-01-001', True),
            (2,    'School 2', 'Municipality', 'Urban',   200, 2.0, 5.0, 'FR-01-002', False),
            (3,    'School 3', 'Federal',      'Rural',   300, 3.0, 4.0, 'FR-01-001', True),
            (4,    'School 4', 'State',        'Urban',   400, 4.0, 3.0, 'FR-01-002', False),
            (4,    'School 4', 'State',        'Urban',   400, 4.0, 3.0, 'FR-01-002', False),
            (None, 'School 5', 'Municipality', 'Rural',   500, 5.0, 2.0, 'FR-01-001', True),
            (6,    'School 6', 'Municipality', 'Rural',  None, 6.0, 1.0, 'FR-01-004', None),
        ]

        columns = ['school_code', 'school_name', 'school_type', 'school_region',
                   'student_count', 'latitude', 'longitude', 'municipality_code',
                   'internet_availability']
        self.school_df = pd.DataFrame(schools, columns=columns)


    def test_clean_data(self):
        school_df = copy.deepcopy(self.school_df)

        cleaned_df = self.school_table_processor._clean_data(school_df)
        self.assertEqual(len(cleaned_df), 5)


    def test_convert_dtypes(self):
        schools = [
            (1,    'School 1', 'State',        'Rural',  100,  1.0, None, 'FR-01-001', True),
            (2,    'School 2', 'Municipality', 'Urban',  200, None,  5.0, 'FR-01-002', None),
            (3,    'School 3', 'Federal',      'Rural', None,  3.0,  4.0, 'FR-01-001', False),
        ]

        columns = ['school_code', 'school_name', 'school_type', 'school_region',
                   'student_count', 'latitude', 'longitude', 'municipality_code',
                   'internet_availability']
        school_df = pd.DataFrame(schools, columns=columns)

        converted_df = self.school_table_processor._convert_dtypes(school_df)
        column_dtype_dict = converted_df.dtypes.to_dict()
        self.assertEqual(column_dtype_dict['school_code'], 'string')
        self.assertEqual(column_dtype_dict['school_name'], 'string')
        self.assertEqual(column_dtype_dict['school_type'], 'string')
        self.assertEqual(column_dtype_dict['school_region'], 'string')
        self.assertEqual(column_dtype_dict['student_count'], 'Int32')
        self.assertEqual(column_dtype_dict['latitude'], 'Float32')
        self.assertEqual(column_dtype_dict['longitude'], 'Float32')
        self.assertEqual(column_dtype_dict['municipality_code'], 'string')
        self.assertEqual(column_dtype_dict['internet_availability'], 'boolean')


    def test_process(self):
        school_df = copy.deepcopy(self.school_df)

        processed_table = self.school_table_processor.process(school_df)
        self.assertTrue(processed_table.is_ok)
        self.assertTrue(processed_table.initial_df.equals(school_df))
        self.assertEqual(len(processed_table.final_df), 5)
        self.assertIsNone(processed_table.failure_cases)


class TestStudentCountEstimator(unittest.TestCase):


    def setUp(self):
        self.student_count_estimator = StudentCountEstimator()

        enhanced_schools = [
            ('US', 'US-CA', 'US-CA-01', 'State', 'Rural', 100),
            ('US', 'US-CA', 'US-CA-01', 'State', 'Rural', 200),
            ('US', 'US-CA', 'US-CA-01', 'State', 'Rural', None),
            ('US', 'US-CA', 'US-CA-02', 'State', 'Rural', None),
            ('US', 'US-CA', 'US-CA-03', 'State', 'Rural', 250),

            ('US', 'US-CA', 'US-CA-03', 'Federal', 'Rural', 250),
            ('US', 'US-CA', 'US-CA-03', 'Federal', 'Urban', None),
            ('US', 'US-CA', 'US-CA-04', 'Federal', 'Urban', None)
        ]

        columns = ['country_code', 'state_code', 'municipality_code',
                   'school_type', 'school_region', 'student_count']
        self.enhanced_school_df = pd.DataFrame(enhanced_schools, columns=columns)


    def test_estimate_student_count(self):
        enhanced_school_df = copy.deepcopy(self.enhanced_school_df)

        self.student_count_estimator.fit(enhanced_school_df)
        estimated_df = self.student_count_estimator.transform(enhanced_school_df)

        self.assertEqual(len(estimated_df), 8)

        student_count = [150, 150, 150, 200, 250, 250, 250, 225]
        self.assertListEqual(estimated_df['student_count'].values.tolist(), student_count)


if __name__ == '__main__':
    unittest.main()
