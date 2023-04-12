import unittest

from services.utils import (
    parse_int,
    parse_boolean,
    convert_to_list,
    fast_mode
)


class TestParseInt(unittest.TestCase):


    def test_parse_int(self):
        self.assertEqual(parse_int(1), 1)
        self.assertEqual(parse_int(1.0), 1)
        self.assertEqual(parse_int('1'), 1)
        self.assertEqual(parse_int('1.0'), '1.0')
        self.assertEqual(parse_int('1.0.0'), '1.0.0')
        self.assertEqual(parse_int(''), '')
        self.assertEqual(parse_int(None), None)


    def test_parse_boolean(self):
        self.assertEqual(parse_boolean(True), True)
        self.assertEqual(parse_boolean('True'), True)
        self.assertEqual(parse_boolean('true'), True)
        self.assertEqual(parse_boolean('TRUE'), True)
        self.assertEqual(parse_boolean('yes'), True)
        self.assertEqual(parse_boolean('YES'), True)
        self.assertEqual(parse_boolean('y'), True)
        self.assertEqual(parse_boolean('Y'), True)
        self.assertEqual(parse_boolean('1'), True)
        self.assertEqual(parse_boolean('1.0'), True)
        self.assertEqual(parse_boolean('2.0'), True)
        self.assertEqual(parse_boolean('-1.0'), True)

        self.assertEqual(parse_boolean(False), False)
        self.assertEqual(parse_boolean('False'), False)
        self.assertEqual(parse_boolean('false'), False)
        self.assertEqual(parse_boolean('FALSE'), False)
        self.assertEqual(parse_boolean('no'), False)
        self.assertEqual(parse_boolean('NO'), False)
        self.assertEqual(parse_boolean('n'), False)
        self.assertEqual(parse_boolean('N'), False)
        self.assertEqual(parse_boolean('0'), False)
        self.assertEqual(parse_boolean('0.0'), False)

        self.assertEqual(parse_boolean(''), '')
        self.assertEqual(parse_boolean(None), None)


    def test_convert_to_list(self):
        self.assertEqual(convert_to_list([1, 2, 3]), [1, 2, 3])
        self.assertEqual(convert_to_list('1,2,3'), [1, 2, 3])
        self.assertEqual(convert_to_list('[A, B, C]'), '[A, B, C]')
        self.assertEqual(convert_to_list('["A", "B", "C"]'), ['A', 'B', 'C'])

        self.assertEqual(convert_to_list('1,2,3', to_type=parse_int), [1, 2, 3])
        self.assertEqual(convert_to_list('1.0,2.0,3.0', to_type=parse_int), [1, 2, 3])

        self.assertEqual(convert_to_list([1.0, 0.0, '1'], to_type=parse_boolean), [True, False, True])
        self.assertEqual(convert_to_list('1,2,3', to_type=parse_boolean), [True, True, True])
        self.assertEqual(convert_to_list('True, False, True', to_type=parse_boolean), [True, False, True])
        self.assertEqual(convert_to_list('True, FALSE, truE', to_type=parse_boolean), 'True, FALSE, truE')
        self.assertEqual(convert_to_list('["True", "FALSE", "truE"]', to_type=parse_boolean), [True, False, True])


    def test_fast_mode(self):
        import pandas as pd
        df = pd.DataFrame({
            'a': [1,     1,   1,   1,   2,   2,   2,   2],
            'b': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'],
            'c': [1,     1,   1,   2,   2,   2,   2,   1]
        })

        result_dict = fast_mode(df, ['a'], 'c').set_index('a').to_dict()['c']
        self.assertEqual(result_dict, {1: 1, 2: 2})

        result_dict = fast_mode(df, ['b'], 'c').set_index('b').to_dict()['c']
        self.assertEqual(result_dict, {'A': 1, 'B': 1})

        result_dict = fast_mode(df, ['a', 'b'], 'c').set_index(['a', 'b']).to_dict()['c']
        self.assertEqual(result_dict, {(1, 'A'): 1, (1, 'B'): 1, (2, 'A'): 2, (2, 'B'): 1})
