import unittest

from mock import mock

class TestMock(unittest.TestCase):

    def test_mock(self):
        self.assertEqual(mock.mock_func(2), 4)

if __name__ == '__main__':
    unittest.main()