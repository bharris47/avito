from unittest import TestCase

from avito.vocabulary import Vocabulary


class VocabularyTests(TestCase):
    def setUp(self):
        self.manufacturers = ['samsung', 'lg', 'apple', 'huawei', 'htc']
        self.vocabulary = Vocabulary(self.manufacturers)

    def test_init(self):
        self.assertEqual(self.vocabulary['apple'], 1)
        self.assertEqual(self.vocabulary['htc'], 2)
        self.assertEqual(self.vocabulary['huawei'], 3)
        self.assertEqual(self.vocabulary['lg'], 4)
        self.assertEqual(self.vocabulary['samsung'], 5)
        self.assertIsNone(self.vocabulary['nokia'])

    def test_contains(self):
        self.assertTrue('apple' in self.vocabulary)
        self.assertFalse('nokia' in self.vocabulary)

    def test_len(self):
        self.assertEqual(len(self.vocabulary), 6)
