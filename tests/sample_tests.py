from csv import DictReader
from unittest import TestCase

from avito.train import trim_sample, get_location
from avito.samples import get_text, get_location, trim_sample
from tests.data_utils import get_data_path

SAMPLE_FILENAME = 'sample.csv'


class SampleTests(TestCase):
    def setUp(self):
        with open(get_data_path(SAMPLE_FILENAME)) as sample_file:
            reader = DictReader(sample_file)
            self.sample = list(reader)[0]

    def test_trim_sample(self):
        trimmed = trim_sample(self.sample)
        expected = {
            'title': 'Кокоби(кокон для сна)',
            'description': 'Кокон для сна малыша,пользовались меньше месяца.цвет серый',
            'city': 'Екатеринбург',
            'region': 'Свердловская область',
            'deal_probability': 0.12789
        }
        self.assertDictEqual(
            trimmed,
            expected
        )

    def test_get_text(self):
        text = get_text(self.sample)
        expected = 'Кокоби(кокон для сна) Кокон для сна малыша,пользовались меньше месяца.цвет серый'
        self.assertEqual(text, expected)

    def test_get_location(self):
        location = get_location(self.sample)
        expected = ('Свердловская область', 'Екатеринбург')
        self.assertTupleEqual(location, expected)
