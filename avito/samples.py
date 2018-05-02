from random import Random

from avito.vocabulary import Vocabulary


def get_text(sample):
    return ' '.join([sample['title'], sample['description']])


def get_location(sample):
    return sample['region'], sample['city']


def trim_sample(sample):
    return {
        'item_id': sample['item_id'],
        'title': sample['title'],
        'description': sample['description'],
        'city': sample['city'],
        'region': sample['region'],
        'deal_probability': float(sample.get('deal_probability', -1))
    }


def transform_sample(sample, text_vocabulary, location_vocabulary):
    text = [text_vocabulary[character] for character in get_text(sample) if character in text_vocabulary]
    location = location_vocabulary[get_location(sample)]
    return (text, location), sample['deal_probability']


def train_val_split(samples, split=0.2):
    r = Random(1)
    r.shuffle(samples)
    n_val = int(split * len(samples))
    return samples[:-n_val], samples[-n_val:]


def build_text_vocabulary(samples):
    texts = [get_text(sample) for sample in samples]
    vocabulary = set(c for text in texts for c in text)
    return Vocabulary(vocabulary)


def build_location_vocabulary(samples):
    locations = {get_location(sample) for sample in samples}
    return Vocabulary(locations)