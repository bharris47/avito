import csv
from argparse import ArgumentParser
from functools import partial
from random import Random

import numpy as np
from keras import Input, Model
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.layers import Embedding, CuDNNLSTM, Concatenate, Dense, Reshape
from tqdm import tqdm


class Vocabulary:
    def __init__(self, items):
        self._dictionary = {c: i for i, c in enumerate(sorted(items), 1)}

    def __getitem__(self, item):
        return self._dictionary.get(item)

    def __len__(self):
        return len(self._dictionary) + 1

    def __contains__(self, item):
        return item in self._dictionary


def chunks(items, chunk_size):
    chunk = []
    for item in items:
        chunk.append(item)
        if len(chunk) == chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


class BatchGenerator:
    def __init__(self, samples, transformer, batch_size=128, shuffle=False):
        self._samples = samples
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._transformer = transformer

    def generate_batches(self):
        batches = chunks(self.inputs(), self._batch_size)
        for batch in batches:
            inputs, outputs = zip(*batch)
            texts, locations = zip(*inputs)
            locations = np.array(locations)
            outputs = np.array(outputs)
            max_text_length = max(len(text) for text in texts)
            text_batch = np.zeros((len(texts), max_text_length))
            for i, text in enumerate(texts):
                text_batch[i, :len(text)] = text
            yield [text_batch, locations], outputs

    def inputs(self):
        mask = list(range(len(self._samples)))
        r = Random(1)
        while True:
            if self._shuffle:
                r.shuffle(mask)
            for i in mask:
                inputs, output = self._transformer(self._samples[i])
                yield inputs, output

    def __len__(self):
        return int(len(self._samples) / self._batch_size)


def text_location_model(text_vocab_size, location_vocab_size, text_embedding_size=8, text_hidden_size=300,
                        location_embedding_size=8):
    text = Input(shape=(None,))
    location = Input(shape=(1,))

    text_embedding = Embedding(text_vocab_size, text_embedding_size)(text)
    text_hidden = CuDNNLSTM(text_hidden_size)(text_embedding)

    location_embedding = Embedding(location_vocab_size, location_embedding_size)(location)
    location_embedding = Reshape((location_embedding_size,))(location_embedding)

    features = Concatenate()([text_hidden, location_embedding])
    prediction = Dense(1, activation='sigmoid')(features)

    model = Model(inputs=[text, location], outputs=prediction)
    return model


def get_text(sample):
    return ' '.join([sample['title'], sample['description']])


def get_location(sample):
    return sample['region'], sample['city']


def trim_sample(sample):
    return {
        'title': sample['title'],
        'description': sample['description'],
        'city': sample['city'],
        'region': sample['region'],
        'deal_probability': float(sample['deal_probability'])
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


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('samples_path')
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()

    with open(args.samples_path) as f:
        samples = list(map(trim_sample, tqdm(csv.DictReader(f))))

    train_samples, val_samples = train_val_split(samples)

    text_vocabulary = build_text_vocabulary(train_samples)
    location_vocabulary = build_location_vocabulary(train_samples)

    transformer = partial(transform_sample, text_vocabulary=text_vocabulary, location_vocabulary=location_vocabulary)
    train_generator = BatchGenerator(train_samples, transformer, batch_size=args.batch_size, shuffle=True)
    val_generator = BatchGenerator(val_samples, transformer, batch_size=args.batch_size)

    model = text_location_model(len(text_vocabulary), len(location_vocabulary))
    model.compile('adam', loss='mse', metrics=['acc'])
    model.summary()

    model.fit_generator(
        train_generator.generate_batches(),
        len(train_generator),
        epochs=100,
        validation_data=val_generator.generate_batches(),
        validation_steps=len(val_generator),
        callbacks=[
            TensorBoard(),
            ModelCheckpoint('text_location.{epoch:02d}-{val_loss:.6f}.hdf5')
        ]
    )
