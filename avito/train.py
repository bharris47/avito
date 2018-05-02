import csv
from argparse import ArgumentParser
from functools import partial

import joblib
from keras.callbacks import TensorBoard, ModelCheckpoint
from tqdm import tqdm

from avito.generator import BatchGenerator
from avito.loss import root_mean_squared_error
from avito.models import text_location_model
from avito.samples import trim_sample, transform_sample, train_val_split, build_text_vocabulary, \
    build_location_vocabulary

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('samples_path')
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()

    with open(args.samples_path) as f:
        samples = list(map(trim_sample, tqdm(csv.DictReader(f))))

    train_samples, val_samples = train_val_split(samples)

    text_vocabulary = build_text_vocabulary(train_samples)
    joblib.dump(text_vocabulary, 'text_vocabulary.pkl')
    location_vocabulary = build_location_vocabulary(train_samples)
    joblib.dump(location_vocabulary, 'location_vocabulary.pkl')

    transformer = partial(transform_sample, text_vocabulary=text_vocabulary, location_vocabulary=location_vocabulary)
    train_generator = BatchGenerator(train_samples, transformer, batch_size=args.batch_size, shuffle=True)
    val_generator = BatchGenerator(val_samples, transformer, batch_size=args.batch_size)

    model = text_location_model(len(text_vocabulary), len(location_vocabulary))
    model.compile('adam', loss=root_mean_squared_error)
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
