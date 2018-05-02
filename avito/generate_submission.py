import csv
from argparse import ArgumentParser
from functools import partial
from time import time

import joblib
from tqdm import tqdm

from avito.generator import BatchGenerator
from avito.models import text_location_model
from avito.samples import trim_sample, transform_sample

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('samples_path')
    parser.add_argument('model_path')
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()

    with open(args.samples_path) as f:
        samples = list(map(trim_sample, tqdm(csv.DictReader(f))))

    text_vocabulary = joblib.load('text_vocabulary.pkl')
    location_vocabulary = joblib.load('location_vocabulary.pkl')

    transformer = partial(transform_sample, text_vocabulary=text_vocabulary, location_vocabulary=location_vocabulary)
    generator = BatchGenerator(samples, transformer, batch_size=args.batch_size, run_forever=False)
    batches = generator.generate_batches()

    model = text_location_model(len(text_vocabulary), len(location_vocabulary))
    model.load_weights(args.model_path)

    timestamp = str(int(time()))
    with open('submission_' + timestamp + '.csv', 'w') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['item_id', 'deal_probability'])

        i = 0
        for batch, _, _ in tqdm(batches):
            predictions = model.predict(batch, batch_size=args.batch_size)
            for deal_prob in predictions:
                sample_id = samples[i]['item_id']
                writer.writerow([sample_id, deal_prob[0]])
                i += 1
