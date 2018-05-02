from random import Random

import numpy as np

ZERO_PROB_WEIGHT = 0.5

EPSILON = 1e-3


class BatchGenerator:
    def __init__(self, samples, transformer, batch_size=128, shuffle=False, run_forever=True):
        self._samples = samples
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._run_forever = run_forever
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

            mask = outputs == 0
            sample_weights = np.ones_like(outputs)
            sample_weights[mask] = ZERO_PROB_WEIGHT
            yield [text_batch, locations], outputs, sample_weights

    def inputs(self):
        mask = list(range(len(self._samples)))
        r = Random(1)
        while True:
            if self._shuffle:
                r.shuffle(mask)
            for i in mask:
                inputs, output = self._transformer(self._samples[i])
                yield inputs, output
            if not self._run_forever:
                break

    def __len__(self):
        return int(len(self._samples) / self._batch_size)


def chunks(items, chunk_size):
    chunk = []
    for item in items:
        chunk.append(item)
        if len(chunk) == chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk
