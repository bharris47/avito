class Vocabulary:
    def __init__(self, items):
        self._dictionary = {c: i for i, c in enumerate(sorted(items), 1)}

    def __getitem__(self, item):
        return self._dictionary.get(item)

    def __len__(self):
        return len(self._dictionary) + 1

    def __contains__(self, item):
        return item in self._dictionary
