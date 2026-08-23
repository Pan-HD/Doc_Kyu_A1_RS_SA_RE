class EvaluationCache:
    """
    Debug-oriented cache.

    Key includes architecture + training seed + epochs so intentional
    re-training with a different seed is still possible later.
    """

    def __init__(self):
        self._cache = {}

    @staticmethod
    def make_key(architecture, training_seed, epochs):
        return architecture, int(training_seed), int(epochs)

    def get(self, architecture, training_seed, epochs):
        return self._cache.get(
            self.make_key(architecture, training_seed, epochs)
        )

    def put(self, result):
        key = self.make_key(
            result.architecture,
            result.training_seed,
            result.epochs,
        )
        self._cache[key] = result

    def __len__(self):
        return len(self._cache)
