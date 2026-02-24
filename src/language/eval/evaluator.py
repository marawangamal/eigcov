from src.language.eval.scorer import Scorer


class Evaluator(object):
    """Accumulates batches and deduplicates by idx before computing metrics."""

    def __init__(self, metrics, prediction_fp=None):
        self.scorer = Scorer(metrics)
        self.prediction_fp = prediction_fp
        self.seen_idxs = {}

    def add_batch(self, batchOf_evalInfo):
        batchOf_idxs = batchOf_evalInfo["idx"]

        idx_toRemove = []
        for batch_idx, idx in enumerate(batchOf_idxs):
            if idx in self.seen_idxs:
                idx_toRemove.append(batch_idx)
            self.seen_idxs[idx] = True

        filteredBatch_ofEvalInfo = {}
        for key, batchOf_values in batchOf_evalInfo.items():
            filtered_value = [
                value
                for batch_idx, value in enumerate(batchOf_values)
                if batch_idx not in idx_toRemove
            ]
            filteredBatch_ofEvalInfo[key] = filtered_value

        self.scorer.add_batch(filteredBatch_ofEvalInfo)

    def get_result(self):
        return self.scorer.get_score()
