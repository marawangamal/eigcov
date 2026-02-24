from evaluate import load
from statistics import mean


def get_average(list_ofNumbers):
    return round(mean(list_ofNumbers), 3)


def convert_dictOfLists_to_listOfDicts(dictOfLists):
    listOfDicts = []
    for datapoint_values in zip(*dictOfLists.values()):
        listOfDicts.append(dict(zip(dictOfLists, datapoint_values)))
    return listOfDicts


class Scorer(object):
    def __init__(self, metrics):
        self.metrics_toCompute = {"accuracy": False, "squad": False}

        if "Accuracy" in metrics:
            self.metrics_toCompute["accuracy"] = True
            self.accuracy_metric = load("accuracy")

        if "Squad" in metrics:
            self.metrics_toCompute["squad"] = True
            self.squad_metric = load("squad")

    def add_batch(self, batchOf_evalInfo):
        if self.metrics_toCompute["accuracy"]:
            self.accuracy_metric.add_batch(
                predictions=batchOf_evalInfo["predicted_choice"],
                references=batchOf_evalInfo["lbl"],
            )

        if self.metrics_toCompute["squad"]:
            self.squad_metric.add_batch(
                predictions=convert_dictOfLists_to_listOfDicts(
                    {
                        "id": batchOf_evalInfo["id"],
                        "prediction_text": batchOf_evalInfo["prediction_text"],
                    }
                ),
                references=convert_dictOfLists_to_listOfDicts(
                    {
                        "id": batchOf_evalInfo["id"],
                        "answers": batchOf_evalInfo["answers"],
                    }
                ),
            )

    def get_score(self):
        score = {}

        if self.metrics_toCompute["accuracy"]:
            score.update(self.accuracy_metric.compute())

        if self.metrics_toCompute["squad"]:
            squad_metrics = self.squad_metric.compute()
            for metric, value in squad_metrics.items():
                squad_metrics[metric] = value / 100
            score.update(squad_metrics)

        for key, value in score.items():
            score[key] = float("%.3f" % value)

        score["average"] = get_average(list(score.values()))

        return score
