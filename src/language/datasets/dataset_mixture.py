import random
import logging

from src.language.datasets.dataset_readers import get_datasetReader, DATASET_CLASSES

logger = logging.getLogger("root")

EIGHT_QA_DATASETS = [
    "cosmos_qa",
    "social_iqa",
    "paws",
    "quail",
    "wiki_qa",
    "quartz",
    "qasc",
    "ropes",
]

T0_HELD_OUT_DATASETS = [
    "rte",
    "cb",
    "winogrande",
    "wic",
    "wsc",
    "copa",
    "h-swag",
    "story_cloze",
    "anli-r1",
    "anli-r2",
    "anli-r3",
]

T5_MIXTURE = ["paws", "qasc", "quartz", "story_cloze", "wiki_qa", "winogrande", "wsc"]

DATASET_MIXTURES = {
    "eight_qa": EIGHT_QA_DATASETS,
    "T0_held_out": T0_HELD_OUT_DATASETS,
    "t5_mixture": T5_MIXTURE,
}


class DatasetMixtureReader(object):
    def __init__(self, mixture_ofDatasetReaders, maximumDatapoints_perDataset):
        self.mixture_ofDatasetReaders = mixture_ofDatasetReaders
        self.maximumDatapoints_perDataset = maximumDatapoints_perDataset

        self.cached_origData = {}
        self.cached_datasets = {}

    def get_dataset(self, split, template_idx, is_evaluation):
        if (split, template_idx) not in self.cached_datasets:
            mixture_dataset = []

            for _, dataset_reader in self.mixture_ofDatasetReaders.items():
                if (template_idx >= -1) or template_idx == -3:
                    dataset = dataset_reader.get_dataset(
                        split,
                        template_idx,
                        is_evaluation,
                        self.maximumDatapoints_perDataset,
                    )
                    mixture_dataset.extend(dataset)
                else:
                    assert template_idx == -2

                    num_templates = dataset_reader.get_numTemplates()
                    maximumDatapoints_perDatasetAndTemplate = (
                        self.maximumDatapoints_perDataset // num_templates
                    )

                    for iterated_templateIdx in range(num_templates):
                        dataset = dataset_reader.get_dataset(
                            split, iterated_templateIdx, is_evaluation
                        )
                        random.seed(0)
                        random.shuffle(dataset)
                        mixture_dataset.extend(
                            dataset[:maximumDatapoints_perDatasetAndTemplate]
                        )

            self.cached_datasets[(split, template_idx)] = mixture_dataset

        logger.info(
            f"\n\nNumber of datapoints in {split} for Mixture Dataset: "
            f"{len(self.cached_datasets[(split, template_idx)])}\n\n"
        )

        return self.cached_datasets[(split, template_idx)]

    def get_numTemplates(self):
        raise ValueError("Cannot get number of templates for mixture of datasets")

    def get_metricsForDataset(self):
        raise ValueError("Cannot get metrics for mixture of datasets")


def get_datasetMixture(dataset_mixture):
    if isinstance(dataset_mixture, list):
        for dataset in dataset_mixture:
            assert dataset in DATASET_CLASSES.keys()
        return dataset_mixture
    else:
        assert dataset_mixture in DATASET_MIXTURES.keys()
        return DATASET_MIXTURES[dataset_mixture]


def get_datasetMixtureReader(datast_mixture, maxDatapoints_perDataset, dataset_kwargs=None):
    mixture_ofDatasetReaders = {}
    for dataset in get_datasetMixture(datast_mixture):
        mixture_ofDatasetReaders[dataset] = get_datasetReader(dataset, dataset_kwargs)

    mixtureDataset_reader = DatasetMixtureReader(
        mixture_ofDatasetReaders, maxDatapoints_perDataset
    )

    return mixtureDataset_reader, mixture_ofDatasetReaders


def get_dataset_from_argparse(all_DatasetMixtures):
    datasets = []
    for dataset_or_mixture in all_DatasetMixtures:
        if dataset_or_mixture in DATASET_CLASSES.keys():
            datasets.append(dataset_or_mixture)
        elif dataset_or_mixture in DATASET_MIXTURES.keys():
            datasets.extend(DATASET_MIXTURES[dataset_or_mixture])
        else:
            raise ValueError(
                f"Invalid dataset or dataset mixture: {dataset_or_mixture}"
            )
    return datasets
