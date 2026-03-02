import os
import logging
import copy
import random
import json

import datasets
from tqdm import tqdm
from promptsource.templates import DatasetTemplates, Template
from datasets import load_dataset

logger = logging.getLogger("root")


class DatasetReader(object):
    """Reads a dataset and holds all dataset-specific attributes."""

    def __init__(self, dataset_stash, template_stash):
        self.dataset_stash = dataset_stash
        self.template_stash = template_stash

        self.all_templates = self._get_datasetTemplates(None, None)

        self.cached_origData = {}
        self.cached_datasets = {}

    def _get_origData(self, split):
        if self.few_shot_random_seed is not None:
            return self._read_few_shot_dataset(split, self.few_shot_random_seed)
        else:
            return self._read_origin_dataset(split)

    def _read_origin_dataset(self, split):
        load_split = "validation" if split == "test" else split
        load_split = "validation" if load_split == "validation_full" else load_split

        if split not in self.cached_origData:
            logger.info(f"\t\tLoading Full Data for {self.name}")
            huggingFace_data = load_dataset(*self.dataset_stash, split=load_split)
            orig_data = []
            for idx, example in enumerate(huggingFace_data):
                example["idx"] = idx
                example["lbl"] = int(example["label"])
                orig_data.append(example)

            if split.lower() in ["validation", "test"]:
                if len(orig_data) > self.num_val_samples:
                    orig_val_data, orig_test_data = self._split_val_into_val_and_test(
                        orig_data, self.num_val_samples
                    )
                else:
                    print(
                        f"Validation/Test split is too small. {len(orig_data)} < {self.num_val_samples}"
                    )
                    print("splitting equally")
                    orig_val_data, orig_test_data = self._split_val_into_val_and_test(
                        orig_data, len(orig_data) // 2
                    )
                self.cached_origData["validation"] = orig_val_data
                self.cached_origData["test"] = orig_test_data
            else:
                self.cached_origData[split] = orig_data

        return self.cached_origData[split]

    def _read_few_shot_dataset(self, split, few_shot_random_seed):
        if split not in self.cached_origData:
            logger.info(
                f"\t\tLoading Few Shot Data for {self.name} with seed {few_shot_random_seed}"
            )
            file_path = os.path.join(
                "data", "few_shot", self.name, f"{few_shot_random_seed}_seed.jsonl"
            )
            if os.path.exists(file_path):
                with open(file_path, "r") as fin:
                    data = []
                    for idx, line in enumerate(fin.readlines()):
                        example = json.loads(line.strip("\n"))
                        example["lbl"] = int(example["label"])
                        data.append(example)
                    self.cached_origData[split] = data
            else:
                raise ValueError(f"Few shot dataset not found at {file_path}")
        return self.cached_origData[split]

    def _get_datasetTemplates(self, templateNames_toIgnore, metrics_toUse):
        all_templates = []
        for template in DatasetTemplates(*self.template_stash).templates.values():
            if template.metadata.original_task:
                should_ignoreTemplate = False
                for metric in template.metadata.metrics:
                    if metric not in metrics_toUse:
                        should_ignoreTemplate = True
                for template_name in templateNames_toIgnore:
                    if template.name == template_name:
                        should_ignoreTemplate = True
                if not should_ignoreTemplate:
                    all_templates.append(template)
        return all_templates

    def _applyTemplate_toData(
        self, orig_data, num_templates, template_idx, is_evaluation
    ):
        dataset = []
        for datapoint_idx, datapoint in enumerate(
            tqdm(orig_data, desc="Applying templates")
        ):
            if template_idx >= 0:
                templateIdx_forDatapoint = template_idx
            elif template_idx == -1:
                templateIdx_forDatapoint = datapoint_idx % num_templates
            elif template_idx == -3:
                templateIdx_forDatapoint = random.randint(0, len(self.all_templates))
            else:
                raise ValueError(f"Invalid template index {template_idx}")

            template = self.all_templates[templateIdx_forDatapoint]
            new_datapoint = copy.deepcopy(datapoint)

            if is_evaluation:
                answer_choices = template.get_answer_choices_list(datapoint)
                if answer_choices is not None:
                    new_datapoint["answer_choices"] = answer_choices

            input_txt, target_txt = template.apply(datapoint)
            new_datapoint["input"] = input_txt

            if not is_evaluation or "answer_choices" not in new_datapoint:
                new_datapoint["target"] = target_txt

            dataset.append(new_datapoint)
        return dataset

    def _split_val_into_val_and_test(self, orig_data, num_val_samples=32, seed=42):
        random.seed(seed)
        random.shuffle(orig_data)
        val_data = orig_data[:num_val_samples]
        test_data = orig_data[num_val_samples:]
        return val_data, test_data

    def get_dataset(
        self, split, template_idx, is_evaluation, max_samples_per_dataset=None
    ):
        if (split, template_idx) not in self.cached_datasets:
            orig_data = self._get_origData(split)
            total_examples = len(orig_data)
            orig_data = (
                orig_data[: self.max_datapoints_per_dataset_without_templates]
                if self.max_datapoints_per_dataset_without_templates
                and split.lower() == "train"
                else orig_data
            )
            logger.info(
                f"\tDataset:{self.name.upper()}\tSplit:{split}\t"
                f"Selected Examples: {len(orig_data)}\tNum Total: {total_examples}"
            )
            num_templates = self.get_numTemplates()

            if template_idx == -2:
                dataset = []
                for iterate_templateIdx in range(num_templates):
                    dataset.extend(
                        self._applyTemplate_toData(
                            orig_data, num_templates, iterate_templateIdx, is_evaluation
                        )
                    )
            else:
                dataset = self._applyTemplate_toData(
                    orig_data, num_templates, template_idx, is_evaluation
                )

            random.Random(4).shuffle(dataset)
            dataset = (
                dataset[:max_samples_per_dataset]
                if max_samples_per_dataset
                else dataset
            )
            self.cached_datasets[(split, template_idx)] = dataset

        return self.cached_datasets[(split, template_idx)]

    def get_numTemplates(self):
        return len(self.all_templates)

    def get_metricsForDataset(self):
        return self.all_templates[0].metadata.metrics


class RTEReader(DatasetReader):
    def __init__(self, dataset_kwargs=None):
        super().__init__(
            dataset_stash=("super_glue", "rte"), template_stash=("super_glue", "rte")
        )
        if dataset_kwargs:
            for k, v in dataset_kwargs.items():
                setattr(self, k, v)
        self.name = "rte"

    def _get_datasetTemplates(self, templateNames_toIgnore, metrics_toUse):
        return super()._get_datasetTemplates([], ["Accuracy"])


class HSwagReader(DatasetReader):
    def __init__(self, dataset_kwargs=None):
        super().__init__(dataset_stash=("hellaswag",), template_stash=("hellaswag",))
        if dataset_kwargs:
            for k, v in dataset_kwargs.items():
                setattr(self, k, v)
        self.name = "h-swag"

    def _get_datasetTemplates(self, templateNames_toIgnore, metrics_toUse):
        all_templates = super()._get_datasetTemplates(
            ["Randomized prompts template"], ["Accuracy"]
        )
        listOf_randomJinjas = [
            (
                "randomized prompt 1",
                "Can you pick the correct ending for the sentence: {{ctx}}|||{{answer_choices [label | int()]}}",
            ),
            (
                "randomized prompt 2",
                "The task is to generate the ending for the sentence: {{ctx}}|||{{answer_choices [label | int()]}}",
            ),
            (
                "randomized prompt 3",
                "How does this sentence end? {{ctx}}|||{{answer_choices [label | int()]}}",
            ),
            (
                "randomized prompt 4",
                "From the list of endings described below, what ending makes the most sense for the sentence {{ctx}}|||{{answer_choices [label | int()]}}",
            ),
        ]
        for name, jinja in listOf_randomJinjas:
            all_templates.append(
                Template(
                    name=name,
                    jinja=jinja,
                    reference="",
                    answer_choices='{{endings | join("|||")}}',
                )
            )
        return all_templates


class COPAReader(DatasetReader):
    def __init__(self, dataset_kwargs=None):
        super().__init__(
            dataset_stash=("super_glue", "copa"), template_stash=("super_glue", "copa")
        )
        if dataset_kwargs:
            for k, v in dataset_kwargs.items():
                setattr(self, k, v)
        self.name = "copa"

    def _get_datasetTemplates(self, templateNames_toIgnore, metrics_toUse):
        return super()._get_datasetTemplates(
            [
                "…which may be caused by",
                "…What could happen next, C1 or C2?",
                "…As a result, C1 or C2?",
                "…why? C1 or C2",
            ],
            ["Accuracy"],
        )


class WiCReader(DatasetReader):
    def __init__(self, dataset_kwargs=None):
        super().__init__(
            dataset_stash=("super_glue", "wic"), template_stash=("super_glue", "wic")
        )
        if dataset_kwargs:
            for k, v in dataset_kwargs.items():
                setattr(self, k, v)
        self.name = "wic"

    def _get_datasetTemplates(self, templateNames_toIgnore, metrics_toUse):
        return super()._get_datasetTemplates([], ["Accuracy"])


class WinograndeReader(DatasetReader):
    def __init__(self, dataset_kwargs=None):
        super().__init__(
            dataset_stash=("winogrande", "winogrande_xl"),
            template_stash=("winogrande", "winogrande_xl"),
        )
        if dataset_kwargs:
            for k, v in dataset_kwargs.items():
                setattr(self, k, v)
        self.name = "winogrande"

    def _read_origin_dataset(self, split):
        load_split = "validation" if split == "test" else split
        load_split = "validation" if load_split == "validation_full" else load_split

        if split not in self.cached_origData:
            huggingFace_data = load_dataset(*self.dataset_stash, split=load_split)
            orig_data = []
            for idx, example in enumerate(huggingFace_data):
                example["idx"] = idx
                example["lbl"] = int(example["answer"]) - 1
                orig_data.append(example)

            if split.lower() in ["validation", "test"]:
                assert len(orig_data) > self.num_val_samples
                orig_val_data, orig_test_data = self._split_val_into_val_and_test(
                    orig_data, self.num_val_samples
                )
                self.cached_origData["validation"] = orig_val_data
                self.cached_origData["test"] = orig_test_data
            else:
                self.cached_origData[split] = orig_data

        return self.cached_origData[split]

    def _get_datasetTemplates(self, templateNames_toIgnore, metrics_toUse):
        return super()._get_datasetTemplates([], ["Accuracy"])


class CBReader(DatasetReader):
    def __init__(self, dataset_kwargs=None):
        super().__init__(
            dataset_stash=("super_glue", "cb"), template_stash=("super_glue", "cb")
        )
        if dataset_kwargs:
            for k, v in dataset_kwargs.items():
                setattr(self, k, v)
        self.name = "cb"

    def _get_datasetTemplates(self, templateNames_toIgnore, metrics_toUse):
        return super()._get_datasetTemplates([], ["Accuracy"])


class StoryClozeReader(DatasetReader):
    def __init__(self, dataset_kwargs=None):
        super().__init__(
            dataset_stash=("story_cloze", "2016"),
            template_stash=("story_cloze", "2016"),
        )
        if dataset_kwargs:
            for k, v in dataset_kwargs.items():
                setattr(self, k, v)
        self.name = "story_cloze"

    def _read_origin_dataset(self, split):
        if split == "train":
            load_split = "validation"
        elif split in ["validation", "validation_full", "test"]:
            load_split = "test"

        if split not in self.cached_origData:
            huggingFace_data = load_dataset(
                "csv",
                data_files=f"datasets_language/cloze_{load_split}_{self.dataset_stash[1]}.csv",
                trust_remote_code=True,
                split="train",
            )
            orig_data = []
            for idx, example in enumerate(huggingFace_data):
                example["idx"] = idx
                example["lbl"] = int(example["answer_right_ending"]) - 1
                orig_data.append(example)

            if split.lower() in ["validation", "test"]:
                assert (
                    len(orig_data) > self.num_val_samples
                ), f"Validation/Test split is too small. {len(orig_data)} < {self.num_val_samples}"
                orig_val_data, orig_test_data = self._split_val_into_val_and_test(
                    orig_data, self.num_val_samples
                )
                self.cached_origData["validation"] = orig_val_data
                self.cached_origData["test"] = orig_test_data
            else:
                self.cached_origData[split] = orig_data

        return self.cached_origData[split]

    def _get_datasetTemplates(self, templateNames_toIgnore, metrics_toUse):
        return super()._get_datasetTemplates([], ["Accuracy"])


class ANLIR1Reader(DatasetReader):
    def __init__(self, dataset_kwargs=None):
        super().__init__(dataset_stash=("anli",), template_stash=("anli",))
        if dataset_kwargs:
            for k, v in dataset_kwargs.items():
                setattr(self, k, v)
        self.name = "anli-r1"

    def _read_origin_dataset(self, split):
        load_split = "dev" if "validation" in split.lower() else split
        if split not in self.cached_origData:
            huggingFace_data = load_dataset(
                *self.dataset_stash, split=f"{load_split}_r1"
            )
            orig_data = []
            for idx, example in enumerate(huggingFace_data):
                example["idx"] = idx
                example["lbl"] = int(example["label"])
                orig_data.append(example)
            if split == "validation":
                random.seed(42)
                random.shuffle(orig_data)
                orig_data = orig_data[: self.num_val_samples]
            self.cached_origData[split] = orig_data
        return self.cached_origData[split]

    def _get_datasetTemplates(self, templateNames_toIgnore, metrics_toUse):
        return super()._get_datasetTemplates([], ["Accuracy"])


class ANLIR2Reader(DatasetReader):
    def __init__(self, dataset_kwargs=None):
        super().__init__(dataset_stash=("anli",), template_stash=("anli",))
        if dataset_kwargs:
            for k, v in dataset_kwargs.items():
                setattr(self, k, v)
        self.name = "anli-r2"

    def _read_origin_dataset(self, split):
        load_split = "dev" if "validation" in split.lower() else split
        if split not in self.cached_origData:
            huggingFace_data = load_dataset(
                *self.dataset_stash, split=f"{load_split}_r2"
            )
            orig_data = []
            for idx, example in enumerate(huggingFace_data):
                example["idx"] = idx
                example["lbl"] = int(example["label"])
                orig_data.append(example)
            if split == "validation":
                random.seed(42)
                random.shuffle(orig_data)
                orig_data = orig_data[: self.num_val_samples]
            self.cached_origData[split] = orig_data
        return self.cached_origData[split]

    def _get_datasetTemplates(self, templateNames_toIgnore, metrics_toUse):
        return super()._get_datasetTemplates([], ["Accuracy"])


class ANLIR3Reader(DatasetReader):
    def __init__(self, dataset_kwargs=None):
        super().__init__(dataset_stash=("anli",), template_stash=("anli",))
        if dataset_kwargs:
            for k, v in dataset_kwargs.items():
                setattr(self, k, v)
        self.name = "anli-r3"

    def _read_origin_dataset(self, split):
        load_split = "dev" if "validation" in split.lower() else split
        if split not in self.cached_origData:
            huggingFace_data = load_dataset(
                *self.dataset_stash, split=f"{load_split}_r3"
            )
            orig_data = []
            for idx, example in enumerate(huggingFace_data):
                example["idx"] = idx
                example["lbl"] = int(example["label"])
                orig_data.append(example)
            if split == "validation":
                random.seed(42)
                random.shuffle(orig_data)
                orig_data = orig_data[: self.num_val_samples]
            self.cached_origData[split] = orig_data
        return self.cached_origData[split]

    def _get_datasetTemplates(self, templateNames_toIgnore, metrics_toUse):
        return super()._get_datasetTemplates([], ["Accuracy"])


class WSCReader(DatasetReader):
    def __init__(self, dataset_kwargs=None):
        super().__init__(
            dataset_stash=("super_glue", "wsc.fixed"),
            template_stash=("super_glue", "wsc.fixed"),
        )
        if dataset_kwargs:
            for k, v in dataset_kwargs.items():
                setattr(self, k, v)
        self.name = "wsc"

    def _get_datasetTemplates(self, templateNames_toIgnore, metrics_toUse):
        return super()._get_datasetTemplates([], ["Accuracy"])


class CosmosQAReader(DatasetReader):
    def __init__(self, dataset_kwargs=None):
        super().__init__(dataset_stash=("cosmos_qa",), template_stash=("cosmos_qa",))
        if dataset_kwargs:
            for k, v in dataset_kwargs.items():
                setattr(self, k, v)
        self.name = "cosmos_qa"

    def _read_origin_dataset(self, split):
        load_split = "validation" if split == "test" else split
        load_split = "validation" if load_split == "validation_full" else load_split
        if split not in self.cached_origData:
            huggingFace_data = load_dataset(*self.dataset_stash, split=load_split)
            orig_data = []
            for idx, example in enumerate(huggingFace_data):
                example["idx"] = idx
                example["lbl"] = int(example["label"])
                orig_data.append(example)
            if split.lower() in ["validation", "test"]:
                assert len(orig_data) > self.num_val_samples
                orig_val_data, orig_test_data = self._split_val_into_val_and_test(
                    orig_data, self.num_val_samples
                )
                self.cached_origData["validation"] = orig_val_data
                self.cached_origData["test"] = orig_test_data
            else:
                self.cached_origData[split] = orig_data
        return self.cached_origData[split]

    def _get_datasetTemplates(self, templateNames_toIgnore, metrics_toUse):
        return super()._get_datasetTemplates([], ["Accuracy"])


class SocialIQAReader(DatasetReader):
    def __init__(self, dataset_kwargs=None):
        super().__init__(
            dataset_stash=("social_i_qa",), template_stash=("social_i_qa",)
        )
        if dataset_kwargs:
            for k, v in dataset_kwargs.items():
                setattr(self, k, v)
        self.name = "social_i_qa"

    def _read_origin_dataset(self, split):
        load_split = "validation" if split == "test" else split
        load_split = "validation" if load_split == "validation_full" else load_split
        if split not in self.cached_origData:
            huggingFace_data = load_dataset(*self.dataset_stash, split=load_split)
            orig_data = []
            for idx, example in enumerate(huggingFace_data):
                example["idx"] = idx
                example["lbl"] = int(example["label"]) - 1
                orig_data.append(example)
            if split.lower() in ["validation", "test"]:
                assert len(orig_data) > self.num_val_samples
                orig_val_data, orig_test_data = self._split_val_into_val_and_test(
                    orig_data, self.num_val_samples
                )
                self.cached_origData["validation"] = orig_val_data
                self.cached_origData["test"] = orig_test_data
            else:
                self.cached_origData[split] = orig_data
        return self.cached_origData[split]

    def _get_datasetTemplates(self, templateNames_toIgnore, metrics_toUse):
        return super()._get_datasetTemplates(
            ["Check if a random answer is valid or not"], ["Accuracy"]
        )


class PAWSReader(DatasetReader):
    def __init__(self, dataset_kwargs=None):
        super().__init__(
            dataset_stash=("paws", "labeled_final"),
            template_stash=("paws", "labeled_final"),
        )
        if dataset_kwargs:
            for k, v in dataset_kwargs.items():
                setattr(self, k, v)
        self.name = "paws"

    def _read_origin_dataset(self, split):
        if split not in self.cached_origData:
            huggingFace_data = load_dataset(*self.dataset_stash, split=split)
            orig_data = []
            for idx, example in enumerate(huggingFace_data):
                example["idx"] = idx
                example["lbl"] = example["label"]
                orig_data.append(example)
            self.cached_origData[split] = orig_data
        return self.cached_origData[split]

    def _get_datasetTemplates(self, templateNames_toIgnore, metrics_toUse):
        return super()._get_datasetTemplates([], ["Accuracy"])


class QuAILReader(DatasetReader):
    def __init__(self, dataset_kwargs=None):
        super().__init__(dataset_stash=("quail",), template_stash=("quail",))
        if dataset_kwargs:
            for k, v in dataset_kwargs.items():
                setattr(self, k, v)
        self.name = "quail"

    def _read_origin_dataset(self, split):
        load_split = "challenge" if split == "test" else split
        if split not in self.cached_origData:
            huggingFace_data = load_dataset(*self.dataset_stash, split=load_split)
            orig_data = []
            for idx, example in enumerate(huggingFace_data):
                example["idx"] = idx
                example["lbl"] = example["correct_answer_id"]
                orig_data.append(example)
            self.cached_origData[split] = orig_data
        return self.cached_origData[split]

    def _get_datasetTemplates(self, templateNames_toIgnore, metrics_toUse):
        return super()._get_datasetTemplates([], ["Accuracy"])


class WikiQAReader(DatasetReader):
    def __init__(self, dataset_kwargs=None):
        super().__init__(dataset_stash=("wiki_qa",), template_stash=("wiki_qa",))
        if dataset_kwargs:
            for k, v in dataset_kwargs.items():
                setattr(self, k, v)
        self.name = "wiki_qa"

    def _read_origin_dataset(self, split):
        if split not in self.cached_origData:
            huggingFace_data = load_dataset(*self.dataset_stash, split=split)
            orig_data = []
            for idx, example in enumerate(huggingFace_data):
                example["idx"] = idx
                example["lbl"] = int(example["label"])
                orig_data.append(example)
            self.cached_origData[split] = orig_data
        return self.cached_origData[split]

    def _get_datasetTemplates(self, templateNames_toIgnore, metrics_toUse):
        return super()._get_datasetTemplates([], ["Accuracy"])


class QuaRTzReader(DatasetReader):
    def __init__(self, dataset_kwargs=None):
        super().__init__(dataset_stash=("quartz",), template_stash=("quartz",))
        if dataset_kwargs:
            for k, v in dataset_kwargs.items():
                setattr(self, k, v)
        self.name = "quartz"
        self.string_toLabelIdx = {"A": 0, "B": 1}

    def _read_origin_dataset(self, split):
        if split not in self.cached_origData:
            huggingFace_data = load_dataset(*self.dataset_stash, split=split)
            orig_data = []
            for idx, example in enumerate(huggingFace_data):
                example["idx"] = idx
                example["lbl"] = self.string_toLabelIdx[example["answerKey"]]
                orig_data.append(example)
            self.cached_origData[split] = orig_data
        return self.cached_origData[split]

    def _get_datasetTemplates(self, templateNames_toIgnore, metrics_toUse):
        return super()._get_datasetTemplates([], ["Accuracy"])


class QASCReader(DatasetReader):
    def __init__(self, dataset_kwargs=None):
        super().__init__(dataset_stash=("qasc",), template_stash=("qasc",))
        if dataset_kwargs:
            for k, v in dataset_kwargs.items():
                setattr(self, k, v)
        self.name = "qasc"
        self.string_toLabelIdx = {
            "A": 0,
            "B": 1,
            "C": 2,
            "D": 3,
            "E": 4,
            "F": 5,
            "G": 6,
            "H": 7,
        }

    def _read_origin_dataset(self, split):
        load_split = "validation" if split == "test" else split
        load_split = "validation" if load_split == "validation_full" else load_split
        if split not in self.cached_origData:
            huggingFace_data = load_dataset(*self.dataset_stash, split=load_split)
            orig_data = []
            for idx, example in enumerate(huggingFace_data):
                example["idx"] = idx
                example["lbl"] = self.string_toLabelIdx[example["answerKey"]]
                orig_data.append(example)
            if split.lower() in ["validation", "test"]:
                assert len(orig_data) > self.num_val_samples
                orig_val_data, orig_test_data = self._split_val_into_val_and_test(
                    orig_data, self.num_val_samples
                )
                self.cached_origData["validation"] = orig_val_data
                self.cached_origData["test"] = orig_test_data
            else:
                self.cached_origData[split] = orig_data
        return self.cached_origData[split]

    def _get_datasetTemplates(self, templateNames_toIgnore, metrics_toUse):
        return super()._get_datasetTemplates([], ["Accuracy"])


class ROPESReader(DatasetReader):
    def __init__(self, dataset_kwargs=None):
        super().__init__(dataset_stash=("ropes",), template_stash=("ropes",))
        if dataset_kwargs:
            for k, v in dataset_kwargs.items():
                setattr(self, k, v)
        self.name = "ropes"

    def _read_origin_dataset(self, split):
        load_split = "validation" if split == "test" else split
        load_split = "validation" if load_split == "validation_full" else load_split
        if split not in self.cached_origData:
            huggingFace_data = load_dataset(*self.dataset_stash, split=load_split)
            orig_data = []
            for idx, example in enumerate(huggingFace_data):
                example["idx"] = idx
                example["answers"]["answer_start"] = [0]
                orig_data.append(example)
            if split.lower() in ["validation", "test"]:
                assert len(orig_data) > self.num_val_samples
                orig_val_data, orig_test_data = self._split_val_into_val_and_test(
                    orig_data, self.num_val_samples
                )
                self.cached_origData["validation"] = orig_val_data
                self.cached_origData["test"] = orig_test_data
            else:
                self.cached_origData[split] = orig_data
        return self.cached_origData[split]

    def _get_datasetTemplates(self, templateNames_toIgnore, metrics_toUse):
        return super()._get_datasetTemplates([], ["Squad"])


DATASET_CLASSES = {
    "rte": RTEReader,
    "h-swag": HSwagReader,
    "copa": COPAReader,
    "wic": WiCReader,
    "winogrande": WinograndeReader,
    "cb": CBReader,
    "story_cloze": StoryClozeReader,
    "anli-r1": ANLIR1Reader,
    "anli-r2": ANLIR2Reader,
    "anli-r3": ANLIR3Reader,
    "wsc": WSCReader,
    "cosmos_qa": CosmosQAReader,
    "social_iqa": SocialIQAReader,
    "paws": PAWSReader,
    "quail": QuAILReader,
    "wiki_qa": WikiQAReader,
    "quartz": QuaRTzReader,
    "qasc": QASCReader,
    "ropes": ROPESReader,
}


def get_datasetReader(dataset_name, dataset_kwargs=None):
    return DATASET_CLASSES[dataset_name](dataset_kwargs)
