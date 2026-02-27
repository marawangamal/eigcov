import copy

import torch
from torch.utils import data


class PytorchDataset(data.Dataset):
    """PyTorch Dataset that returns a dict of tensors for each datapoint."""

    def __init__(self, dataset, tokenizer, device):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, get_idx):
        datapoint = self.dataset[get_idx]
        input_dict = self.tokenizer(
            datapoint["input"], return_tensors="pt", truncation=True
        )
        input_ids = input_dict["input_ids"][0]
        input_mask = input_dict["attention_mask"][0]

        new_datapoint = copy.deepcopy(datapoint)
        new_datapoint.update({"input_ids": input_ids, "input_mask": input_mask})

        if "answer_choices" in datapoint:
            allChoices_ids = []
            allChoices_masks = []
            for choice in datapoint["answer_choices"]:
                choice_dict = self.tokenizer(
                    choice, return_tensors="pt", truncation=True
                )
                allChoices_ids.append(choice_dict["input_ids"][0])
                allChoices_masks.append(choice_dict["attention_mask"][0])

            new_datapoint.update(
                {
                    "all_choices_ids": allChoices_ids,
                    "all_choices_mask": allChoices_masks,
                }
            )
        else:
            assert "target" in datapoint
            target_dict = self.tokenizer(
                datapoint["target"], return_tensors="pt", truncation=True
            )
            new_datapoint.update(
                {
                    "target_ids": target_dict["input_ids"][0],
                    "target_mask": target_dict["attention_mask"][0],
                }
            )

        return new_datapoint

    def collate_fn(self, batch_ofDatapoints):
        """Collate a batch of datapoints into a batched dictionary of tensors."""
        datapoint_batched = {}

        for datapoint in batch_ofDatapoints:
            for k, v in datapoint.items():
                if k in datapoint_batched:
                    if "all_choices" in k:
                        datapoint_batched[k].extend(v)
                    else:
                        datapoint_batched[k].append(v)
                else:
                    if "all_choices" in k:
                        datapoint_batched[k] = list(v)
                    else:
                        datapoint_batched[k] = [v]

        for k, batch_ofValues in datapoint_batched.items():
            if ("ids" in k) or ("mask" in k):
                if "ids" in k:
                    padToken_id = self.tokenizer.pad_token_id
                    if padToken_id is None:
                        padToken_id = self.tokenizer.eos_token_id
                else:
                    padToken_id = 0

                datapoint_batched[k] = torch.nn.utils.rnn.pad_sequence(
                    batch_ofValues, batch_first=True, padding_value=padToken_id
                )
                if self.device is not None:
                    datapoint_batched[k] = datapoint_batched[k].to(self.device)

            elif k == "lbl":
                datapoint_batched[k] = torch.tensor(batch_ofValues)
                if self.device is not None:
                    datapoint_batched[k] = datapoint_batched[k].to(self.device)

        return datapoint_batched
