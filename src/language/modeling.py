import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def greedy_generation(
    transformer,
    input_ids,
    input_mask,
    bos_tokenId,
    eos_tokenId,
    pad_tokenId,
    max_generationLength,
):
    """Greedy generation for encoder-decoder models, caching encoder outputs."""
    past_key_values = None
    batch_size = input_ids.shape[0]

    current_decoderInputIds = torch.tensor([bos_tokenId] * batch_size)[:, None].to(
        input_ids.device
    )
    current_decoderMask = torch.ones((batch_size, 1)).to(input_ids.device)

    encoder_outputs = transformer.get_encoder()(input_ids, input_mask)

    generated_ids = current_decoderInputIds

    hasSequence_hitEOS = torch.zeros(size=(batch_size, 1), dtype=torch.int).to(
        input_ids.device
    )

    for i in range(max_generationLength):
        output = transformer(
            attention_mask=input_mask,
            decoder_input_ids=current_decoderInputIds,
            decoder_attention_mask=current_decoderMask,
            encoder_outputs=encoder_outputs,
            use_cache=True,
            past_key_values=past_key_values,
        )

        past_key_values = output.past_key_values

        predicted_nextToken = torch.argmax(output.logits, -1)

        predicted_nextToken = (
            1 - hasSequence_hitEOS
        ) * predicted_nextToken + hasSequence_hitEOS * pad_tokenId

        generated_ids = torch.cat((generated_ids, predicted_nextToken), dim=1)

        isToken_EOSToken = predicted_nextToken == eos_tokenId
        hasSequence_hitEOS = torch.bitwise_or(hasSequence_hitEOS, isToken_EOSToken)

        if torch.sum(hasSequence_hitEOS) == batch_size:
            break

        current_decoderInputIds = predicted_nextToken

    return generated_ids


class T5Wrapper(nn.Module):
    """Wraps a HuggingFace T5 model with task-arithmetic-compatible interface."""

    def __init__(self, transformer, tokenizer):
        super().__init__()
        self.transformer = transformer
        self.tokenizer = tokenizer

    def forward(self, batch):
        transformer_outputs = self.transformer(
            input_ids=batch["input_ids"],
            attention_mask=batch["input_mask"],
            labels=batch["target_ids"],
        )

        target_logits = transformer_outputs[1].float()
        vocab_size = target_logits.shape[-1]

        logProbs_ofTargetIds = F.cross_entropy(
            target_logits.reshape(-1, vocab_size),
            batch["target_ids"].reshape(-1),
            reduction="none",
        )
        target_mask = batch["target_mask"].reshape(-1)
        logProbs_ofTargetIds_zeroOutPadIds = logProbs_ofTargetIds * target_mask

        loss = torch.sum(logProbs_ofTargetIds_zeroOutPadIds) / torch.sum(target_mask)

        return loss, {"loss": loss.detach().cpu().item(), "target_logits": target_logits}

    def _broadcast_tensors(self, input_masks, encoder_outputs, num_choices):
        input_masks = torch.repeat_interleave(input_masks, num_choices, dim=0)
        encoder_outputs = (
            torch.repeat_interleave(encoder_outputs[0], num_choices, dim=0),
        )
        return input_masks, encoder_outputs

    def compute_logProb(
        self,
        logProbs_ofAllChoices_ids,
        allChoices_masks,
        num_choices,
        maxChoice_len,
        length_normalization,
    ):
        logProbs_ofAllChoices_ids = logProbs_ofAllChoices_ids.reshape(
            -1, num_choices, maxChoice_len
        )
        allChoices_masks = allChoices_masks.reshape(-1, num_choices, maxChoice_len)
        logProbs_ofAllChoicesIds_zeroOutPadIds = (
            logProbs_ofAllChoices_ids * allChoices_masks
        )

        logProbs_ofAllChoices = torch.sum(logProbs_ofAllChoicesIds_zeroOutPadIds, dim=2)
        len_allChoices = torch.sum(allChoices_masks, dim=2)

        if length_normalization:
            logProbs_ofAllChoices = logProbs_ofAllChoices / len_allChoices

        return (
            logProbs_ofAllChoices,
            logProbs_ofAllChoicesIds_zeroOutPadIds,
            len_allChoices,
        )

    def compute_logProb_ofAllChoices(
        self,
        input_ids,
        input_masks,
        allChoices_ids,
        allChoices_masks,
        length_normalization,
    ):
        encoder_outputs = self.transformer.get_encoder()(input_ids, input_masks)

        assert allChoices_ids.shape[0] % input_masks.shape[0] == 0
        num_choices = allChoices_ids.shape[0] // input_masks.shape[0]

        input_masks, encoder_outputs = self._broadcast_tensors(
            input_masks, encoder_outputs, num_choices
        )

        transformer_outputs = self.transformer(
            attention_mask=input_masks,
            encoder_outputs=encoder_outputs,
            labels=allChoices_ids,
        )

        logits_ofAllChoices = transformer_outputs[1].float()
        maxChoice_len = logits_ofAllChoices.shape[1]
        vocab_size = logits_ofAllChoices.shape[-1]

        logProbs_ofAllChoices_ids = -F.cross_entropy(
            logits_ofAllChoices.view(-1, vocab_size),
            allChoices_ids.view(-1),
            reduction="none",
        )

        return self.compute_logProb(
            logProbs_ofAllChoices_ids,
            allChoices_masks,
            num_choices,
            maxChoice_len,
            length_normalization,
        )

    def predict_mulChoice(self, batch, length_normalization):
        (
            score_ofChoices,
            logProbs_ofAllChoicesIds,
            len_allChoices,
        ) = self.compute_logProb_ofAllChoices(
            batch["input_ids"],
            batch["input_mask"],
            batch["all_choices_ids"],
            batch["all_choices_mask"],
            length_normalization,
        )

        _, predicted_choice = torch.max(score_ofChoices, dim=1)

        return (
            predicted_choice.cpu().numpy().tolist(),
            score_ofChoices.cpu().numpy().tolist(),
            logProbs_ofAllChoicesIds.cpu().numpy().tolist(),
            len_allChoices.cpu().numpy().tolist(),
        )

    def generate(self, batch, max_generationLength):
        generated_ids = greedy_generation(
            self.transformer,
            batch["input_ids"],
            batch["input_mask"],
            self.tokenizer.pad_token_id,
            self.tokenizer.eos_token_id,
            self.tokenizer.pad_token_id,
            max_generationLength,
        )

        generated_ids = generated_ids.cpu().numpy().tolist()
        generated_txt = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )

        return generated_ids, generated_txt

    def save(self, filename):
        from src.utils import torch_save
        print(f"Saving model to {filename}")
        torch_save(self, filename)

    @classmethod
    def load(cls, filename, device=None):
        from src.utils import torch_load
        model = torch_load(filename, device=device)
        return model
