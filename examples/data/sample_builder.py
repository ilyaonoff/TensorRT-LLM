from typing import Dict

import torch
import transformers


class SampleBuilder:
    def __init__(self, tokenizer: transformers.PreTrainedTokenizerBase):
        self._tokenizer = tokenizer

    def _tokenize(self, prefix: str, text: str):
        text = text.replace('\n', '[NL]')
        input_ids = self._tokenizer(text, add_special_tokens=False)['input_ids']
        input_ids = [self._tokenizer.bos_token_id] + input_ids + [self._tokenizer.eos_token_id]
        return {
            prefix + '_input_ids': torch.tensor(input_ids, dtype=torch.int32)
        }

    def __call__(self, sample: Dict):
        prepared_sample = {}
        for key in ['query_pos', 'query_neg', 'doc_pos', 'doc_neg', 'document']:
            if key not in sample:
                continue
            prepared_sample.update(self._tokenize(key, sample[key]))
        return prepared_sample


class EvaluationSampleBuilder:
    def __init__(self, tokenizer: transformers.PreTrainedTokenizerBase):
        self._base = SampleBuilder(tokenizer)

    def __call__(self, sample: Dict):
        prepared = self._base(sample)
        prepared['label'] = sample['label']
        return prepared
