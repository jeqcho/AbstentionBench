"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import datasets
from recipe.abstention_datasets.abstract_abstention_dataset import AbstentionDataset
from recipe.abstention_datasets.abstract_abstention_dataset import Prompt


class Squad2Dataset(AbstentionDataset):

    _PREPROMPT = "Respond to the question using only information given in the context."
    _TEMPLATE = "{preprompt}\nContext: {context}\nQuestion: {question}"

    def __init__(self, max_num_samples=None):
        super().__init__()

        self.dataset = datasets.load_dataset(
            "rajpurkar/squad_v2",
        )["validation"]

        self.max_num_samples = max_num_samples

    def __len__(self):
        return self.max_num_samples or len(self.dataset)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError

        item = self.dataset[idx]

        question = self._TEMPLATE.format(
            preprompt=self._PREPROMPT,
            context=item["context"],
            question=item["question"],
        )
        should_abstain = item["answers"]["text"] == []
        reference_answers = (
            list(set(item["answers"]["text"])) if not should_abstain else None
        )
        metadata = {"SQuAD2.0_id": item["id"]}

        return Prompt(
            question=question,
            reference_answers=reference_answers,
            should_abstain=should_abstain,
            metadata=metadata,
        )
