"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

import pytest
from hydra import compose, initialize

from main import main
from recipe.evaluation import Responses


@pytest.mark.parametrize("run_single_job_for_inference_and_judge", [True, False])
def test_main_with_dummy_data_and_dummy_model(
    tmp_path, run_single_job_for_inference_and_judge
):
    logs_dir = tmp_path
    save_dir = os.path.join(logs_dir, "results/")
    num_samples = 100
    with initialize(version_base=None, config_path="../configs"):
        # config is relative to a module
        cfg = compose(
            config_name="default_pipeline.yaml",
            overrides=[
                "dataset=dummy",
                "model=dummy",
                "abstention_detector=contains_abstention_keyword",
                f"datamodule.num_samples={num_samples}",
                f"run_single_job_for_inference_and_judge={run_single_job_for_inference_and_judge}",
                f"logs_dir={logs_dir}",
                f"save_dir={save_dir}",
            ],
        )
        main(cfg)
    results_path = os.path.join(save_dir, "GroundTruthAbstentionEvaluator.json")
    print("results path", results_path)
    responses = Responses.load(results_path)
    assert len(responses) == num_samples


@pytest.mark.slow
def test_main_two_stage_dummy_end_to_end_pipeline(tmp_path):
    """
    To test end-to-end run:
    python main.py -m mode=cluster model=llama_3_1_8B_instruct dataset=big_bench_known_unknowns abstention_detector=llm_judge_llama_3_1_8B_instruct
    """
    logs_dir = tmp_path
    Path(logs_dir).mkdir(exist_ok=True, parents=True)
    save_dir = os.path.join(logs_dir, "results/")
    num_samples = 100
    with initialize(version_base=None, config_path="../configs"):
        # config is relative to a module
        cfg = compose(
            config_name="default_pipeline.yaml",
            overrides=[
                "dataset=dummy",
                f"datamodule.num_samples={num_samples}",
                "mode=cluster",
                "model=dummy",
                "abstention_detector=contains_abstention_keyword",
                f"logs_dir={logs_dir}",
                f"save_dir={save_dir}",
            ],
        )
        main(cfg)
    # check saved files
    results_path = os.path.join(save_dir, "GroundTruthAbstentionEvaluator.json")
    print("results path", results_path)
    responses = Responses.load(results_path)
    assert len(responses) == num_samples


def test_main_with_dataset_indices_subset(tmp_path):
    logs_dir = tmp_path
    save_dir = os.path.join(logs_dir, "results/")
    num_samples = 100
    with initialize(version_base=None, config_path="../configs"):
        # config is relative to a module
        cfg = compose(
            config_name="default_pipeline.yaml",
            overrides=[
                "dataset=dummy",
                "model=dummy",
                "abstention_detector=contains_abstention_keyword",
                "dataset_indices_path=null",
                "dataset_indices_subset=[0, 5, 10]",
                f"logs_dir={logs_dir}",
                f"save_dir={save_dir}",
            ],
        )
        main(cfg)

    results_path = os.path.join(save_dir, "GroundTruthAbstentionEvaluator.json")
    responses = Responses.load(results_path)

    # Length should match number of dataset_indices_subset items
    assert len(responses) == 3


def test_main_with_dataset_indices_path(tmp_path):
    logs_dir = tmp_path

    # Write a tmp file with 3 indices for DummyDataset, and a different number for a different dataset
    dataset_indices_path = tmp_path / "test-subsampling-indices.json"
    with open(dataset_indices_path, "w") as f:
        json.dump({"DummyDataset": [0, 5, 10], "DifferentDataset": [0, 2, 4, 6, 8]}, f)

    save_dir = os.path.join(logs_dir, "results/")
    with initialize(version_base=None, config_path="../configs"):
        # config is relative to a module
        cfg = compose(
            config_name="default_pipeline.yaml",
            overrides=[
                "dataset=dummy",
                "model=dummy",
                "abstention_detector=contains_abstention_keyword",
                f"dataset_indices_path={dataset_indices_path}",
                "dataset_indices_subset=null",
                f"logs_dir={logs_dir}",
                f"save_dir={save_dir}",
            ],
        )
        main(cfg)

    results_path = os.path.join(save_dir, "GroundTruthAbstentionEvaluator.json")
    responses = Responses.load(results_path)

    # Length should match number of DummyDataset indices specified
    assert len(responses) == 3


def test_main_with_dataset_indices_subset_and_dataset_indices_path_raises(tmp_path):
    logs_dir = tmp_path
    save_dir = os.path.join(logs_dir, "results/")
    with initialize(version_base=None, config_path="../configs"):
        # config is relative to a module
        cfg = compose(
            config_name="default_pipeline.yaml",
            overrides=[
                "dataset=dummy",
                "model=dummy",
                "abstention_detector=contains_abstention_keyword",
                "dataset_indices_path=some-path.json",
                "dataset_indices_subset=[0, 5, 10]",
                f"logs_dir={logs_dir}",
                f"save_dir={save_dir}",
            ],
        )

        with pytest.raises(ValueError) as e:
            main(cfg)

        assert (
            e.value.args[0]
            == "Can't set both dataset_indices_subset and dataset_indices_path"
        )
