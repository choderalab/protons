from protons.app.analysis.processing import dataset_to_dataframe
import pytest
import pandas as pd
from os import path
from uuid import uuid4

test_root = path.abspath(path.dirname(__file__))


class TestProcessing:

    one_residue_calibration_data = []
    multi_residue_calibration_data = []
    one_residue_ais_data = []
    multi_residue_ais_data = []
    ncmc_benchmark_data = []

    def test_single_dataset(self):
        df = dataset_to_dataframe([path.join(test_root, "viologen-b1.nc")])
        assert len(df) == 1, "The result should be a list with one dataframe."
        assert len(df[0]) == 1000, "There should be 3000 entries."
        return

    def test_multiple_datasets_merged(self):
        df = dataset_to_dataframe(
            [
                path.join(test_root, "viologen-{}.nc".format(x))
                for x in ["b1", "b2", "b3"]
            ],
            return_as_merged=True,
        )

        assert len(df) == 1, "The result should be a list with one dataframe."
        assert len(df[0]) == 3000, "There should be 3000 entries."

        return

    def test_multiple_datasets_separate(self):
        dfs = dataset_to_dataframe(
            [
                path.join(test_root, "viologen-{}.nc".format(x))
                for x in ["b1", "b2", "b3"]
            ],
            return_as_merged=False,
        )
        assert len(dfs) == 3, "The result should be a list with three dataframes."
        for i, df in enumerate(dfs):
            assert len(df) == 1000, f"df {i} has the wrong size."

        return

    def test_ncmc_traj_dataset(self):
        df = dataset_to_dataframe(
            [path.join(test_root, "viologen-importance-state-1-25000-a1.nc")]
        )
        assert len(df) == 1, "The result should be a list with one dataframe."
        assert len(df[0]) == 98, "There should be 303 entries."

        return

    def test_multiple_ncmc_traj_datasets(self):
        datasets = [
            path.join(test_root, "viologen-importance-state-1-{}-a1.nc".format(x))
            for x in ["25000", "50000", "100000"]
        ]
        sizes = [98, 36, 22]
        dfs = dataset_to_dataframe(datasets, return_as_merged=False)
        assert len(dfs) == 3, "The result should be a list with three dataframes."
        for i, df in enumerate(dfs):
            assert len(df) == sizes[i], f"df {i} has the wrong size."
        return

    def test_multiple_ncmc_traj_datasets_merged(self):
        datasets = [
            path.join(test_root, "viologen-importance-state-1-{}-a1.nc".format(x))
            for x in ["25000", "50000", "100000"]
        ]
        dfs = dataset_to_dataframe(datasets, return_as_merged=True)
        assert len(dfs) == 1, "The result should be a list with one dataframe."
        for i, df in enumerate(dfs):
            assert len(df) == 156, f"df {i} has the wrong size."
        return
