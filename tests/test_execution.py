import pytest 
import os

config_settings = [
    f"configs/{config}.yaml" for config in ["lsst", "astro-nn", "gz2"]
]


@pytest.mark.parametrize(
    "config, data", zip(config_settings, ["1", "2", "3"]), ids=["lsst", "astro-nn", "gz2"],
    )
def test_pipeline(config, execute, data, mock_data, output_path): 

    mock_data_dir = mock_data(data)

    cmd = f"astrouda run --config {config} --data {mock_data_dir} --output {output_path} train"
    exitcode = execute(cmd)
    assert exitcode == 0

    cmd = f"astrouda run --config {config} --data {mock_data_dir} --output {output_path} inference"
    exitcode = execute(cmd)
    assert exitcode == 0

    assert os.path.join()