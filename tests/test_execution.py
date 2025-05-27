import pytest 

config_settings = [

]


pytest.parametrize("config", config_settings, ids=[f"config_{i}" for i in range(len(config_settings))])
def test_pipeline(config, execute, mock_data, output_path, config_path): 
    ""