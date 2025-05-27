import subprocess
from typing import Literal
import pytest

@pytest.fixture(scope="function")
def execute(command: str) -> None: 
    command = [i for i in command.split(" ") if i not in ('', ' ')]  # Strip out the random spaces
    process = subprocess.run(command, capture_output=True)
    print(process.stdout.decode())
    print(process.stderr.decode())
    return process.returncode

@pytest.fixture(scope="session")
def mock_data() -> callable: 
    ""
    def make_mock_data(type_: Literal["lsst", "galaxy"]) -> str: 
        ""

    yield make_mock_data # Returns the function, then deletes the data path after the session

def mock_lsst_data() -> str: 
    "Return mock LSST data path"

def mock_galaxy_data() -> str: 
    "Mock galazy zoo data path - contains markers for sdss, decals, stripe 82 deep fields"

@pytest.fixture(scope="session")
def output_path() -> str: 
    ""

@pytest.fixture(scope="session")
def config_path(settings: dict) -> str: 
    ""