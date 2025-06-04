import os
import subprocess
from typing import Literal
import pytest

@pytest.fixture(scope="function")
def execute() -> callable: 
    def internal_execute(command: str) -> int:
        """
        Execute a command in the shell and return the exit code.
        """
        command = [i for i in command.split(" ") if i not in ('', ' ')]  # Strip out the random spaces
        process = subprocess.run(command, capture_output=True)
        print(process.stdout.decode())
        print(process.stderr.decode())
        return process.returncode
    return internal_execute

@pytest.fixture(scope="session")
def mock_data() -> callable: 
    ""
    _dir = "tests/mock_data"
    if not os.path.exists(_dir):
        os.makedirs(_dir)
        
    def make_mock_data(type_: Literal["lsst", "astro-nn", 'gz2']) -> str: 
        ""

    yield make_mock_data # Returns the function, then deletes the data path after the session



def _mock_lsst_data(): 
    pass

def _mock_galaxy_data(): 
    pass


@pytest.fixture(scope="session")
def output_path() -> str: 
    ""

@pytest.fixture(scope="session")
def config_path(settings: dict) -> str: 
    ""