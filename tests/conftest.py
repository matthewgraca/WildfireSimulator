import pytest
from wildfire_simulator.dataloader import WildfireDataLoader, TrialFileLoader, TrialCollection
from dotenv import load_dotenv
from pathlib import Path

# Load test .env BEFORE any dataloader imports that call load_dotenv()
load_dotenv(Path(__file__).parent / ".tenv", override=True)

@pytest.fixture(scope="session")
def dataloader():
    trials = TrialCollection(TrialFileLoader())
    return WildfireDataLoader(trials)
