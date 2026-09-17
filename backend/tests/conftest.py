import os
from pathlib import Path


TEST_DB = Path(__file__).parent / "test_parking.db"
os.environ["DATABASE_URL"] = f"sqlite:///{TEST_DB.as_posix()}"
