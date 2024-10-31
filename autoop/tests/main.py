
import unittest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from autoop.tests.test_database import TestDatabase
from autoop.tests.test_storage import TestStorage
from autoop.tests.test_features import TestFeatures
from autoop.tests.test_pipeline import TestPipeline

if __name__ == '__main__':
    unittest.main()
