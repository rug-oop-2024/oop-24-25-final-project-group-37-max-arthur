import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from autoop.tests.test_database import TestDatabase  # noqa
from autoop.tests.test_features import TestFeatures  # noqa
from autoop.tests.test_metrics import TestMetrics  # noqa
from autoop.tests.test_pipeline import TestPipeline  # noqa
from autoop.tests.test_storage import TestStorage  # noqa

if __name__ == '__main__':
    unittest.main()
