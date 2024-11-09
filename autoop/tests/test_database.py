import random
import tempfile
import unittest

from autoop.core.database import Database
from autoop.core.storage import LocalStorage


class TestDatabase(unittest.TestCase):
    """Unit tests for the Database class."""

    def setUp(self) -> None:
        """Set up a storage location and initialize the Database instance."""
        self.storage = LocalStorage(tempfile.mkdtemp())
        self.db = Database(self.storage)

    def test_init(self) -> None:
        """Test that the Database instance is initialized correctly."""
        self.assertIsInstance(self.db, Database)

    def test_set(self) -> None:
        """Test that an entry can be added and retrieved."""
        id = str(random.randint(0, 100))
        entry = {"key": random.randint(0, 100)}
        self.db.set("collection", id, entry)
        self.assertEqual(self.db.get("collection", id)["key"], entry["key"])

    def test_delete(self) -> None:
        """Test that an entry can be deleted and is no longer retrievable."""
        id = str(random.randint(0, 100))
        value = {"key": random.randint(0, 100)}
        self.db.set("collection", id, value)
        self.db.delete("collection", id)
        self.assertIsNone(self.db.get("collection", id))
        self.db.refresh()
        self.assertIsNone(self.db.get("collection", id))

    def test_persistance(self) -> None:
        """Test that data is persisted correctly."""
        id = str(random.randint(0, 100))
        value = {"key": random.randint(0, 100)}
        self.db.set("collection", id, value)
        other_db = Database(self.storage)
        self.assertEqual(other_db.get("collection", id)["key"], value["key"])

    def test_refresh(self) -> None:
        """Test that the refresh method reloads data accurately."""
        key = str(random.randint(0, 100))
        value = {"key": random.randint(0, 100)}
        other_db = Database(self.storage)
        self.db.set("collection", key, value)
        other_db.refresh()
        self.assertEqual(other_db.get("collection", key)["key"], value["key"])

    def test_list(self) -> None:
        """Test that all entries in a collection are listed correctly."""
        key = str(random.randint(0, 100))
        value = {"key": random.randint(0, 100)}
        self.db.set("collection", key, value)
        self.assertIn((key, value), self.db.list("collection"))
