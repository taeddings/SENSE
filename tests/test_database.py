import sqlite3
import unittest
import os
from sense_v2.database.database import initialize_database, get_db_connection

class TestDatabase(unittest.TestCase):

    def setUp(self):
        """Set up a clean database for each test."""
        if os.path.exists('sense.db'):
            os.remove('sense.db')

    def tearDown(self):
        """Clean up the database file after each test."""
        if os.path.exists('sense.db'):
            os.remove('sense.db')

    def test_users_table_creation(self):
        """Test that the users table is created correctly."""
        # Run the initialization
        initialize_database()

        # Connect to the database and check for the table
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users';")
        table_exists = cursor.fetchone()
        conn.close()

        self.assertIsNotNone(table_exists, "The 'users' table was not created.")

if __name__ == '__main__':
    unittest.main()
