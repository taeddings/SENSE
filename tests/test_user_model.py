import unittest
import os
from sense.models.user import User
from sense.database.database import initialize_database

class TestUserModel(unittest.TestCase):

    def setUp(self):
        """Set up a clean database for each test."""
        if os.path.exists('sense.db'):
            os.remove('sense.db')
        initialize_database()

    def tearDown(self):
        """Clean up the database file after each test."""
        if os.path.exists('sense.db'):
            os.remove('sense.db')

    def test_create_and_find_user(self):
        """Test creating a user and finding them by email."""
        email = "test@example.com"
        password_hash = "some_hash"
        
        # Create a user
        user = User.create(email, password_hash)
        self.assertIsNotNone(user)
        
        # Find the user by email
        found_user = User.find_by_email(email)
        
        self.assertIsNotNone(found_user)
        self.assertEqual(found_user.email, email)

if __name__ == '__main__':
    unittest.main()
