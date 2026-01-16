import unittest
import json
import os
from sense_v2.api.app import create_app
from sense_v2.database.database import initialize_database

class TestApi(unittest.TestCase):

    def setUp(self):
        """Set up a clean database and a test client for each test."""
        if os.path.exists('sense.db'):
            os.remove('sense.db')
        initialize_database()
        self.app = create_app()
        self.client = self.app.test_client()

    def tearDown(self):
        """Clean up the database file after each test."""
        if os.path.exists('sense.db'):
            os.remove('sense.db')

    def test_register_user(self):
        """Test user registration."""
        response = self.client.post('/register', data=json.dumps({
            'email': 'test@example.com',
            'password': 'password123'
        }), content_type='application/json')
        
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data)
        self.assertEqual(data['email'], 'test@example.com')
        self.assertIn('id', data)

    def test_login_user(self):
        """Test user login."""
        # First, register a user
        self.client.post('/register', data=json.dumps({
            'email': 'test@example.com',
            'password': 'password123'
        }), content_type='application/json')

        # Now, try to login
        response = self.client.post('/login', data=json.dumps({
            'email': 'test@example.com',
            'password': 'password123'
        }), content_type='application/json')

        self.assertEqual(response.status_code, 200)

if __name__ == '__main__':
    unittest.main()
