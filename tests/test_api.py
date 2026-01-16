import unittest
import json
import os
from sense_v2.api.app import create_app
from sense_v2.database.database import initialize_database
from sense_v2.models.user import User

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

    def test_logout_user(self):
        """Test user logout."""
        # First, register and login a user
        self.client.post('/register', data=json.dumps({
            'email': 'test@example.com',
            'password': 'password123'
        }), content_type='application/json')
        self.client.post('/login', data=json.dumps({
            'email': 'test@example.com',
            'password': 'password123'
        }), content_type='application/json')

        # Now, try to logout
        response = self.client.post('/logout')

        self.assertEqual(response.status_code, 200)

    def test_profile_access(self):
        """Test access to the protected profile endpoint."""
        # Try to access profile without logging in
        response = self.client.get('/profile')
        self.assertEqual(response.status_code, 401)

        # Register and login a user
        self.client.post('/register', data=json.dumps({
            'email': 'test@example.com',
            'password': 'password123'
        }), content_type='application/json')
        with self.client.session_transaction() as session:
            user = User.find_by_email('test@example.com')
            session['user_id'] = user.id

        # Access profile again
        response = self.client.get('/profile')
        self.assertEqual(response.status_code, 200)

    def test_authentication_flow(self):
        """Test the complete authentication flow."""
        # Register a new user
        response = self.client.post('/register', data=json.dumps({
            'email': 'flow@example.com',
            'password': 'password123'
        }), content_type='application/json')
        self.assertEqual(response.status_code, 201)

        # Log in
        with self.client.session_transaction() as session:
            user = User.find_by_email('flow@example.com')
            session['user_id'] = user.id

        # Access protected profile
        response = self.client.get('/profile')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['email'], 'flow@example.com')

        # Log out
        response = self.client.post('/logout')
        self.assertEqual(response.status_code, 200)

        # Try to access profile again
        response = self.client.get('/profile')
        self.assertEqual(response.status_code, 401)

if __name__ == '__main__':
    unittest.main()
