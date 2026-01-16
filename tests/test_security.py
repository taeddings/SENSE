import unittest
from sense_v2.utils.security import hash_password, verify_password

class TestSecurity(unittest.TestCase):

    def test_password_hashing_and_verification(self):
        """Test that password hashing and verification work correctly."""
        password = "mysecretpassword"
        
        # Hash the password
        hashed_password = hash_password(password)
        
        # Verify the correct password
        self.assertTrue(verify_password(password, hashed_password))
        
        # Verify an incorrect password
        self.assertFalse(verify_password("wrongpassword", hashed_password))

if __name__ == '__main__':
    unittest.main()
