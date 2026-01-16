from sense_v2.database.database import initialize_database
from sense_v2.models.user import User
from sense_v2.utils.security import hash_password, verify_password

def verify_backend_foundation():
    """
    This script verifies the backend foundation phase of the user authentication track.
    It performs the following steps:
    1. Initializes the database (creates the 'users' table).
    2. Creates a new user with a hashed password.
    3. Retrieves the user from the database.
    4. Verifies the password.
    """
    print("Starting verification for Phase 1: Backend Foundation...")

    # 1. Initialize the database
    initialize_database()
    print("Database initialized successfully.")

    # 2. Create a new user
    email = "verify@example.com"
    password = "password123"
    hashed_pw = hash_password(password)
    User.create(email, hashed_pw)
    print(f"User '{email}' created successfully.")

    # 3. Retrieve the user
    retrieved_user = User.find_by_email(email)
    if not retrieved_user:
        print("Verification FAILED: Could not retrieve the user.")
        return

    print(f"User '{email}' retrieved successfully.")

    # 4. Verify the password
    if verify_password(password, retrieved_user.password_hash):
        print("Password verification successful.")
        print("\nVerification PASSED!")
    else:
        print("Verification FAILED: Password verification failed.")

if __name__ == '__main__':
    verify_backend_foundation()
