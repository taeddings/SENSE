import sqlite3
from .schema import create_users_table

def get_db_connection():
    """Establishes a connection to the SQLite database."""
    conn = sqlite3.connect('sense.db')
    conn.row_factory = sqlite3.Row
    return conn

def initialize_database():
    """Initializes the database by creating necessary tables."""
    conn = get_db_connection()
    create_users_table(conn)
    conn.commit()
    conn.close()

if __name__ == '__main__':
    initialize_database()
