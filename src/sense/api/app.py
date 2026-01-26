from flask import Flask, request, jsonify, session
from sense.models.user import User
from sense.utils.security import hash_password, verify_password
from sense import cli
from sense.config import Config
from functools import wraps
from collections import defaultdict
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    In-memory rate limiter using sliding window algorithm.
    For production, consider Redis-backed implementation.
    """
    def __init__(self, max_requests=10, window_seconds=60):
        self.max_requests = max_requests
        self.window = timedelta(seconds=window_seconds)
        self.requests = defaultdict(list)  # IP -> [timestamp, ...]

    def is_allowed(self, identifier):
        """Check if request from identifier is allowed."""
        now = datetime.now()

        # Clean old requests outside the time window
        self.requests[identifier] = [
            ts for ts in self.requests[identifier]
            if now - ts < self.window
        ]

        # Check if under limit
        if len(self.requests[identifier]) < self.max_requests:
            self.requests[identifier].append(now)
            return True

        return False

    def get_retry_after(self, identifier):
        """Get seconds until next request is allowed."""
        if not self.requests[identifier]:
            return 0

        oldest = min(self.requests[identifier])
        retry_time = oldest + self.window
        seconds_left = (retry_time - datetime.now()).total_seconds()
        return max(0, int(seconds_left))


# Global rate limiters for different endpoint types
auth_limiter = RateLimiter(max_requests=5, window_seconds=60)    # 5 requests per minute
api_limiter = RateLimiter(max_requests=20, window_seconds=60)     # 20 requests per minute


def rate_limit(limiter):
    """
    Decorator to apply rate limiting to endpoints.

    Usage:
        @rate_limit(auth_limiter)
        def my_endpoint():
            ...
    """
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            # Get client identifier (IP address)
            identifier = request.remote_addr or 'unknown'

            if not limiter.is_allowed(identifier):
                retry_after = limiter.get_retry_after(identifier)
                logger.warning(f"Rate limit exceeded for {identifier}")
                return jsonify({
                    'error': 'Rate limit exceeded',
                    'retry_after_seconds': retry_after
                }), 429

            return f(*args, **kwargs)
        return wrapped
    return decorator

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    cli.init_app(app)

    @app.route('/register', methods=['POST'])
    @rate_limit(auth_limiter)
    def register():
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')

        if not email or not password:
            return jsonify({'error': 'Email and password are required'}), 400

        if User.find_by_email(email):
            return jsonify({'error': 'User already exists'}), 400

        hashed_pw = hash_password(password)
        user = User.create(email, hashed_pw)

        return jsonify({'id': user.id, 'email': user.email}), 201

    @app.route('/login', methods=['POST'])
    @rate_limit(auth_limiter)
    def login():
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')

        if not email or not password:
            return jsonify({'error': 'Email and password are required'}), 400

        user = User.find_by_email(email)

        if not user or not verify_password(password, user.password_hash):
            return jsonify({'error': 'Invalid credentials'}), 401
        
        session['user_id'] = user.id

        return jsonify({'message': 'Login successful'}), 200

    @app.route('/logout', methods=['POST'])
    def logout():
        session.pop('user_id', None)
        return jsonify({'message': 'Logout successful'}), 200

    @app.route('/profile')
    @rate_limit(api_limiter)
    def profile():
        if 'user_id' not in session:
            return jsonify({'error': 'Unauthorized'}), 401

        user = User.find_by_id(session['user_id'])
        if not user:
            return jsonify({'error': 'User not found'}), 404
            
        return jsonify({'id': user.id, 'email': user.email})

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)
