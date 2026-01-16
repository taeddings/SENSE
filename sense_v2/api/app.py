from flask import Flask, request, jsonify
from sense_v2.models.user import User
from sense_v2.utils.security import hash_password

def create_app():
    app = Flask(__name__)

    @app.route('/register', methods=['POST'])
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

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)
