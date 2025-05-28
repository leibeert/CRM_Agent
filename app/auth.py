"""
Authentication utilities for the Flask application.
"""

import os
import jwt as PyJWT
from datetime import datetime, timedelta
from functools import wraps
from flask import request, jsonify, current_app

from db import AuthSessionLocal, User

# JWT configuration
JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', 'your-secret-key')  # Change this in production
JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=1)


def token_required(f):
    """Decorator to require JWT token authentication."""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        auth_header = request.headers.get('Authorization')
        
        if auth_header and auth_header.startswith('Bearer '):
            token = auth_header.split(' ')[1]
        
        if not token:
            return jsonify({'error': 'Token is missing'}), 401
        
        try:
            data = PyJWT.decode(token, JWT_SECRET_KEY, algorithms=['HS256'])
            db = AuthSessionLocal()
            current_user = db.query(User).filter_by(id=data['user_id']).first()
            db.close()
            
            if not current_user:
                return jsonify({'error': 'Invalid token'}), 401
                
        except PyJWT.ExpiredSignatureError:
            return jsonify({'error': 'Token has expired'}), 401
        except PyJWT.InvalidTokenError:
            return jsonify({'error': 'Invalid token'}), 401
            
        return f(current_user, *args, **kwargs)
    return decorated 