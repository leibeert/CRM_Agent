from flask import Flask, request, jsonify, render_template, send_file
from candidate_matcher import CandidateMatcher
import os
from dotenv import load_dotenv
from db import AuthSessionLocal, ArgoteamSessionLocal, User, Conversation, Message, AuthBase, ArgoteamBase, auth_engine, argoteam_engine, conversation_participants, Resource, Skill, ResourceSkill, Experience, Study, DegreeType, School
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
from flask_cors import CORS
import logging
from werkzeug.utils import secure_filename
import pandas as pd
import io
from search import SearchService, SearchQuery, SavedSearchCreate, SkillFilter, ExperienceFilter, EducationFilter
import jwt as PyJWT
from functools import wraps

# Import enhanced search blueprint
from api.enhanced_search_routes import enhanced_search_bp

# Import auth utilities
from auth import token_required, JWT_SECRET_KEY, JWT_ACCESS_TOKEN_EXPIRES

# Import necessary Langchain components
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Register blueprints
app.register_blueprint(enhanced_search_bp)

matcher = CandidateMatcher()

# Initialize Langchain components
# Ensure OPENAI_API_KEY is set in your .env file
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# Define a general prompt for the chatbot
general_chatbot_prompt = PromptTemplate.from_template(
    "You are a helpful recruitment assistant chatbot. Respond to the user's query.\n\nUser: {query}\nAgent:"
)

# Define a prompt for explaining candidate matching
explanation_prompt = PromptTemplate.from_template(
    "You are a recruitment assistant chatbot explaining candidate matching results.\nExplain to the user how the candidates were matched based on the provided job description and their skills.\n\nJob Description: {job_description}\nCandidate Matches: {match_results}\n\nExplain the matching process:"
)

output_parser = StrOutputParser()

# Create Langchain chains
general_chatbot_chain = (
    {"query": RunnablePassthrough()} 
    | general_chatbot_prompt 
    | llm 
    | output_parser
)

explanation_chain = (
    RunnablePassthrough() 
    | explanation_prompt 
    | llm 
    | output_parser
)

# Helper function to get LLM response
def get_agent_response(query: str, context: dict = None, explanation_context: dict = None) -> str:
    """Get response from the LLM, potentially with context."""
    try:
        if explanation_context:
            # Use explanation chain if explanation context is provided
            return explanation_chain.invoke(explanation_context)
        else:
            # Use general chatbot chain for other queries
            return general_chatbot_chain.invoke({"query": query})
    except Exception as e:
        logger.error(f"Error getting LLM response: {str(e)}")
        return "I'm sorry, I couldn't process that request right now."

# Create database tables
try:
    AuthBase.metadata.create_all(bind=auth_engine)
    logger.info("Database tables created successfully")
except Exception as e:
    logger.error(f"Error creating database tables: {str(e)}")
    raise

# @app.route('/')
# def index():
#     """Serve the React app."""
#     return render_template('index.html')

@app.route('/signup', methods=['POST'])
def signup():
    """Handle user signup."""
    try:
        data = request.get_json()
        logger.debug(f"Received signup request with data: {data}")
        
        username = data.get('username')
        password = data.get('password')
        email = data.get('email')

        if not username or not password or not email:
            logger.warning("Missing required fields in signup request")
            return jsonify({'error': 'Username, password, and email are required'}), 400

        db = AuthSessionLocal()
        try:
            existing_user = db.query(User).filter_by(username=username).first()
            if existing_user:
                logger.warning(f"Username {username} already exists")
                return jsonify({'error': 'Username already exists'}), 409

            existing_email = db.query(User).filter_by(email=email).first()
            if existing_email:
                logger.warning(f"Email {email} already exists")
                return jsonify({'error': 'Email already exists'}), 409

            hashed_password = generate_password_hash(password)
            new_user = User(username=username, hashed_password=hashed_password, email=email)
            db.add(new_user)
            db.commit()
            db.refresh(new_user)
            logger.info(f"Successfully created user: {username}")
            
            return jsonify({
                'message': 'User created successfully',
                'user': {
                    'id': new_user.id,
                    'username': new_user.username,
                    'email': new_user.email
                }
            }), 201

        except Exception as e:
            db.rollback()
            logger.error(f"Database error during signup: {str(e)}")
            raise
        finally:
            db.close()

    except Exception as e:
        logger.error(f"Error during signup: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/login', methods=['POST'])
def login():
    """Handle user login."""
    try:
        data = request.get_json()
        logger.debug(f"Received login request with data: {data}")
        
        username = data.get('username')
        password = data.get('password')

        if not username or not password:
            logger.warning("Missing username or password in login request")
            return jsonify({'error': 'Username and password are required'}), 400

        db = AuthSessionLocal()
        try:
            user = db.query(User).filter_by(username=username).first()
            if not user:
                logger.warning(f"User not found: {username}")
                return jsonify({'error': 'Invalid username or password'}), 401

            if not check_password_hash(user.hashed_password, password):
                logger.warning(f"Invalid password for user: {username}")
                return jsonify({'error': 'Invalid username or password'}), 401

            # Generate JWT token
            token = PyJWT.encode(
                {
                    'user_id': user.id,
                    'username': user.username,
                    'exp': datetime.utcnow() + JWT_ACCESS_TOKEN_EXPIRES
                },
                JWT_SECRET_KEY,
                algorithm='HS256'
            )

            logger.info(f"Successful login for user: {username}")
            return jsonify({
                'message': 'Login successful',
                'user': {
                    'id': user.id,
                    'username': user.username,
                    'email': user.email
                },
                'token': token
            }), 200
        finally:
            db.close()

    except Exception as e:
        logger.error(f"Error during login: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'healthy'})

@app.route('/conversations', methods=['POST'])
@token_required
def create_conversation(current_user):
    """Create a new conversation."""
    db = AuthSessionLocal()
    try:
        data = request.get_json()
        logger.debug(f"Received conversation creation request: {data}")
        
        title = data.get('title')
        participant_ids = data.get('participant_ids', [])
        
        if not title or not participant_ids:
            logger.warning("Missing title or participant_ids in conversation creation")
            return jsonify({'error': 'Title and participant IDs are required'}), 400

        # Verify all participants exist
        participants = db.query(User).filter(User.id.in_(participant_ids)).all()
        if len(participants) != len(participant_ids):
            logger.warning(f"Some participants not found. Requested: {participant_ids}, Found: {[p.id for p in participants]}")
            return jsonify({'error': 'One or more participants not found'}), 404

        # Create conversation
        conversation = Conversation(title=title)
        conversation.participants = participants
        db.add(conversation)
        db.commit()
        db.refresh(conversation)
        
        response = {
            'id': conversation.id,
            'title': conversation.title,
            'created_at': conversation.created_at.isoformat(),
            'updated_at': conversation.updated_at.isoformat(),
            'participants': [{'id': p.id, 'username': p.username} for p in participants]
        }
        
        logger.info(f"Successfully created conversation: {conversation.id}")
        return jsonify(response), 201

    except Exception as e:
        db.rollback()
        logger.error(f"Error creating conversation: {str(e)}")
        return jsonify({'error': str(e)}), 500
    finally:
        db.close()

@app.route('/conversations/<int:user_id>', methods=['GET'])
@token_required
def get_user_conversations(current_user, user_id):
    """Get all conversations for a user."""
    db = AuthSessionLocal()
    try:
        logger.debug(f"Fetching conversations for user {user_id}")
        user = db.query(User).filter_by(id=user_id).first()
        
        if not user:
            logger.warning(f"User {user_id} not found")
            return jsonify({'error': 'User not found'}), 404

        # Get conversations where user is a participant
        conversations = db.query(Conversation).join(
            conversation_participants
        ).filter(
            conversation_participants.c.user_id == user_id
        ).all()

        logger.debug(f"Found {len(conversations)} conversations for user {user_id}")
        
        response = []
        for conv in conversations:
            # Get the last message for each conversation
            last_message = db.query(Message).filter_by(
                conversation_id=conv.id
            ).order_by(Message.created_at.desc()).first()

            response.append({
                'id': conv.id,
                'title': conv.title,
                'created_at': conv.created_at.isoformat(),
                'updated_at': conv.updated_at.isoformat(),
                'participants': [{'id': p.id, 'username': p.username} for p in conv.participants],
                'last_message': {
                    'content': last_message.content,
                    'created_at': last_message.created_at.isoformat()
                } if last_message else None
            })

        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error fetching conversations: {str(e)}")
        return jsonify({'error': str(e)}), 500
    finally:
        db.close()

@app.route('/conversations/<int:conversation_id>/messages', methods=['POST'])
@token_required
def send_message(current_user, conversation_id):
    """Send a message in a conversation."""
    try:
        data = request.get_json()
        logger.debug(f"Received message request: {data}")
        
        sender_id = data.get('sender_id')
        content = data.get('content')
        is_agent = data.get('is_agent', False)

        if not sender_id or not content:
            logger.warning("Missing sender_id or content in message request")
            return jsonify({'error': 'Sender ID and content are required'}), 400

        db = AuthSessionLocal()
        try:
            # Verify conversation exists
            conversation = db.query(Conversation).filter_by(id=conversation_id).first()
            if not conversation:
                logger.warning(f"Conversation {conversation_id} not found")
                db.close()
                return jsonify({'error': 'Conversation not found'}), 404

            # For agent messages, we don't need to verify the sender
            if not is_agent:
                sender = db.query(User).filter_by(id=sender_id).first()
                if not sender or sender not in conversation.participants:
                    logger.warning(f"Sender {sender_id} not found or not a participant in conversation {conversation_id}")
                    db.close()
                    return jsonify({'error': 'Sender not found or not a participant'}), 403

            # Create message
            message = Message(
                conversation_id=conversation_id,
                sender_id=sender_id,
                content=content,
                is_agent=is_agent
            )
            db.add(message)
            db.commit()
            db.refresh(message)
            logger.info(f"Created message {message.id} in conversation {conversation_id}")

            # If it's a user message, determine agent response
            if not is_agent:
                job_keywords = ['job', 'position', 'role', 'vacancy', 'opening', 'hiring', 'looking for', 'need', 'required', 'developer', 'engineer', 'programmer']
                explanation_keywords = ['explain', 'how', 'why', 'calculate', 'matching', 'score']

                is_job_description = any(keyword in content.lower() for keyword in job_keywords)
                is_explanation_request = any(keyword in content.lower() for keyword in explanation_keywords)

                response_content = ""
                generate_agent_response = True

                if is_job_description:
                    try:
                        logger.info("Processing job description for candidate matching")
                        # Find matching candidates
                        matches = matcher.find_matching_candidates(content)
                        logger.info(f"Found {len(matches)} matching candidates")

                        # Format the response message using Markdown
                        if matches:
                            response_content = "**Here are the matching candidates:**\n\n"
                            for i, candidate in enumerate(matches, 1):
                                response_content += f"**{i}. {candidate['first_name']} {candidate['last_name']}** ({(candidate['match_score'] * 100):.1f}% match)\n"
                                if candidate['match_details']:
                                    response_content += "*Skills match:*\n"
                                    for detail in candidate['match_details']:
                                        response_content += f"* {detail}\n"
                                response_content += "\n"
                        else:
                           response_content = "I couldn't find any matching candidates based on your request."

                    except Exception as e:
                        logger.error(f"Error in candidate matching: {str(e)}")
                        response_content = "I'm sorry, I encountered an error while trying to find matching candidates. Please try again."

                elif is_explanation_request:
                     logger.info("Processing explanation request")
                     # To provide a good explanation, we need the context of the previous job description
                     # and the match results. This would typically involve fetching the last message
                     # from the agent that contained match results.

                     previous_messages = db.query(Message).filter_by(
                         conversation_id=conversation_id,
                         is_agent=True
                     ).order_by(Message.created_at.desc()).limit(5).all() # Look at last 5 agent messages

                     last_match_message = None
                     for prev_msg in previous_messages:
                         if "Here are the matching candidates:" in prev_msg.content:
                             last_match_message = prev_msg.content
                             break

                     if last_match_message:
                         # Use LLM to explain based on the last match results and the original job description
                         # Note: Retrieving the *original* job description requires more sophisticated context handling.
                         # For this simplified version, I'll use the last agent message containing results.
                         explanation_context = {
                             "job_description": "(Previous job description - needs to be retrieved)", # Placeholder, ideally fetch actual previous JD
                             "match_results": last_match_message
                         }
                         response_content = get_agent_response(query=content, explanation_context=explanation_context)

                     else:
                         # If no recent match results found, ask for clarification or give a general explanation
                         response_content = "I can explain the matching process, but I need to know which job description and results you're asking about. Please provide the job description again or refer to a previous result."

                else:
                    logger.info("Processing general chat query")
                    # Use LLM for general queries
                    response_content = get_agent_response(query=content)

                # Send the agent's response if response_content was generated
                if response_content:
                    response_message = Message(
                        conversation_id=conversation_id,
                        sender_id=sender_id, # Using sender_id for simplicity, but could use a dedicated agent user ID
                        content=response_content,
                        is_agent=True
                    )
                    db.add(response_message)
                    db.commit()
                    db.refresh(response_message)
                    logger.info(f"Created agent response message {response_message.id}")

            response = {
                'id': message.id,
                'content': message.content,
                'created_at': message.created_at.isoformat(),
                'sender': {'id': sender_id, 'username': 'Agent' if is_agent else db.query(User).filter_by(id=sender_id).first().username}, # Fetch username if not agent
                'is_agent': message.is_agent
            }
            
            logger.debug(f"Sending response: {response}")
            return jsonify(response), 201

        except Exception as e:
            db.rollback()
            logger.error(f"Database error in send_message: {str(e)}")
            # Consider sending an error message back to the user via the chat if possible
            # For now, re-raise the exception to be caught by the outer try-except
            raise
        finally:
            db.close()

    except Exception as e:
        logger.error(f"Error in send_message: {str(e)}")
        # Decide how to handle top-level errors - perhaps return a generic error message
        return jsonify({'error': str(e)}), 500

@app.route('/conversations/<int:conversation_id>/messages', methods=['GET'])
@token_required
def get_conversation_messages(current_user, conversation_id):
    """Get all messages in a conversation."""
    try:
        db = AuthSessionLocal()
        conversation = db.query(Conversation).filter_by(id=conversation_id).first()
        
        if not conversation:
            db.close()
            return jsonify({'error': 'Conversation not found'}), 404

        # Eager load sender relationship to avoid N+1 queries
        messages = db.query(Message).filter_by(conversation_id=conversation_id).order_by(Message.created_at).all()

        response = []
        for msg in messages:
             sender_username = 'Agent' if msg.is_agent and msg.sender_id is not None else (msg.sender.username if msg.sender else 'Unknown User')

             response.append({
                'id': msg.id,
                'content': msg.content,
                'created_at': msg.created_at.isoformat(),
                'is_read': bool(msg.is_read),
                'is_agent': bool(msg.is_agent),
                'is_edited': bool(msg.is_edited), # Include edited status
                'is_deleted': bool(msg.is_deleted), # Include deleted status
                'read_at': msg.read_at.isoformat() if msg.read_at else None, # Include read_at timestamp
                'updated_at': msg.updated_at.isoformat(), # Include updated_at timestamp
                'sender': {
                    'id': msg.sender_id,
                    'username': sender_username
                },
                'reactions': [{'id': r.id, 'reaction': r.reaction, 'user_id': r.user_id} for r in msg.reactions], # Include reactions
                'attachments': [{'id': a.id, 'file_name': a.file_name, 'file_type': a.file_type, 'file_size': a.file_size} for a in msg.attachments] # Include attachments

            })

        db.close()
        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error getting conversation messages: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/messages/<int:message_id>/read', methods=['POST'])
@token_required
def mark_message_read(current_user, message_id):
    """Mark a message as read."""
    try:
        db = AuthSessionLocal()
        message = db.query(Message).filter_by(id=message_id).first()
        
        if not message:
            db.close()
            return jsonify({'error': 'Message not found'}), 404

        message.is_read = True
        message.read_at = datetime.utcnow()
        db.commit()
        db.close()

        return jsonify({'message': 'Message marked as read'}), 200

    except Exception as e:
        logger.error(f"Error marking message as read: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/messages/<int:message_id>/edit', methods=['PUT'])
@token_required
def edit_message(current_user, message_id):
    """Edit a message and delete all subsequent messages in the conversation."""
    try:
        data = request.get_json()
        new_content = data.get('content')
        
        if not new_content:
            return jsonify({'error': 'New content is required'}), 400

        db = AuthSessionLocal()
        message = db.query(Message).filter_by(id=message_id).first()
        
        if not message:
            db.close()
            return jsonify({'error': 'Message not found'}), 404

        # Store edit history
        if not message.edit_history:
            message.edit_history = []
        message.edit_history.append({
            'content': message.content,
            'edited_at': message.updated_at.isoformat()
        })

        # Delete all messages that come after this message in the conversation
        subsequent_messages = db.query(Message).filter(
            Message.conversation_id == message.conversation_id,
            Message.created_at > message.created_at
        ).all()
        
        for subsequent_message in subsequent_messages:
            db.delete(subsequent_message)
        
        logger.info(f"Deleted {len(subsequent_messages)} subsequent messages after editing message {message_id}")

        # Update the message content
        message.content = new_content
        message.is_edited = True
        message.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(message)
        
        response = {
            'id': message.id,
            'content': message.content,
            'is_edited': message.is_edited,
            'updated_at': message.updated_at.isoformat(),
            'deleted_subsequent_count': len(subsequent_messages)
        }
        
        db.close()
        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error editing message: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/messages/<int:message_id>/delete', methods=['DELETE'])
@token_required
def delete_message(current_user, message_id):
    """Delete a message."""
    try:
        db = AuthSessionLocal()
        message = db.query(Message).filter_by(id=message_id).first()
        
        if not message:
            db.close()
            return jsonify({'error': 'Message not found'}), 404

        message.is_deleted = True
        message.content = "This message was deleted"
        db.commit()
        db.close()

        return jsonify({'message': 'Message deleted successfully'}), 200

    except Exception as e:
        logger.error(f"Error deleting message: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/messages/<int:message_id>/reactions', methods=['POST'])
@token_required
def add_reaction(current_user, message_id):
    """Add a reaction to a message."""
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        reaction = data.get('reaction')

        if not user_id or not reaction:
            return jsonify({'error': 'User ID and reaction are required'}), 400

        db = AuthSessionLocal()
        message = db.query(Message).filter_by(id=message_id).first()
        
        if not message:
            db.close()
            return jsonify({'error': 'Message not found'}), 404

        # Check if user already reacted with this emoji
        existing_reaction = db.query(MessageReaction).filter_by(
            message_id=message_id,
            user_id=user_id,
            reaction=reaction
        ).first()

        if existing_reaction:
            # Remove reaction if it already exists
            db.delete(existing_reaction)
        else:
            # Add new reaction
            new_reaction = MessageReaction(
                message_id=message_id,
                user_id=user_id,
                reaction=reaction
            )
            db.add(new_reaction)

        db.commit()
        db.close()

        return jsonify({'message': 'Reaction updated successfully'}), 200

    except Exception as e:
        logger.error(f"Error adding reaction: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/messages/<int:message_id>/attachments', methods=['POST'])
@token_required
def add_attachment(current_user, message_id):
    """Add an attachment to a message."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        db = AuthSessionLocal()
        message = db.query(Message).filter_by(id=message_id).first()
        
        if not message:
            db.close()
            return jsonify({'error': 'Message not found'}), 404

        # Create upload directory if it doesn't exist
        upload_dir = os.path.join(app.root_path, 'uploads')
        os.makedirs(upload_dir, exist_ok=True)

        # Save file
        filename = secure_filename(file.filename)
        file_path = os.path.join(upload_dir, filename)
        file.save(file_path)

        # Create attachment record
        attachment = MessageAttachment(
            message_id=message_id,
            file_name=filename,
            file_type=file.content_type,
            file_size=os.path.getsize(file_path),
            file_path=file_path
        )
        db.add(attachment)
        db.commit()
        db.refresh(attachment)
        db.close()

        return jsonify({
            'id': attachment.id,
            'file_name': attachment.file_name,
            'file_type': attachment.file_type,
            'file_size': attachment.file_size
        }), 201

    except Exception as e:
        logger.error(f"Error adding attachment: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/conversations/<int:conversation_id>', methods=['DELETE'])
@token_required
def delete_conversation(current_user, conversation_id):
    """Delete a conversation."""
    db = AuthSessionLocal()
    try:
        logger.debug(f"Attempting to delete conversation {conversation_id}")
        conversation = db.query(Conversation).filter_by(id=conversation_id).first()
        
        if not conversation:
            logger.warning(f"Conversation {conversation_id} not found")
            return jsonify({'error': 'Conversation not found'}), 404

        # Delete all messages in the conversation first
        db.query(Message).filter_by(conversation_id=conversation_id).delete()
        
        # Delete the conversation
        db.delete(conversation)
        db.commit()
        
        logger.info(f"Successfully deleted conversation {conversation_id}")
        return jsonify({'message': 'Conversation deleted successfully'}), 200

    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting conversation: {str(e)}")
        return jsonify({'error': str(e)}), 500
    finally:
        db.close()

@app.route('/conversations/<int:conversation_id>', methods=['PUT'])
@token_required
def update_conversation(current_user, conversation_id):
    """Update a conversation's title."""
    db = AuthSessionLocal()
    try:
        data = request.get_json()
        new_title = data.get('title')
        
        if not new_title:
            return jsonify({'error': 'New title is required'}), 400

        conversation = db.query(Conversation).filter_by(id=conversation_id).first()
        if not conversation:
            return jsonify({'error': 'Conversation not found'}), 404

        conversation.title = new_title
        db.commit()
        db.refresh(conversation)
        
        response = {
            'id': conversation.id,
            'title': conversation.title,
            'updated_at': conversation.updated_at.isoformat()
        }
        
        return jsonify(response), 200

    except Exception as e:
        db.rollback()
        logger.error(f"Error updating conversation: {str(e)}")
        return jsonify({'error': str(e)}), 500
    finally:
        db.close()

@app.route('/download_candidates', methods=['POST'])
def download_candidates():
    """Download selected candidate information as an Excel file."""
    try:
        data = request.get_json()
        candidate_names = data.get('candidate_ids', [])  # This contains the full names
        
        logger.info(f"Received download request for candidate names: {candidate_names}")

        if not candidate_names:
            logger.warning("No candidate names provided in request")
            return jsonify({'error': 'No candidate names provided'}), 400

        argoteam_db = ArgoteamSessionLocal()
        try:
            # Search for candidates by their names
            candidates = []
            for full_name in candidate_names:
                # Split the name into first and last name
                name_parts = full_name.split()
                if len(name_parts) >= 2:
                    first_name = name_parts[0]
                    last_name = name_parts[-1]
                    
                    # Search for candidates with matching first and last name
                    candidate = argoteam_db.query(
                        Resource.id,
                        Resource.first_name,
                        Resource.last_name,
                        Resource.email,
                        Resource.phone_number
                    ).filter(
                        Resource.first_name.ilike(f"%{first_name}%"),
                        Resource.last_name.ilike(f"%{last_name}%")
                    ).first()
                    
                    if candidate:
                        candidates.append(candidate)
                    else:
                        logger.warning(f"No candidate found for name: {full_name}")

            if not candidates:
                logger.warning(f"No valid candidates found for names: {candidate_names}")
                return jsonify({'message': 'No valid candidates found'}), 404

            # Prepare data for DataFrame with contact info including phone
            candidates_data = []
            for candidate in candidates:
                candidates_data.append({
                    'ID': candidate.id,
                    'First Name': candidate.first_name,
                    'Last Name': candidate.last_name,
                    'Email': candidate.email,
                    'Phone Number': candidate.phone_number
                })

            logger.info(f"Prepared data for {len(candidates_data)} candidates")

            # Create a pandas DataFrame
            df = pd.DataFrame(candidates_data)

            # Create an Excel file in memory
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='Candidates')
                
                # Get the xlsxwriter workbook and worksheet objects
                workbook = writer.book
                worksheet = writer.sheets['Candidates']
                
                # Add some formatting
                header_format = workbook.add_format({
                    'bold': True,
                    'bg_color': '#D9E1F2',
                    'border': 1
                })
                
                # Write the column headers with the defined format
                for col_num, value in enumerate(df.columns.values):
                    worksheet.write(0, col_num, value, header_format)
                    # Adjust column width based on content
                    max_length = max(
                        df[value].astype(str).apply(len).max(),
                        len(str(value))
                    )
                    worksheet.set_column(col_num, col_num, max_length + 2)

            excel_buffer.seek(0)
            logger.info("Excel file created successfully")

            # Return the Excel file as a downloadable response
            return send_file(
                excel_buffer,
                mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                as_attachment=True,
                download_name='candidates.xlsx'
            )

        except Exception as e:
            logger.error(f"Database error fetching candidates for download: {str(e)}")
            raise
        finally:
            argoteam_db.close()

    except Exception as e:
        logger.error(f"Error generating candidate download file: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/search/candidates', methods=['POST'])
@token_required
def search_candidates(current_user):
    """Search candidates with advanced filtering."""
    try:
        data = request.get_json()
        search_service = SearchService()
        
        # Convert request data to SearchQuery
        query = SearchQuery(
            keywords=data.get('keywords'),
            skills=[SkillFilter(**skill) for skill in data.get('skills', [])],
            experience=ExperienceFilter(**data.get('experience', {})) if data.get('experience') else None,
            education=EducationFilter(**data.get('education', {})) if data.get('education') else None,
            location=data.get('location'),
            min_match_score=data.get('min_match_score'),
            sort_by=data.get('sort_by'),
            sort_order=data.get('sort_order'),
            page=data.get('page', 1),
            page_size=data.get('page_size', 10)
        )
        
        # Perform search
        results = search_service.search_candidates(query)
        return jsonify(results.dict()), 200
        
    except Exception as e:
        logger.error(f"Error in search_candidates: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/search/save', methods=['POST'])
@token_required
def save_search(current_user):
    """Save a search query for future use."""
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        
        if not user_id:
            return jsonify({'error': 'User ID is required'}), 400
            
        search_service = SearchService()
        
        # Create SavedSearchCreate object
        saved_search = SavedSearchCreate(
            name=data.get('name'),
            description=data.get('description'),
            filters=SearchQuery(**data.get('filters', {})),
            sort_by=data.get('sort_by'),
            sort_order=data.get('sort_order')
        )
        
        # Save search
        result = search_service.save_search(user_id, saved_search)
        return jsonify({
            'id': result.id,
            'name': result.name,
            'description': result.description,
            'filters': result.filters,
            'sort_by': result.sort_by,
            'sort_order': result.sort_order,
            'created_at': result.created_at.isoformat(),
            'updated_at': result.updated_at.isoformat()
        }), 201
        
    except Exception as e:
        logger.error(f"Error in save_search: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/search/saved/<int:user_id>', methods=['GET'])
@token_required
def get_saved_searches(current_user, user_id):
    """Get all saved searches for a user."""
    try:
        search_service = SearchService()
        saved_searches = search_service.get_saved_searches(user_id)
        
        return jsonify([{
            'id': search.id,
            'name': search.name,
            'description': search.description,
            'filters': search.filters,
            'sort_by': search.sort_by,
            'sort_order': search.sort_order,
            'created_at': search.created_at.isoformat(),
            'updated_at': search.updated_at.isoformat()
        } for search in saved_searches]), 200
        
    except Exception as e:
        logger.error(f"Error in get_saved_searches: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/search/saved/<int:search_id>', methods=['DELETE'])
@token_required
def delete_saved_search(current_user, search_id):
    """Delete a saved search."""
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        
        if not user_id:
            return jsonify({'error': 'User ID is required'}), 400
            
        search_service = SearchService()
        success = search_service.delete_saved_search(search_id, user_id)
        
        if success:
            return jsonify({'message': 'Search deleted successfully'}), 200
        else:
            return jsonify({'error': 'Search not found or unauthorized'}), 404
            
    except Exception as e:
        logger.error(f"Error in delete_saved_search: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(
        host=os.getenv('HOST', '0.0.0.0'),
        port=int(os.getenv('PORT', 5000)),
        debug=os.getenv('DEBUG', 'False').lower() == 'true'
    ) 