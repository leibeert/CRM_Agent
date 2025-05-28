# 🚀 AI-Powered CRM System

A sophisticated Customer Relationship Management system with advanced AI-powered candidate matching, built with Flask (Python) backend and React (TypeScript) frontend.

## ✨ Features

### 🤖 AI-Powered Search
- **Enhanced Candidate Matching**: Multi-dimensional AI scoring using OpenAI GPT-3.5
- **Semantic Skill Matching**: Intelligent skill similarity using sentence transformers
- **Job Description Parsing**: Automatic extraction of requirements from job descriptions
- **Skill Recommendations**: AI-generated learning paths and skill gap analysis
- **Market Analysis**: Demand trends and salary insights for skills

### 💬 Intelligent Chat System
- **Conversational Interface**: Natural language candidate queries
- **Real-time Responses**: Instant AI-powered candidate suggestions
- **Context-Aware**: Remembers conversation history and context
- **Multi-format Support**: Text, markdown, and rich media responses

### 🔍 Advanced Search & Filtering
- **Multi-criteria Search**: Skills, experience, education, location
- **Saved Searches**: Store and reuse frequent search queries
- **Export Functionality**: Download candidate data in various formats
- **Real-time Filtering**: Dynamic search results with instant updates

### 🔐 Security & Authentication
- **JWT Authentication**: Secure token-based authentication
- **Role-based Access**: Different permission levels for users
- **Protected Routes**: All sensitive endpoints secured
- **Session Management**: Automatic token refresh and logout

### 📊 Analytics & Insights
- **Match Score Breakdown**: Detailed scoring explanations
- **Candidate Analytics**: Performance metrics and trends
- **Search Insights**: Success rates and optimization suggestions
- **Market Intelligence**: Industry trends and competitive analysis

## 🏗️ Architecture

### Backend (Flask + Python)
- **Flask**: Web framework with RESTful API design
- **SQLAlchemy**: ORM for database management
- **OpenAI Integration**: GPT-3.5 for job parsing and analysis
- **Sentence Transformers**: Semantic similarity calculations
- **JWT**: Authentication and authorization
- **MySQL**: Primary database for candidate data

### Frontend (React + TypeScript)
- **React 18**: Modern component-based UI
- **TypeScript**: Type-safe development
- **Material-UI**: Professional design system
- **Zustand**: Lightweight state management
- **Axios**: HTTP client with interceptors
- **Vite**: Fast build tool and development server

### AI/ML Components
- **OpenAI GPT-3.5**: Job description parsing and analysis
- **Sentence Transformers**: Skill semantic matching
- **Scikit-learn**: Similarity calculations and ML utilities
- **Custom Scoring Engine**: Multi-dimensional candidate evaluation

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Node.js 16+
- MySQL 8.0+
- Git

### Backend Setup

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd crm
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment configuration**
   ```bash
   cp .env.example .env
   # Edit .env with your database and API credentials
   ```

5. **Database setup**
   ```bash
   # Import the provided database tables
   mysql -u root -p < database_tables/setup.sql
   ```

6. **Run the backend**
   ```bash
   cd app
   python -m flask --app app run --host=0.0.0.0 --port=5000 --debug
   ```

### Frontend Setup

1. **Navigate to frontend directory**
   ```bash
   cd frontend
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Start development server**
   ```bash
   npm run dev
   ```

4. **Access the application**
   - Frontend: http://localhost:5173
   - Backend API: http://localhost:5000

## 📁 Project Structure

```
crm/
├── app/                          # Backend Flask application
│   ├── api/                      # API route blueprints
│   ├── matching/                 # AI matching engine
│   │   ├── core/                 # Core matching algorithms
│   │   ├── parsers/              # Job description parsers
│   │   └── utils/                # Utilities and helpers
│   ├── models/                   # Database models
│   ├── search/                   # Search functionality
│   └── app.py                    # Main Flask application
├── frontend/                     # React frontend
│   ├── src/
│   │   ├── components/           # React components
│   │   ├── store/                # State management
│   │   ├── utils/                # Utilities and API calls
│   │   └── contexts/             # React contexts
│   └── package.json
├── database_tables/              # Database schema and data
├── config/                       # Configuration files
├── requirements.txt              # Python dependencies
└── README.md
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the root directory:

```env
# Database Configuration
AUTH_DB_HOST=localhost
AUTH_DB_PORT=3306
AUTH_DB_NAME=auth_db
AUTH_DB_USER=root
AUTH_DB_PASSWORD=your_password

ARGOTEAM_DB_HOST=localhost
ARGOTEAM_DB_PORT=3306
ARGOTEAM_DB_NAME=argoteam
ARGOTEAM_DB_USER=root
ARGOTEAM_DB_PASSWORD=your_password

# JWT Configuration
JWT_SECRET_KEY=your-secret-key

# OpenAI Configuration
OPENAI_API_KEY=your-openai-api-key

# Redis (Optional)
REDIS_URL=redis://localhost:6379
```

## 🧪 Testing

### Backend Tests
```bash
cd app
python -m pytest tests/
```

### Frontend Tests
```bash
cd frontend
npm test
```

## 📊 Database Schema

The system uses two MySQL databases:

1. **auth_db**: User authentication and chat functionality
2. **argoteam**: Candidate data and search functionality

Key tables:
- `resources`: Candidate profiles
- `skills`: Available skills
- `resource_skills`: Candidate-skill relationships
- `experiences`: Work experience
- `studies`: Educational background

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI for GPT-3.5 integration
- Hugging Face for sentence transformers
- Material-UI for the design system
- The open-source community for various libraries and tools

## 📞 Support

For support, email support@yourcompany.com or create an issue in this repository.

---

**Built with ❤️ by Your Development Team** 