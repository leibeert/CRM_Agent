import axios from 'axios';

const API_URL = 'http://localhost:5000';

const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add a request interceptor to add the auth token to requests
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Add a response interceptor to handle token expiration
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // Clear auth data and redirect to login
      localStorage.removeItem('token');
      localStorage.removeItem('user');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

// Auth functions
export const login = async (username: string, password: string) => {
  const response = await api.post('/login', { username, password });
  return response.data;
};

export const signup = async (username: string, password: string, email: string) => {
  const response = await api.post('/signup', { username, password, email });
  return response.data;
};

// Chat functions
export const createConversation = async (userId: number) => {
  const response = await api.post('/conversations', {
    title: `Chat ${new Date().toLocaleString()}`,
    participant_ids: [userId]
  });
  return response.data;
};

export const getConversations = async (userId: number) => {
  const response = await api.get(`/conversations/${userId}`);
  return response.data;
};

export const deleteConversation = async (conversationId: number) => {
  const response = await api.delete(`/conversations/${conversationId}`);
  return response.data;
};

export const renameConversation = async (conversationId: number, newTitle: string) => {
  const response = await api.put(`/conversations/${conversationId}`, {
    title: newTitle
  });
  return response.data;
};

export const sendMessage = async (conversationId: number, content: string, senderId: number, model?: string) => {
  const response = await api.post(`/conversations/${conversationId}/messages`, {
    content,
    sender_id: senderId,
    model: model
  });
  return response.data;
};

export const editMessage = async (messageId: number, newContent: string) => {
  const response = await api.put(`/messages/${messageId}/edit`, {
    content: newContent
  });
  return response.data;
};

export const deleteMessage = async (messageId: number) => {
  const response = await api.delete(`/messages/${messageId}/delete`);
  return response.data;
};

export const getMessages = async (conversationId: number) => {
  const response = await api.get(`/conversations/${conversationId}/messages`);
  return response.data;
};

// Search functions
export const searchCandidates = async (query: any) => {
  const response = await api.post('/api/search/candidates', query);
  return response.data;
};

// Enhanced Search functions
export const enhancedSearchCandidates = async (searchData: any) => {
  const response = await api.post('/api/enhanced-search/candidates', searchData);
  return response.data;
};

export const getSkillRecommendations = async (jobDescription: string, currentSkills: string[]) => {
  const response = await api.post('/api/enhanced-search/skill-recommendations', {
    job_description: jobDescription,
    current_skills: currentSkills
  });
  return response.data;
};

export const getMarketAnalysis = async (skills: string[], location: string = 'global') => {
  const response = await api.post('/api/enhanced-search/market-analysis', {
    skills,
    location
  });
  return response.data;
};

export const parseJobDescription = async (jobDescription: string, jobTitle?: string, companyName?: string) => {
  const response = await api.post('/api/enhanced-search/parse-job', {
    job_description: jobDescription,
    job_title: jobTitle,
    company_name: companyName
  });
  return response.data;
};

export const scoreSingleCandidate = async (candidateData: any, jobRequirements: any) => {
  const response = await api.post('/api/enhanced-search/score-candidate', {
    candidate_data: candidateData,
    job_requirements: jobRequirements
  });
  return response.data;
};

export const getEnhancedSearchHealth = async () => {
  const response = await api.get('/api/enhanced-search/health');
  return response.data;
};

export const saveSearch = async (searchData: any) => {
  const response = await api.post('/api/search/save', searchData);
  return response.data;
};

export const getSavedSearches = async (userId: number) => {
  const response = await api.get(`/api/search/saved/${userId}`);
  return response.data;
};

export const deleteSavedSearch = async (searchId: number) => {
  const response = await api.delete(`/api/search/saved/${searchId}`);
  return response.data;
};

// AI Model Management functions
export const getAvailableModels = async () => {
  const response = await api.get('/api/models');
  return response.data;
};

export const testModel = async (modelName: string) => {
  const response = await api.post(`/api/models/test/${modelName}`);
  return response.data;
};

export default api; 