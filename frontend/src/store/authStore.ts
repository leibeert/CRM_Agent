import { create } from 'zustand';

interface User {
  id: number;
  username: string;
  email: string;
}

interface AuthState {
  user: User | null;
  token: string | null;
  isInitialized: boolean;
  setAuth: (user: User, token: string) => void;
  clearAuth: () => void;
  initializeAuth: () => Promise<boolean>;
}

const useAuthStore = create<AuthState>((set, get) => ({
  // Don't auto-load from localStorage - require explicit initialization
  user: null,
  token: null,
  isInitialized: false,
  setAuth: (user, token) => {
    localStorage.setItem('user', JSON.stringify(user));
    localStorage.setItem('token', token);
    set({ user, token, isInitialized: true });
  },
  clearAuth: () => {
    localStorage.removeItem('user');
    localStorage.removeItem('token');
    set({ user: null, token: null, isInitialized: true });
  },
  initializeAuth: async () => {
    const storedToken = localStorage.getItem('token');
    const storedUser = localStorage.getItem('user');
    
    if (!storedToken || !storedUser) {
      set({ user: null, token: null, isInitialized: true });
      return false;
    }

    try {
      // Validate the token by making a test API call
      const response = await fetch('http://localhost:5000/api/auth/validate', {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${storedToken}`,
          'Content-Type': 'application/json',
        },
      });

      if (response.ok) {
        const user = JSON.parse(storedUser);
        set({ user, token: storedToken, isInitialized: true });
        return true;
      } else {
        // Token is invalid, clear stored data
        localStorage.removeItem('user');
        localStorage.removeItem('token');
        set({ user: null, token: null, isInitialized: true });
        return false;
      }
    } catch (error) {
      // Network error or other issue, assume not authenticated
      localStorage.removeItem('user');
      localStorage.removeItem('token');
      set({ user: null, token: null, isInitialized: true });
      return false;
    }
  },
}));

export default useAuthStore; 