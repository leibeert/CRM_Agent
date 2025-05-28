import React, { useEffect } from 'react';
import { BrowserRouter as Router, Route, Routes, Navigate, Link, Outlet } from 'react-router-dom';
import { AppBar, Toolbar, Typography, Button, Container } from '@mui/material';
import Login from './components/Login';
import Search from './components/Search';
import EnhancedSearch from './components/EnhancedSearch';
import Home from './components/Home';
import Chat from './components/Chat';
import useAuthStore from './store/authStore';
import api from './utils/api';
import { CustomThemeProvider } from './contexts/ThemeContext';

// Protected Route component
const ProtectedRoute: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const user = useAuthStore((state) => state.user);
  const token = useAuthStore((state) => state.token);
  const clearAuth = useAuthStore((state) => state.clearAuth);

  useEffect(() => {
    // Verify token on mount
    if (token) {
      api.get('/health')
        .catch((error) => {
          if (error.response?.status === 401) {
            clearAuth();
          }
        });
    }
  }, [token, clearAuth]);

  if (!user || !token) {
    return <Navigate to="/login" replace />;
  }
  return <>{children}</>;
};

// Layout component with navigation
const Layout: React.FC = () => {
  const user = useAuthStore((state) => state.user);
  const token = useAuthStore((state) => state.token);
  const clearAuth = useAuthStore((state) => state.clearAuth);

  const handleLogout = () => {
    clearAuth();
    window.location.href = '/login';
  };

  return (
    <>
      {user && token && (
        <AppBar position="static">
          <Toolbar>
            <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
              CRM System
            </Typography>
            <Button color="inherit" component={Link} to="/dashboard">Dashboard</Button>
            <Button color="inherit" component={Link} to="/search">Search</Button>
            <Button color="inherit" component={Link} to="/enhanced-search">AI Search</Button>
            <Button color="inherit" component={Link} to="/chat">Chat</Button>
            <Button color="inherit" onClick={handleLogout}>
              Logout
            </Button>
          </Toolbar>
        </AppBar>
      )}
      <Container>
        <Outlet />
      </Container>
    </>
  );
};

const AppContent: React.FC = () => {
  const user = useAuthStore((state) => state.user);
  const token = useAuthStore((state) => state.token);

  return (
    <Router>
      <Routes>
        <Route 
          path="/login" 
          element={
            user && token ? <Navigate to="/dashboard" replace /> : <Login />
          } 
        />
        <Route 
          path="/" 
          element={
            user && token ? <Navigate to="/dashboard" replace /> : <Navigate to="/login" replace />
          } 
        />
        <Route path="/" element={<Layout />}>
          <Route 
            path="dashboard" 
            element={
              <ProtectedRoute>
                <Home />
              </ProtectedRoute>
            } 
          />
          <Route 
            path="search" 
            element={
              <ProtectedRoute>
                <Search />
              </ProtectedRoute>
            } 
          />
          <Route 
            path="enhanced-search" 
            element={
              <ProtectedRoute>
                <EnhancedSearch />
              </ProtectedRoute>
            } 
          />
          <Route 
            path="chat" 
            element={
              <ProtectedRoute>
                <Chat />
              </ProtectedRoute>
            } 
          />
        </Route>
        <Route 
          path="*" 
          element={<Navigate to="/login" replace />} 
        />
      </Routes>
    </Router>
  );
};

const App: React.FC = () => {
  return (
    <CustomThemeProvider>
      <AppContent />
    </CustomThemeProvider>
  );
};

export default App; 