import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Container,
  Paper,
  TextField,
  Button,
  Typography,
  Box,
  Alert,
  IconButton,
  InputAdornment,
  Divider,
  Link,
  useTheme as useMuiTheme,
} from '@mui/material';
import {
  Visibility,
  VisibilityOff,
  LightMode,
  DarkMode,
  Email,
  Lock,
  Google,
} from '@mui/icons-material';
import useAuthStore from '../store/authStore';
import { login } from '../utils/api';
import { useTheme } from '../contexts/ThemeContext';

const Login: React.FC = () => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const navigate = useNavigate();
  const setAuth = useAuthStore((state) => state.setAuth);
  const { isDarkMode, toggleTheme } = useTheme();
  const muiTheme = useMuiTheme();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');

    try {
      const data = await login(username, password);
      setAuth(data.user, data.token);
      navigate('/dashboard');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred. Please try again.');
    }
  };

  const handleClickShowPassword = () => {
    setShowPassword(!showPassword);
  };

  return (
    <Box
      sx={{
        minHeight: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: isDarkMode 
          ? 'linear-gradient(135deg, #0f0f0f 0%, #1a1a1a 100%)'
          : 'linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%)',
        position: 'relative',
      }}
    >
      {/* Theme Toggle Button */}
      <IconButton
        onClick={toggleTheme}
        sx={{
          position: 'absolute',
          top: 24,
          right: 24,
          backgroundColor: muiTheme.palette.background.paper,
          border: `1px solid ${muiTheme.palette.divider}`,
          '&:hover': {
            backgroundColor: muiTheme.palette.action.hover,
          },
        }}
      >
        {isDarkMode ? <LightMode /> : <DarkMode />}
      </IconButton>

      <Container component="main" maxWidth="sm">
        <Paper
          elevation={0}
          sx={{
            padding: { xs: 3, sm: 6 },
            borderRadius: 3,
            backgroundColor: muiTheme.palette.background.paper,
            border: `1px solid ${muiTheme.palette.divider}`,
            boxShadow: isDarkMode 
              ? '0 20px 25px -5px rgba(0, 0, 0, 0.3), 0 10px 10px -5px rgba(0, 0, 0, 0.1)'
              : '0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)',
          }}
        >
          {/* Logo and Title */}
          <Box sx={{ textAlign: 'center', mb: 4 }}>
            <Box
              sx={{
                width: 80,
                height: 80,
                borderRadius: '50%',
                background: 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                margin: '0 auto 24px',
                boxShadow: '0 8px 32px rgba(99, 102, 241, 0.3)',
              }}
            >
              <Typography
                variant="h4"
                sx={{
                  color: 'white',
                  fontWeight: 700,
                  fontFamily: '"Inter", sans-serif',
                }}
              >
                C
              </Typography>
            </Box>
            <Typography
              variant="h4"
              sx={{
                fontWeight: 700,
                mb: 1,
                background: 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)',
                backgroundClip: 'text',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
              }}
            >
              CRM System
            </Typography>
            <Typography
              variant="body1"
              color="text.secondary"
              sx={{ mb: 3 }}
            >
              Only login via email, Google, or phone number login is supported in your region.
            </Typography>
          </Box>

          {error && (
            <Alert 
              severity="error" 
              sx={{ 
                mb: 3,
                borderRadius: 2,
                backgroundColor: isDarkMode ? '#2d1b1b' : '#fef2f2',
                border: `1px solid ${isDarkMode ? '#7f1d1d' : '#fecaca'}`,
              }}
            >
              {error}
            </Alert>
          )}

          <Box component="form" onSubmit={handleSubmit}>
            <TextField
              margin="normal"
              required
              fullWidth
              id="username"
              placeholder="Phone number / email address"
              name="username"
              autoComplete="username"
              autoFocus
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              InputProps={{
                startAdornment: (
                  <InputAdornment position="start">
                    <Email sx={{ color: 'text.secondary' }} />
                  </InputAdornment>
                ),
              }}
              sx={{
                mb: 2,
                '& .MuiOutlinedInput-root': {
                  height: 56,
                },
              }}
            />

            <TextField
              margin="normal"
              required
              fullWidth
              name="password"
              placeholder="Password"
              type={showPassword ? 'text' : 'password'}
              id="password"
              autoComplete="current-password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              InputProps={{
                startAdornment: (
                  <InputAdornment position="start">
                    <Lock sx={{ color: 'text.secondary' }} />
                  </InputAdornment>
                ),
                endAdornment: (
                  <InputAdornment position="end">
                    <IconButton
                      aria-label="toggle password visibility"
                      onClick={handleClickShowPassword}
                      edge="end"
                    >
                      {showPassword ? <VisibilityOff /> : <Visibility />}
                    </IconButton>
                  </InputAdornment>
                ),
              }}
              sx={{
                mb: 3,
                '& .MuiOutlinedInput-root': {
                  height: 56,
                },
              }}
            />

            <Typography
              variant="body2"
              color="text.secondary"
              sx={{ mb: 3, textAlign: 'center' }}
            >
              By signing up or logging in, you consent to CRM's{' '}
              <Link href="#" underline="hover" color="primary">
                Terms of Use
              </Link>{' '}
              and{' '}
              <Link href="#" underline="hover" color="primary">
                Privacy Policy
              </Link>
              .
            </Typography>

            <Button
              type="submit"
              fullWidth
              variant="contained"
              sx={{
                height: 48,
                mb: 3,
                background: 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)',
                '&:hover': {
                  background: 'linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%)',
                },
              }}
            >
              Log in
            </Button>

            <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 3 }}>
              <Link
                href="#"
                variant="body2"
                color="primary"
                underline="hover"
                sx={{ fontWeight: 500 }}
              >
                Forgot password?
              </Link>
              <Link
                href="#"
                variant="body2"
                color="primary"
                underline="hover"
                sx={{ fontWeight: 500 }}
              >
                Sign up
              </Link>
            </Box>

            <Divider sx={{ mb: 3 }}>
              <Typography variant="body2" color="text.secondary">
                OR
              </Typography>
            </Divider>

            <Button
              fullWidth
              variant="outlined"
              startIcon={<Google />}
              sx={{
                height: 48,
                borderColor: muiTheme.palette.divider,
                color: 'text.primary',
                '&:hover': {
                  borderColor: muiTheme.palette.primary.main,
                  backgroundColor: muiTheme.palette.action.hover,
                },
              }}
            >
              Log in with Google
            </Button>
          </Box>
        </Paper>
      </Container>
    </Box>
  );
};

export default Login; 