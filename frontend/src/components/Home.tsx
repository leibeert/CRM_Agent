import React from 'react';
import {
  Container,
  Typography,
  Paper,
  Box,
  Grid,
  Card,
  CardContent,
  IconButton,
  useTheme,
} from '@mui/material';
import {
  Chat,
  Search,
  People,
  TrendingUp,
  Bookmark,
  Analytics,
  LightMode,
  DarkMode,
} from '@mui/icons-material';
import useAuthStore from '../store/authStore';
import { useTheme as useCustomTheme } from '../contexts/ThemeContext';

const Home: React.FC = () => {
  const user = useAuthStore((state) => state.user);
  const { isDarkMode, toggleTheme } = useCustomTheme();
  const theme = useTheme();

  const statsCards = [
    {
      title: 'Total Candidates',
      value: '1,234',
      icon: <People />,
      color: '#6366f1',
      change: '+12%',
    },
    {
      title: 'Active Searches',
      value: '23',
      icon: <Search />,
      color: '#10b981',
      change: '+5%',
    },
    {
      title: 'Saved Searches',
      value: '45',
      icon: <Bookmark />,
      color: '#f59e0b',
      change: '+8%',
    },
    {
      title: 'Success Rate',
      value: '87%',
      icon: <TrendingUp />,
      color: '#ef4444',
      change: '+3%',
    },
  ];

  const quickActions = [
    {
      title: 'AI Chat Assistant',
      description: 'Get help with candidate matching and queries',
      icon: <Chat />,
      color: '#6366f1',
      path: '/chat',
    },
    {
      title: 'Search Candidates',
      description: 'Find the perfect candidates for your needs',
      icon: <Search />,
      color: '#10b981',
      path: '/search',
    },
    {
      title: 'Analytics Dashboard',
      description: 'View detailed insights and reports',
      icon: <Analytics />,
      color: '#f59e0b',
      path: '/analytics',
    },
  ];

  return (
    <Box sx={{ flexGrow: 1, p: 3 }}>
      {/* Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 4 }}>
        <Box>
          <Typography
            variant="h4"
            sx={{
              fontWeight: 700,
              background: 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)',
              backgroundClip: 'text',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              mb: 1,
            }}
          >
            Welcome back, {user?.username}!
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Here's what's happening with your CRM today.
          </Typography>
        </Box>
        <IconButton
          onClick={toggleTheme}
          sx={{
            backgroundColor: theme.palette.background.paper,
            border: `1px solid ${theme.palette.divider}`,
            '&:hover': {
              backgroundColor: theme.palette.action.hover,
            },
          }}
        >
          {isDarkMode ? <LightMode /> : <DarkMode />}
        </IconButton>
      </Box>

      {/* Stats Cards */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        {statsCards.map((stat, index) => (
          <Grid xs={12} sm={6} md={3} key={index}>
            <Card
              sx={{
                background: isDarkMode
                  ? 'linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%)'
                  : 'linear-gradient(135deg, #ffffff 0%, #f8fafc 100%)',
                border: `1px solid ${theme.palette.divider}`,
                borderRadius: 3,
                transition: 'all 0.3s ease',
                '&:hover': {
                  transform: 'translateY(-4px)',
                  boxShadow: isDarkMode
                    ? '0 20px 25px -5px rgba(0, 0, 0, 0.3)'
                    : '0 20px 25px -5px rgba(0, 0, 0, 0.1)',
                },
              }}
            >
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <Box>
                    <Typography variant="body2" color="text.secondary" gutterBottom>
                      {stat.title}
                    </Typography>
                    <Typography variant="h4" sx={{ fontWeight: 700, mb: 1 }}>
                      {stat.value}
                    </Typography>
                    <Typography
                      variant="body2"
                      sx={{
                        color: '#10b981',
                        fontWeight: 500,
                      }}
                    >
                      {stat.change} from last month
                    </Typography>
                  </Box>
                  <Box
                    sx={{
                      width: 56,
                      height: 56,
                      borderRadius: 2,
                      backgroundColor: `${stat.color}20`,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      color: stat.color,
                    }}
                  >
                    {stat.icon}
                  </Box>
                </Box>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      {/* Quick Actions */}
      <Typography variant="h5" sx={{ fontWeight: 600, mb: 3 }}>
        Quick Actions
      </Typography>
      <Grid container spacing={3}>
        {quickActions.map((action, index) => (
          <Grid xs={12} md={4} key={index}>
            <Card
              sx={{
                background: isDarkMode
                  ? 'linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%)'
                  : 'linear-gradient(135deg, #ffffff 0%, #f8fafc 100%)',
                border: `1px solid ${theme.palette.divider}`,
                borderRadius: 3,
                cursor: 'pointer',
                transition: 'all 0.3s ease',
                '&:hover': {
                  transform: 'translateY(-4px)',
                  boxShadow: isDarkMode
                    ? '0 20px 25px -5px rgba(0, 0, 0, 0.3)'
                    : '0 20px 25px -5px rgba(0, 0, 0, 0.1)',
                },
              }}
              onClick={() => window.location.href = action.path}
            >
              <CardContent sx={{ p: 3 }}>
                <Box
                  sx={{
                    width: 64,
                    height: 64,
                    borderRadius: 2,
                    backgroundColor: `${action.color}20`,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    color: action.color,
                    mb: 2,
                  }}
                >
                  {action.icon}
                </Box>
                <Typography variant="h6" sx={{ fontWeight: 600, mb: 1 }}>
                  {action.title}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {action.description}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>
    </Box>
  );
};

export default Home; 