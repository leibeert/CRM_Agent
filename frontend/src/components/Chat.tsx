import React, { useState, useEffect, useRef } from 'react';
import {
  Box,
  Paper,
  Typography,
  TextField,
  IconButton,
  Avatar,
  Divider,
  Chip,
  Fade,
  useTheme,
  InputAdornment,
  Tooltip,
  CircularProgress,
  Menu,
  MenuItem,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  ListItemIcon,
  ListItemText,
  AppBar,
  Toolbar,
} from '@mui/material';
import {
  Send,
  Add,
  SmartToy,
  Person,
  MoreVert,
  AttachFile,
  EmojiEmotions,
  LightMode,
  DarkMode,
  Delete,
  Edit,
  Check,
  Close,
  DriveFileRenameOutline,
  Menu as MenuIcon,
  MenuOpen,
  Search as SearchIcon,
  Logout,
  Settings,
  AccountCircle,
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import useAuthStore from '../store/authStore';
import { 
  getConversations, 
  createConversation, 
  sendMessage, 
  getMessages,
  deleteConversation,
  renameConversation,
  editMessage,
  deleteMessage
} from '../utils/api';
import { useTheme as useCustomTheme } from '../contexts/ThemeContext';

interface Message {
  id: number;
  content: string;
  created_at: string;
  sender: {
    id: number;
    username: string;
  };
  is_agent: boolean;
  is_edited?: boolean;
}

interface Conversation {
  id: number;
  title: string;
  created_at: string;
  updated_at: string;
  participants: Array<{
    id: number;
    username: string;
  }>;
  last_message?: {
    content: string;
    created_at: string;
  };
}

const Chat: React.FC = () => {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [selectedConversation, setSelectedConversation] = useState<Conversation | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [newMessage, setNewMessage] = useState('');
  const [loading, setLoading] = useState(false);
  const [sending, setSending] = useState(false);
  const [error, setError] = useState('');
  const [editingMessageId, setEditingMessageId] = useState<number | null>(null);
  const [editingContent, setEditingContent] = useState('');
  const [conversationMenuAnchor, setConversationMenuAnchor] = useState<null | HTMLElement>(null);
  const [selectedConversationForMenu, setSelectedConversationForMenu] = useState<Conversation | null>(null);
  const [renameDialogOpen, setRenameDialogOpen] = useState(false);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [newConversationTitle, setNewConversationTitle] = useState('');
  const [sidebarVisible, setSidebarVisible] = useState(true);
  const [userMenuAnchor, setUserMenuAnchor] = useState<null | HTMLElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const user = useAuthStore((state) => state.user);
  const clearAuth = useAuthStore((state) => state.clearAuth);
  const { isDarkMode, toggleTheme } = useCustomTheme();
  const theme = useTheme();
  const navigate = useNavigate();

  useEffect(() => {
    if (user) {
      loadConversations();
    }
  }, [user]);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const loadConversations = async () => {
    try {
      setLoading(true);
      setError('');
      const data = await getConversations(user!.id);
      setConversations(data);
      
      // Auto-select first conversation if none selected
      if (data.length > 0 && !selectedConversation) {
        handleConversationSelect(data[0]);
      }
    } catch (err) {
      setError('Failed to load conversations. Please try again.');
      console.error('Error loading conversations:', err);
    } finally {
      setLoading(false);
    }
  };

  const loadMessages = async (conversationId: number) => {
    try {
      setLoading(true);
      setError('');
      const data = await getMessages(conversationId);
      setMessages(data);
    } catch (err) {
      setError('Failed to load messages. Please try again.');
      console.error('Error loading messages:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleConversationSelect = async (conversation: Conversation) => {
    setSelectedConversation(conversation);
    await loadMessages(conversation.id);
  };

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newMessage.trim() || !selectedConversation || !user || sending) return;

    const messageContent = newMessage.trim();
    setNewMessage('');
    setSending(true);

    try {
      setError('');
      
      // Add user message to UI immediately for better UX
      const tempUserMessage: Message = {
        id: Date.now(), // Temporary ID
        content: messageContent,
        created_at: new Date().toISOString(),
        sender: { id: user.id, username: user.username },
        is_agent: false
      };
      setMessages(prev => [...prev, tempUserMessage]);

      // Send message to backend
      const message = await sendMessage(selectedConversation.id, messageContent, user.id);
      
      // Replace temp message with real one and reload to get agent response
      await loadMessages(selectedConversation.id);
      await loadConversations(); // Refresh conversations to update last message
    } catch (err) {
      setError('Failed to send message. Please try again.');
      console.error('Error sending message:', err);
      // Remove the temporary message on error
      setMessages(prev => prev.filter(msg => msg.id !== Date.now()));
    } finally {
      setSending(false);
    }
  };

  const handleNewConversation = async () => {
    if (!user) return;
    
    try {
      setError('');
      const conversation = await createConversation(user.id);
      await loadConversations();
      handleConversationSelect(conversation);
    } catch (err) {
      setError('Failed to create new conversation. Please try again.');
      console.error('Error creating conversation:', err);
    }
  };

  const handleDeleteConversation = async () => {
    if (!selectedConversationForMenu) return;
    
    try {
      await deleteConversation(selectedConversationForMenu.id);
      
      // If the deleted conversation was selected, clear selection
      if (selectedConversation?.id === selectedConversationForMenu.id) {
        setSelectedConversation(null);
        setMessages([]);
      }
      
      await loadConversations();
      setDeleteDialogOpen(false);
      setConversationMenuAnchor(null);
    } catch (err) {
      setError('Failed to delete conversation. Please try again.');
      console.error('Error deleting conversation:', err);
    }
  };

  const handleRenameConversation = async () => {
    if (!selectedConversationForMenu || !newConversationTitle.trim()) return;
    
    try {
      await renameConversation(selectedConversationForMenu.id, newConversationTitle.trim());
      await loadConversations();
      
      // Update selected conversation if it was the one renamed
      if (selectedConversation?.id === selectedConversationForMenu.id) {
        setSelectedConversation({
          ...selectedConversation,
          title: newConversationTitle.trim()
        });
      }
      
      setRenameDialogOpen(false);
      setConversationMenuAnchor(null);
      setNewConversationTitle('');
    } catch (err) {
      setError('Failed to rename conversation. Please try again.');
      console.error('Error renaming conversation:', err);
    }
  };

  const handleEditMessage = async (messageId: number, newContent: string) => {
    try {
      await editMessage(messageId, newContent);
      await loadMessages(selectedConversation!.id);
      setEditingMessageId(null);
      setEditingContent('');
      
      // After editing, send the new content to get AI response
      if (selectedConversation && user) {
        setSending(true);
        try {
          await sendMessage(selectedConversation.id, newContent, user.id);
          await loadMessages(selectedConversation.id);
          await loadConversations(); // Refresh conversations to update last message
        } catch (err) {
          console.error('Error getting AI response after edit:', err);
          setError('Message edited successfully, but failed to get AI response.');
        } finally {
          setSending(false);
        }
      }
    } catch (err) {
      setError('Failed to edit message. Please try again.');
      console.error('Error editing message:', err);
    }
  };

  const startEditingMessage = (message: Message) => {
    setEditingMessageId(message.id);
    setEditingContent(message.content);
  };

  const cancelEditingMessage = () => {
    setEditingMessageId(null);
    setEditingContent('');
  };

  const handleConversationMenuOpen = (event: React.MouseEvent<HTMLElement>, conversation: Conversation) => {
    event.stopPropagation();
    setConversationMenuAnchor(event.currentTarget);
    setSelectedConversationForMenu(conversation);
  };

  const handleConversationMenuClose = () => {
    setConversationMenuAnchor(null);
    setSelectedConversationForMenu(null);
  };

  const formatTime = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    const today = new Date();
    const yesterday = new Date(today);
    yesterday.setDate(yesterday.getDate() - 1);

    if (date.toDateString() === today.toDateString()) {
      return 'Today';
    } else if (date.toDateString() === yesterday.toDateString()) {
      return 'Yesterday';
    } else {
      return date.toLocaleDateString();
    }
  };

  const handleLogout = () => {
    clearAuth();
    navigate('/login');
  };

  const handleUserMenuClose = () => {
    setUserMenuAnchor(null);
  };

  const handleNavigateToSearch = () => {
    navigate('/search');
    setUserMenuAnchor(null);
  };

  const getUserInitials = (username: string) => {
    return username
      .split(' ')
      .map(name => name.charAt(0))
      .join('')
      .toUpperCase()
      .slice(0, 2);
  };

  if (!user) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
        <Typography variant="h6" color="text.secondary">
          Please log in to access the chat.
        </Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ height: '100vh', display: 'flex', flexDirection: 'column', background: theme.palette.background.default }}>
      {/* Top Navbar */}
      <AppBar 
        position="static" 
        elevation={0}
        sx={{ 
          background: isDarkMode 
            ? 'linear-gradient(180deg, #1a1a1a 0%, #0f0f0f 100%)' 
            : 'linear-gradient(180deg, #ffffff 0%, #f8fafc 100%)',
          borderBottom: isDarkMode 
            ? '1px solid rgba(255, 255, 255, 0.1)' 
            : '1px solid rgba(0, 0, 0, 0.05)',
        }}
      >
        <Toolbar sx={{ minHeight: '64px !important' }}>
          <Box sx={{ flexGrow: 1, display: 'flex', alignItems: 'center', gap: 2 }}>
            <Box
              sx={{
                width: 32,
                height: 32,
                borderRadius: '10px',
                background: 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              <SmartToy sx={{ fontSize: 18, color: 'white' }} />
            </Box>
            <Typography 
              variant="h6" 
              component="div"
              sx={{ 
                fontWeight: 600,
                color: theme.palette.text.primary,
              }}
            >
              CRM Assistant
            </Typography>
          </Box>
          
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Tooltip title="Search">
              <IconButton
                onClick={handleNavigateToSearch}
                sx={{
                  borderRadius: '12px',
                  backgroundColor: isDarkMode ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.05)',
                  '&:hover': {
                    backgroundColor: isDarkMode ? 'rgba(255, 255, 255, 0.15)' : 'rgba(0, 0, 0, 0.1)',
                  },
                }}
              >
                <SearchIcon />
              </IconButton>
            </Tooltip>
            
            <Tooltip title="Toggle theme">
              <IconButton
                onClick={toggleTheme}
                sx={{
                  borderRadius: '12px',
                  backgroundColor: isDarkMode ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.05)',
                  '&:hover': {
                    backgroundColor: isDarkMode ? 'rgba(255, 255, 255, 0.15)' : 'rgba(0, 0, 0, 0.1)',
                  },
                }}
              >
                {isDarkMode ? <LightMode /> : <DarkMode />}
              </IconButton>
            </Tooltip>
            
            <Tooltip title="User menu">
              <IconButton
                onClick={(e) => setUserMenuAnchor(e.currentTarget)}
                sx={{
                  p: 0,
                  ml: 1,
                }}
              >
                <Avatar
                  sx={{
                    width: 36,
                    height: 36,
                    background: 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)',
                    fontSize: '0.875rem',
                    fontWeight: 600,
                  }}
                >
                  {user ? getUserInitials(user.username) : 'U'}
                </Avatar>
              </IconButton>
            </Tooltip>
          </Box>
        </Toolbar>
      </AppBar>

      {/* Main Chat Content Area */}
      <Box sx={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
        {/* Sidebar - Conversations List */}
        {sidebarVisible && (
          <Box
            sx={{
              width: 320,
              display: 'flex',
              flexDirection: 'column',
              background: isDarkMode 
                ? 'linear-gradient(180deg, #1a1a1a 0%, #0f0f0f 100%)'
                : 'linear-gradient(180deg, #ffffff 0%, #f8fafc 100%)',
            }}
          >
            {/* Header */}
            <Box sx={{ p: 3 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
                <Typography variant="h5" sx={{ fontWeight: 700, color: theme.palette.text.primary }}>
                  Messages
                </Typography>
                <Box sx={{ display: 'flex', gap: 1 }}>
                  <IconButton 
                    onClick={handleNewConversation} 
                    size="small"
                    sx={{
                      borderRadius: '12px',
                      background: 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)',
                      color: 'white',
                      '&:hover': {
                        background: 'linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%)',
                        transform: 'translateY(-1px)',
                      },
                      transition: 'all 0.2s ease',
                    }}
                  >
                    <Add />
                  </IconButton>
                </Box>
              </Box>
              <Box
                sx={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: 1,
                  px: 2,
                  py: 1,
                  borderRadius: '20px',
                  background: 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)',
                  color: 'white',
                  fontSize: '0.875rem',
                  fontWeight: 500,
                }}
              >
                <SmartToy sx={{ fontSize: 16 }} />
                AI Assistant
              </Box>
            </Box>

            {/* Conversations List */}
            <Box sx={{ flex: 1, overflow: 'auto', px: 2 }}>
              {loading && conversations.length === 0 ? (
                <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
                  <CircularProgress size={24} />
                </Box>
              ) : (
                conversations.map((conversation) => (
                  <Box
                    key={conversation.id}
                    onClick={() => handleConversationSelect(conversation)}
                    sx={{
                      p: 2,
                      mb: 1,
                      cursor: 'pointer',
                      borderRadius: '16px',
                      backgroundColor: selectedConversation?.id === conversation.id 
                        ? isDarkMode 
                          ? 'rgba(99, 102, 241, 0.15)'
                          : 'rgba(99, 102, 241, 0.08)'
                        : 'transparent',
                      '&:hover': {
                        backgroundColor: isDarkMode 
                          ? 'rgba(255, 255, 255, 0.05)'
                          : 'rgba(0, 0, 0, 0.02)',
                        transform: 'translateY(-1px)',
                      },
                      transition: 'all 0.2s ease',
                    }}
                  >
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                      <Box
                        sx={{
                          width: 44,
                          height: 44,
                          borderRadius: '12px',
                          background: 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                        }}
                      >
                        <SmartToy sx={{ color: 'white', fontSize: 20 }} />
                      </Box>
                      <Box sx={{ flex: 1, minWidth: 0 }}>
                        <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 0.5 }}>
                          {conversation.title}
                        </Typography>
                        <Typography
                          variant="body2"
                          color="text.secondary"
                          sx={{
                            overflow: 'hidden',
                            textOverflow: 'ellipsis',
                            whiteSpace: 'nowrap',
                            fontSize: '0.8rem',
                          }}
                        >
                          {conversation.last_message?.content || 'Start a conversation...'}
                        </Typography>
                      </Box>
                      <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: 1 }}>
                        <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.75rem' }}>
                          {conversation.last_message?.created_at 
                            ? formatTime(conversation.last_message.created_at)
                            : formatTime(conversation.created_at)
                          }
                        </Typography>
                        <IconButton
                          size="small"
                          onClick={(e) => handleConversationMenuOpen(e, conversation)}
                          sx={{ 
                            opacity: 0.6, 
                            '&:hover': { opacity: 1 },
                            borderRadius: '8px',
                          }}
                        >
                          <MoreVert fontSize="small" />
                        </IconButton>
                      </Box>
                    </Box>
                  </Box>
                ))
              )}
            </Box>
          </Box>
        )}

        {/* Conversation Context Menu */}
        <Menu
          anchorEl={conversationMenuAnchor}
          open={Boolean(conversationMenuAnchor)}
          onClose={handleConversationMenuClose}
          PaperProps={{
            sx: {
              minWidth: 200,
              border: `1px solid ${theme.palette.divider}`,
            }
          }}
        >
          <MenuItem 
            onClick={() => {
              setNewConversationTitle(selectedConversationForMenu?.title || '');
              setRenameDialogOpen(true);
            }}
          >
            <ListItemIcon>
              <DriveFileRenameOutline fontSize="small" />
            </ListItemIcon>
            <ListItemText>Rename</ListItemText>
          </MenuItem>
          <MenuItem 
            onClick={() => setDeleteDialogOpen(true)}
            sx={{ color: theme.palette.error.main }}
          >
            <ListItemIcon>
              <Delete fontSize="small" sx={{ color: theme.palette.error.main }} />
            </ListItemIcon>
            <ListItemText>Delete</ListItemText>
          </MenuItem>
        </Menu>

        {/* Main Chat Area */}
        <Box sx={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
          {selectedConversation ? (
            <>
              {/* Chat Header */}
              <Box
                sx={{
                  p: 3,
                  background: isDarkMode 
                    ? 'linear-gradient(180deg, #1a1a1a 0%, #0f0f0f 100%)'
                    : 'linear-gradient(180deg, #ffffff 0%, #f8fafc 100%)',
                }}
              >
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                    <IconButton
                      onClick={() => setSidebarVisible(!sidebarVisible)}
                      sx={{ 
                        borderRadius: '12px',
                        backgroundColor: isDarkMode ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.05)',
                        '&:hover': {
                          backgroundColor: isDarkMode ? 'rgba(255, 255, 255, 0.15)' : 'rgba(0, 0, 0, 0.1)',
                          transform: 'translateY(-1px)',
                        },
                        transition: 'all 0.2s ease',
                      }}
                    >
                      {sidebarVisible ? <MenuOpen /> : <MenuIcon />}
                    </IconButton>
                    <Box
                      sx={{
                        width: 40,
                        height: 40,
                        borderRadius: '12px',
                        background: 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                      }}
                    >
                      <SmartToy sx={{ color: 'white', fontSize: 20 }} />
                    </Box>
                    <Box>
                      <Typography variant="h6" sx={{ fontWeight: 600 }}>
                        {selectedConversation.title}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Always here to help with your CRM needs
                      </Typography>
                    </Box>
                  </Box>
                </Box>
              </Box>

              {/* Messages Area */}
              <Box
                sx={{
                  flex: 1,
                  overflow: 'auto',
                  p: 3,
                  pb: 0, // Remove bottom padding since input is fixed
                  background: isDarkMode 
                    ? 'linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%)'
                    : 'linear-gradient(135deg, #f8fafc 0%, #ffffff 100%)',
                }}
              >
                {error && (
                  <Box sx={{ p: 2, mb: 3 }}>
                    <Box
                      sx={{
                        p: 3,
                        borderRadius: '16px',
                        backgroundColor: theme.palette.error.light,
                        color: theme.palette.error.contrastText,
                      }}
                    >
                      <Typography variant="body2">{error}</Typography>
                    </Box>
                  </Box>
                )}

                {messages.map((message, index) => {
                  const showDate = index === 0 || 
                    formatDate(message.created_at) !== formatDate(messages[index - 1].created_at);
                  const isEditing = editingMessageId === message.id;
                  
                  return (
                    <React.Fragment key={message.id}>
                      {showDate && (
                        <Box sx={{ display: 'flex', justifyContent: 'center', my: 3 }}>
                          <Box
                            sx={{
                              px: 3,
                              py: 1,
                              borderRadius: '20px',
                              backgroundColor: isDarkMode ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.05)',
                              color: theme.palette.text.secondary,
                              fontSize: '0.875rem',
                              fontWeight: 500,
                            }}
                          >
                            {formatDate(message.created_at)}
                          </Box>
                        </Box>
                      )}
                      
                      <Fade in timeout={300}>
                        <Box
                          sx={{
                            display: 'flex',
                            justifyContent: message.is_agent ? 'flex-start' : 'flex-end',
                            mb: 3,
                            alignItems: 'flex-end',
                            gap: 2,
                          }}
                        >
                          {message.is_agent && (
                            <Box
                              sx={{
                                width: 32,
                                height: 32,
                                borderRadius: '10px',
                                background: 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)',
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                              }}
                            >
                              <SmartToy sx={{ fontSize: 16, color: 'white' }} />
                            </Box>
                          )}
                          
                          <Box
                            sx={{
                              maxWidth: '70%',
                              background: message.is_agent
                                ? isDarkMode 
                                  ? 'rgba(255, 255, 255, 0.05)'
                                  : 'rgba(255, 255, 255, 0.8)'
                                : 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)',
                              color: message.is_agent ? theme.palette.text.primary : 'white',
                              borderRadius: '20px',
                              boxShadow: isDarkMode
                                ? '0 8px 32px rgba(0, 0, 0, 0.3)'
                                : '0 8px 32px rgba(0, 0, 0, 0.1)',
                              position: 'relative',
                              '&:hover .message-actions': {
                                opacity: 1,
                              },
                            }}
                          >
                            {isEditing ? (
                              <Box sx={{ p: 3 }}>
                                <TextField
                                  fullWidth
                                  multiline
                                  value={editingContent}
                                  onChange={(e) => setEditingContent(e.target.value)}
                                  variant="outlined"
                                  size="small"
                                  sx={{
                                    mb: 1,
                                    '& .MuiOutlinedInput-root': {
                                      backgroundColor: 'transparent',
                                      color: message.is_agent ? theme.palette.text.primary : 'white',
                                      borderRadius: '12px',
                                    },
                                  }}
                                />
                                <Box sx={{ display: 'flex', gap: 1, justifyContent: 'flex-end' }}>
                                  <IconButton
                                    size="small"
                                    onClick={() => handleEditMessage(message.id, editingContent)}
                                    sx={{ 
                                      color: message.is_agent ? theme.palette.primary.main : 'white',
                                      borderRadius: '8px',
                                    }}
                                  >
                                    <Check fontSize="small" />
                                  </IconButton>
                                  <IconButton
                                    size="small"
                                    onClick={cancelEditingMessage}
                                    sx={{ 
                                      color: message.is_agent ? theme.palette.text.secondary : 'rgba(255,255,255,0.7)',
                                      borderRadius: '8px',
                                    }}
                                  >
                                    <Close fontSize="small" />
                                  </IconButton>
                                </Box>
                              </Box>
                            ) : (
                              <Box sx={{ p: 3 }}>
                                <Typography
                                  variant="body1"
                                  sx={{
                                    whiteSpace: 'pre-wrap',
                                    wordBreak: 'break-word',
                                    lineHeight: 1.6,
                                  }}
                                >
                                  {message.content}
                                </Typography>
                                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mt: 2 }}>
                                  <Typography
                                    variant="caption"
                                    sx={{
                                      opacity: 0.7,
                                      fontSize: '0.75rem',
                                    }}
                                  >
                                    {formatTime(message.created_at)}
                                    {message.is_edited && ' (edited)'}
                                  </Typography>
                                  {!message.is_agent && (
                                    <Box 
                                      className="message-actions"
                                      sx={{ 
                                        opacity: 0, 
                                        transition: 'opacity 0.2s',
                                        display: 'flex',
                                        gap: 0.5
                                      }}
                                    >
                                      <IconButton
                                        size="small"
                                        onClick={() => startEditingMessage(message)}
                                        sx={{ 
                                          color: 'rgba(255,255,255,0.7)', 
                                          '&:hover': { color: 'white' },
                                          borderRadius: '6px',
                                        }}
                                      >
                                        <Edit fontSize="small" />
                                      </IconButton>
                                    </Box>
                                  )}
                                </Box>
                              </Box>
                            )}
                          </Box>

                          {!message.is_agent && (
                            <Box
                              sx={{
                                width: 32,
                                height: 32,
                                borderRadius: '10px',
                                backgroundColor: isDarkMode ? 'rgba(255, 255, 255, 0.1)' : theme.palette.primary.main,
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                              }}
                            >
                              <Person sx={{ fontSize: 16, color: isDarkMode ? 'white' : 'white' }} />
                            </Box>
                          )}
                        </Box>
                      </Fade>
                    </React.Fragment>
                  );
                })}
                
                {sending && (
                  <Box sx={{ display: 'flex', justifyContent: 'flex-start', mb: 3, alignItems: 'flex-end', gap: 2 }}>
                    <Box
                      sx={{
                        width: 32,
                        height: 32,
                        borderRadius: '10px',
                        background: 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                      }}
                    >
                      <SmartToy sx={{ fontSize: 16, color: 'white' }} />
                    </Box>
                    <Box
                      sx={{
                        p: 3,
                        borderRadius: '20px',
                        background: isDarkMode 
                          ? 'rgba(255, 255, 255, 0.05)'
                          : 'rgba(255, 255, 255, 0.8)',
                        boxShadow: isDarkMode
                          ? '0 8px 32px rgba(0, 0, 0, 0.3)'
                          : '0 8px 32px rgba(0, 0, 0, 0.1)',
                      }}
                    >
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                        <CircularProgress size={16} />
                        <Typography variant="body2" color="text.secondary">
                          AI is thinking...
                        </Typography>
                      </Box>
                    </Box>
                  </Box>
                )}
                
                {/* Spacer for fixed input */}
                <Box sx={{ height: 120 }} />
                <div ref={messagesEndRef} />
              </Box>

              {/* Fixed Message Input */}
              <Box
                sx={{
                  position: 'fixed',
                  bottom: 0,
                  left: sidebarVisible ? 320 : 0,
                  right: 0,
                  p: 3,
                  background: isDarkMode 
                    ? 'linear-gradient(180deg, transparent 0%, #1a1a1a 50%, #0f0f0f 100%)'
                    : 'linear-gradient(180deg, transparent 0%, #ffffff 50%, #f8fafc 100%)',
                  zIndex: 1000,
                  display: 'flex',
                  justifyContent: 'center',
                }}
              >
                <Box sx={{ width: '100%', maxWidth: 800 }}>
                  <Box component="form" onSubmit={handleSendMessage}>
                    <TextField
                      fullWidth
                      multiline
                      maxRows={4}
                      value={newMessage}
                      onChange={(e) => setNewMessage(e.target.value)}
                      placeholder="Message CRM Assistant"
                      variant="outlined"
                      disabled={sending}
                      InputProps={{
                        endAdornment: (
                          <InputAdornment position="end">
                            <IconButton
                              type="submit"
                              disabled={!newMessage.trim() || sending}
                              sx={{
                                borderRadius: '12px',
                                background: newMessage.trim() 
                                  ? 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)'
                                  : 'transparent',
                                color: newMessage.trim() ? 'white' : theme.palette.text.disabled,
                                '&:hover': {
                                  background: newMessage.trim()
                                    ? 'linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%)'
                                    : isDarkMode ? 'rgba(255, 255, 255, 0.05)' : 'rgba(0, 0, 0, 0.02)',
                                  transform: newMessage.trim() ? 'translateY(-1px)' : 'none',
                                },
                                '&:disabled': {
                                  background: 'transparent',
                                  color: theme.palette.text.disabled,
                                },
                                transition: 'all 0.2s ease',
                              }}
                            >
                              <Send />
                            </IconButton>
                          </InputAdornment>
                        ),
                        sx: {
                          borderRadius: '20px',
                          backgroundColor: isDarkMode ? 'rgba(255, 255, 255, 0.05)' : 'rgba(255, 255, 255, 0.8)',
                          '&:hover': {
                            borderColor: isDarkMode 
                              ? 'rgba(99, 102, 241, 0.5)'
                              : 'rgba(99, 102, 241, 0.3)',
                          },
                          '&.Mui-focused': {
                            borderColor: theme.palette.primary.main,
                            boxShadow: `0 0 0 3px ${theme.palette.primary.main}20`,
                          },
                          '& .MuiOutlinedInput-notchedOutline': {
                            border: 'none',
                          },
                        },
                      }}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' && !e.shiftKey) {
                          e.preventDefault();
                          handleSendMessage(e);
                        }
                      }}
                    />
                  </Box>
                </Box>
              </Box>
            </>
          ) : (
            <Box
              sx={{
                flex: 1,
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'center',
                alignItems: 'center',
                gap: 3,
                position: 'relative',
                background: isDarkMode 
                  ? 'linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%)'
                  : 'linear-gradient(135deg, #f8fafc 0%, #ffffff 100%)',
              }}
            >
              {/* Toggle button for when sidebar is hidden and no conversation selected */}
              {!sidebarVisible && (
                <Box sx={{ position: 'absolute', top: 20, left: 20 }}>
                  <Tooltip title="Show sidebar">
                    <IconButton
                      onClick={() => setSidebarVisible(true)}
                      sx={{
                        background: 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)',
                        color: 'white',
                        '&:hover': {
                          background: 'linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%)',
                        },
                      }}
                    >
                      <MenuIcon />
                    </IconButton>
                  </Tooltip>
                </Box>
              )}
              
              {/* Centered Welcome Content */}
              <Box sx={{ textAlign: 'center', maxWidth: 600, px: 3 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', mb: 3 }}>
                  <Avatar
                    sx={{
                      width: 48,
                      height: 48,
                      background: 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)',
                      mr: 2,
                    }}
                  >
                    <SmartToy sx={{ fontSize: 24 }} />
                  </Avatar>
                  <Typography 
                    variant="h4" 
                    sx={{ 
                      fontWeight: 600,
                      background: 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)',
                      backgroundClip: 'text',
                      WebkitBackgroundClip: 'text',
                      WebkitTextFillColor: 'transparent',
                    }}
                  >
                    Hi, I'm CRM Assistant.
                  </Typography>
                </Box>
                
                <Typography 
                  variant="h6" 
                  color="text.secondary" 
                  sx={{ 
                    mb: 4,
                    fontWeight: 400,
                    lineHeight: 1.6,
                  }}
                >
                  How can I help you today?
                </Typography>

                {/* Quick Action Buttons */}
                <Box sx={{ display: 'flex', gap: 1, justifyContent: 'center', flexWrap: 'wrap' }}>
                  <Chip
                    icon={<SmartToy />}
                    label="Find Candidates"
                    variant="outlined"
                    clickable
                    onClick={() => setNewMessage("Help me find candidates for a software developer position")}
                    sx={{
                      '&:hover': {
                        backgroundColor: theme.palette.primary.main,
                        color: 'white',
                        '& .MuiChip-icon': {
                          color: 'white',
                        },
                      },
                    }}
                  />
                  <Chip
                    icon={<Person />}
                    label="Search Tips"
                    variant="outlined"
                    clickable
                    onClick={() => setNewMessage("How can I search for candidates effectively?")}
                    sx={{
                      '&:hover': {
                        backgroundColor: theme.palette.primary.main,
                        color: 'white',
                        '& .MuiChip-icon': {
                          color: 'white',
                        },
                      },
                    }}
                  />
                </Box>
              </Box>

              {/* Fixed Message Input at Bottom */}
              <Box
                sx={{
                  position: 'absolute',
                  bottom: 0,
                  left: 0,
                  right: 0,
                  p: 3,
                  background: isDarkMode 
                    ? 'linear-gradient(180deg, transparent 0%, #1a1a1a 50%, #0f0f0f 100%)'
                    : 'linear-gradient(180deg, transparent 0%, #ffffff 50%, #f8fafc 100%)',
                  display: 'flex',
                  justifyContent: 'center',
                }}
              >
                <Box sx={{ maxWidth: 600, width: '100%' }}>
                  <Box component="form" onSubmit={handleSendMessage}>
                    <TextField
                      fullWidth
                      multiline
                      maxRows={4}
                      value={newMessage}
                      onChange={(e) => setNewMessage(e.target.value)}
                      placeholder="Message CRM Assistant"
                      variant="outlined"
                      disabled={sending}
                      InputProps={{
                        endAdornment: (
                          <InputAdornment position="end">
                            <IconButton
                              type="submit"
                              disabled={!newMessage.trim() || sending}
                              sx={{
                                borderRadius: '12px',
                                background: newMessage.trim() 
                                  ? 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)'
                                  : 'transparent',
                                color: newMessage.trim() ? 'white' : theme.palette.text.disabled,
                                '&:hover': {
                                  background: newMessage.trim()
                                    ? 'linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%)'
                                    : isDarkMode ? 'rgba(255, 255, 255, 0.05)' : 'rgba(0, 0, 0, 0.02)',
                                  transform: newMessage.trim() ? 'translateY(-1px)' : 'none',
                                },
                                '&:disabled': {
                                  background: 'transparent',
                                  color: theme.palette.text.disabled,
                                },
                                transition: 'all 0.2s ease',
                              }}
                            >
                              <Send />
                            </IconButton>
                          </InputAdornment>
                        ),
                        sx: {
                          borderRadius: '20px',
                          backgroundColor: isDarkMode ? 'rgba(255, 255, 255, 0.05)' : 'rgba(255, 255, 255, 0.8)',
                          '&:hover': {
                            borderColor: isDarkMode 
                              ? 'rgba(99, 102, 241, 0.5)'
                              : 'rgba(99, 102, 241, 0.3)',
                          },
                          '&.Mui-focused': {
                            borderColor: theme.palette.primary.main,
                            boxShadow: `0 0 0 3px ${theme.palette.primary.main}20`,
                          },
                          '& .MuiOutlinedInput-notchedOutline': {
                            border: 'none',
                          },
                        },
                      }}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' && !e.shiftKey) {
                          e.preventDefault();
                          handleSendMessage(e);
                        }
                      }}
                    />
                  </Box>
                </Box>
              </Box>
            </Box>
          )}
        </Box>
      </Box>

      {/* Rename Dialog */}
      <Dialog open={renameDialogOpen} onClose={() => setRenameDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Rename Conversation</DialogTitle>
        <DialogContent>
          <TextField
            autoFocus
            margin="dense"
            label="Conversation Title"
            fullWidth
            variant="outlined"
            value={newConversationTitle}
            onChange={(e) => setNewConversationTitle(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                handleRenameConversation();
              }
            }}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setRenameDialogOpen(false)}>Cancel</Button>
          <Button onClick={handleRenameConversation} variant="contained">
            Rename
          </Button>
        </DialogActions>
      </Dialog>

      {/* Delete Dialog */}
      <Dialog open={deleteDialogOpen} onClose={() => setDeleteDialogOpen(false)}>
        <DialogTitle>Delete Conversation</DialogTitle>
        <DialogContent>
          <Typography>
            Are you sure you want to delete "{selectedConversationForMenu?.title}"? 
            This action cannot be undone.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteDialogOpen(false)}>Cancel</Button>
          <Button onClick={handleDeleteConversation} color="error" variant="contained">
            Delete
          </Button>
        </DialogActions>
      </Dialog>

      {/* User Menu */}
      <Menu
        anchorEl={userMenuAnchor}
        open={Boolean(userMenuAnchor)}
        onClose={handleUserMenuClose}
        PaperProps={{
          sx: {
            minWidth: 200,
            mt: 1,
            borderRadius: '12px',
            border: `1px solid ${theme.palette.divider}`,
            background: isDarkMode 
              ? 'linear-gradient(180deg, #1a1a1a 0%, #0f0f0f 100%)'
              : 'linear-gradient(180deg, #ffffff 0%, #f8fafc 100%)',
          }
        }}
        transformOrigin={{ horizontal: 'right', vertical: 'top' }}
        anchorOrigin={{ horizontal: 'right', vertical: 'bottom' }}
      >
        <Box sx={{ px: 2, py: 1.5, borderBottom: `1px solid ${theme.palette.divider}` }}>
          <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
            {user?.username}
          </Typography>
          <Typography variant="caption" color="text.secondary">
            {user?.email || 'user@example.com'}
          </Typography>
        </Box>
        
        <MenuItem onClick={handleNavigateToSearch}>
          <ListItemIcon>
            <SearchIcon fontSize="small" />
          </ListItemIcon>
          <ListItemText>Search Candidates</ListItemText>
        </MenuItem>
        
        <MenuItem onClick={() => navigate('/dashboard')}>
          <ListItemIcon>
            <AccountCircle fontSize="small" />
          </ListItemIcon>
          <ListItemText>Dashboard</ListItemText>
        </MenuItem>
        
        <MenuItem onClick={handleUserMenuClose}>
          <ListItemIcon>
            <Settings fontSize="small" />
          </ListItemIcon>
          <ListItemText>Settings</ListItemText>
        </MenuItem>
        
        <Divider />
        
        <MenuItem onClick={handleLogout} sx={{ color: theme.palette.error.main }}>
          <ListItemIcon>
            <Logout fontSize="small" sx={{ color: theme.palette.error.main }} />
          </ListItemIcon>
          <ListItemText>Logout</ListItemText>
        </MenuItem>
      </Menu>
    </Box>
  );
};

export default Chat; 