import React, { useState } from 'react';
import {
  Container,
  Paper,
  Typography,
  TextField,
  Button,
  Box,
  Grid,
  Chip,
  CircularProgress,
  Alert,
  List,
  ListItem,
  ListItemText,
  IconButton
} from '@mui/material';
import DeleteIcon from '@mui/icons-material/Delete';
import useAuthStore from '../store/authStore';
import { searchCandidates, saveSearch, getSavedSearches, deleteSavedSearch } from '../utils/api';

interface Candidate {
  id: number;
  first_name: string;
  last_name: string;
  email: string;
  skills: string[];
  experience: string;
  education: string;
  match_score: number;
}

interface SavedSearch {
  id: number;
  name: string;
  description: string;
  filters: any;
  created_at: string;
}

const Search: React.FC = () => {
  const [keywords, setKeywords] = useState('');
  const [skills, setSkills] = useState<string[]>([]);
  const [newSkill, setNewSkill] = useState('');
  const [candidates, setCandidates] = useState<Candidate[]>([]);
  const [savedSearches, setSavedSearches] = useState<SavedSearch[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const user = useAuthStore((state) => state.user);

  const handleSearch = async () => {
    try {
      setLoading(true);
      setError('');
      const results = await searchCandidates({
        keywords,
        skills: skills.map(skill => ({ name: skill })),
      });
      setCandidates(results);
    } catch (err) {
      setError('Failed to search candidates. Please try again.');
      console.error('Error searching candidates:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleSaveSearch = async () => {
    if (!keywords && skills.length === 0) {
      setError('Please enter search criteria before saving.');
      return;
    }

    try {
      setLoading(true);
      setError('');
      await saveSearch({
        user_id: user!.id,
        name: `Search ${new Date().toLocaleString()}`,
        description: `Keywords: ${keywords}, Skills: ${skills.join(', ')}`,
        filters: {
          keywords,
          skills: skills.map(skill => ({ name: skill })),
        }
      });
      await loadSavedSearches();
    } catch (err) {
      setError('Failed to save search. Please try again.');
      console.error('Error saving search:', err);
    } finally {
      setLoading(false);
    }
  };

  const loadSavedSearches = async () => {
    try {
      setLoading(true);
      setError('');
      const searches = await getSavedSearches(user!.id);
      setSavedSearches(searches);
    } catch (err) {
      setError('Failed to load saved searches. Please try again.');
      console.error('Error loading saved searches:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteSearch = async (searchId: number) => {
    try {
      setLoading(true);
      setError('');
      await deleteSavedSearch(searchId);
      await loadSavedSearches();
    } catch (err) {
      setError('Failed to delete saved search. Please try again.');
      console.error('Error deleting saved search:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleAddSkill = () => {
    if (newSkill.trim() && !skills.includes(newSkill.trim())) {
      setSkills([...skills, newSkill.trim()]);
      setNewSkill('');
    }
  };

  const handleRemoveSkill = (skillToRemove: string) => {
    setSkills(skills.filter(skill => skill !== skillToRemove));
  };

  if (!user) {
    return (
      <Container>
        <Alert severity="error">Please log in to access the search.</Alert>
      </Container>
    );
  }

  return (
    <Container maxWidth="lg" sx={{ mt: 4 }}>
      <Grid container spacing={3}>
        {/* Search Form */}
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Search Candidates
            </Typography>
            {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
            <Box sx={{ mb: 2 }}>
              <TextField
                fullWidth
                label="Keywords"
                value={keywords}
                onChange={(e) => setKeywords(e.target.value)}
                margin="normal"
              />
            </Box>
            <Box sx={{ mb: 2 }}>
              <TextField
                fullWidth
                label="Add Skill"
                value={newSkill}
                onChange={(e) => setNewSkill(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleAddSkill()}
                margin="normal"
              />
              <Box sx={{ mt: 1, display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                {skills.map((skill) => (
                  <Chip
                    key={skill}
                    label={skill}
                    onDelete={() => handleRemoveSkill(skill)}
                  />
                ))}
              </Box>
            </Box>
            <Button
              fullWidth
              variant="contained"
              onClick={handleSearch}
              disabled={loading}
              sx={{ mb: 2 }}
            >
              Search
            </Button>
            <Button
              fullWidth
              variant="outlined"
              onClick={handleSaveSearch}
              disabled={loading}
            >
              Save Search
            </Button>
          </Paper>

          {/* Saved Searches */}
          <Paper sx={{ p: 2, mt: 2 }}>
            <Typography variant="h6" gutterBottom>
              Saved Searches
            </Typography>
            <List>
              {savedSearches.map((search) => (
                <ListItem
                  key={search.id}
                  secondaryAction={
                    <IconButton
                      edge="end"
                      aria-label="delete"
                      onClick={() => handleDeleteSearch(search.id)}
                    >
                      <DeleteIcon />
                    </IconButton>
                  }
                >
                  <ListItemText
                    primary={search.name}
                    secondary={search.description}
                  />
                </ListItem>
              ))}
            </List>
          </Paper>
        </Grid>

        {/* Search Results */}
        <Grid item xs={12} md={8}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Search Results
            </Typography>
            {loading ? (
              <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
                <CircularProgress />
              </Box>
            ) : candidates.length > 0 ? (
              <List>
                {candidates.map((candidate) => (
                  <ListItem key={candidate.id}>
                    <ListItemText
                      primary={`${candidate.first_name} ${candidate.last_name}`}
                      secondary={
                        <>
                          <Typography component="span" variant="body2" color="text.primary">
                            {candidate.email}
                          </Typography>
                          <br />
                          Skills: {candidate.skills.join(', ')}
                          <br />
                          Experience: {candidate.experience}
                          <br />
                          Education: {candidate.education}
                          <br />
                          Match Score: {(candidate.match_score * 100).toFixed(1)}%
                        </>
                      }
                    />
                  </ListItem>
                ))}
              </List>
            ) : (
              <Typography color="text.secondary" align="center">
                No candidates found. Try adjusting your search criteria.
              </Typography>
            )}
          </Paper>
        </Grid>
      </Grid>
    </Container>
  );
};

export default Search; 