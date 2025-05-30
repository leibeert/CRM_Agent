import React, { useState, useEffect } from 'react';
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
  IconButton,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Slider,
  Divider
} from '@mui/material';
import DeleteIcon from '@mui/icons-material/Delete';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import useAuthStore from '../store/authStore';
import { searchCandidates, saveSearch, getSavedSearches, deleteSavedSearch } from '../utils/api';

interface Skill {
  skill_name: string;
  min_level?: number;
  min_duration?: number;
}

interface Experience {
  title?: string;
  company?: string;
  min_years?: number;
}

interface Education {
  degree_type?: string;
  field_of_study?: string;
  school?: string;
}

interface SearchQuery {
  keywords?: string;
  skills?: Skill[];
  experience?: Experience;
  education?: Education;
  sort_by?: string;
  sort_order?: string;
  page?: number;
  page_size?: number;
}

interface Candidate {
  id: number;
  first_name: string;
  last_name: string;
  email: string;
  skills: Array<{name: string, level: number, duration?: number}>;
  experience: Array<{title: string, company: string, description?: string}>;
  education: Array<{degree_name: string, school_id: number}>;
  match_score: number;
}

interface SavedSearch {
  id: number;
  name: string;
  description: string;
  filters: SearchQuery;
  created_at: string;
}

interface SearchResults {
  candidates: Candidate[];
  total: number;
  page: number;
  page_size: number;
}

const Search: React.FC = () => {
  // Basic search states
  const [keywords, setKeywords] = useState<string>('');
  const [sortBy, setSortBy] = useState<string>('match_score');
  const [sortOrder, setSortOrder] = useState<string>('desc');
  
  // Skills search states
  const [skills, setSkills] = useState<Skill[]>([]);
  const [newSkillName, setNewSkillName] = useState<string>('');
  const [newSkillLevel, setNewSkillLevel] = useState<number>(1);
  const [newSkillDuration, setNewSkillDuration] = useState<number>(0);
  
  // Experience search states
  const [experienceTitle, setExperienceTitle] = useState<string>('');
  const [experienceCompany, setExperienceCompany] = useState<string>('');
  const [experienceMinYears, setExperienceMinYears] = useState<number>(0);
  
  // Education search states
  const [educationDegree, setEducationDegree] = useState<string>('');
  const [educationField, setEducationField] = useState<string>('');
  const [educationSchool, setEducationSchool] = useState<string>('');
  
  // Results and UI states
  const [candidates, setCandidates] = useState<Candidate[]>([]);
  const [savedSearches, setSavedSearches] = useState<SavedSearch[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>('');
  const [total, setTotal] = useState<number>(0);
  
  const user = useAuthStore((state) => state.user);

  // Load saved searches on component mount
  useEffect(() => {
    if (user?.id) {
      loadSavedSearches();
    }
  }, [user?.id]);

  const buildSearchQuery = (): SearchQuery => {
    const query: SearchQuery = {
      sort_by: sortBy,
      sort_order: sortOrder,
      page: 1,
      page_size: 20
    };

    if (keywords.trim()) {
      query.keywords = keywords.trim();
    }

    if (skills.length > 0) {
      query.skills = skills;
    }

    const hasExperience = experienceTitle || experienceCompany || experienceMinYears > 0;
    if (hasExperience) {
      query.experience = {};
      if (experienceTitle) query.experience.title = experienceTitle;
      if (experienceCompany) query.experience.company = experienceCompany;
      if (experienceMinYears > 0) query.experience.min_years = experienceMinYears;
    }

    const hasEducation = educationDegree || educationField || educationSchool;
    if (hasEducation) {
      query.education = {};
      if (educationDegree) query.education.degree_type = educationDegree;
      if (educationField) query.education.field_of_study = educationField;
      if (educationSchool) query.education.school = educationSchool;
    }

    return query;
  };

  const handleSearch = async (): Promise<void> => {
    try {
      setLoading(true);
      setError('');
      
      const searchQuery = buildSearchQuery();
      console.log('Search Query:', searchQuery);
      
      const results: SearchResults = await searchCandidates(searchQuery);
      setCandidates(results.candidates || []);
      setTotal(results.total || 0);
    } catch (err: any) {
      setError('Failed to search candidates. Please try again.');
      console.error('Error searching candidates:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleAddSkill = (): void => {
    if (newSkillName.trim() && !skills.some(s => s.skill_name === newSkillName.trim())) {
      const newSkill: Skill = {
        skill_name: newSkillName.trim(),
        min_level: newSkillLevel > 1 ? newSkillLevel : undefined,
        min_duration: newSkillDuration > 0 ? newSkillDuration : undefined
      };
      setSkills([...skills, newSkill]);
      setNewSkillName('');
      setNewSkillLevel(1);
      setNewSkillDuration(0);
    }
  };

  const handleRemoveSkill = (skillToRemove: string): void => {
    setSkills(skills.filter(skill => skill.skill_name !== skillToRemove));
  };

  const clearSearch = (): void => {
    setKeywords('');
    setSkills([]);
    setExperienceTitle('');
    setExperienceCompany('');
    setExperienceMinYears(0);
    setEducationDegree('');
    setEducationField('');
    setEducationSchool('');
    setCandidates([]);
    setTotal(0);
  };

  const handleSaveSearch = async (): Promise<void> => {
    if (!keywords && skills.length === 0) {
      setError('Please enter search criteria before saving.');
      return;
    }

    if (!user?.id) {
      setError('Please log in to save searches.');
      return;
    }

    try {
      setLoading(true);
      setError('');
      
      const searchData = {
        user_id: user.id,
        name: `Search ${new Date().toLocaleString()}`,
        description: `Keywords: ${keywords}, Skills: ${skills.map(s => s.skill_name).join(', ')}`,
        filters: buildSearchQuery(),
        sort_by: sortBy,
        sort_order: sortOrder
      };
      
      await saveSearch(searchData);
      await loadSavedSearches();
    } catch (err: any) {
      setError('Failed to save search. Please try again.');
      console.error('Error saving search:', err);
    } finally {
      setLoading(false);
    }
  };

  const loadSavedSearches = async (): Promise<void> => {
    if (!user?.id) return;
    
    try {
      const searches: SavedSearch[] = await getSavedSearches(user.id);
      setSavedSearches(searches);
    } catch (err: any) {
      console.error('Error loading saved searches:', err);
      // Don't show error for loading saved searches as it's not critical
    }
  };

  const handleDeleteSearch = async (searchId: number): Promise<void> => {
    if (!user?.id) return;
    
    try {
      setLoading(true);
      await deleteSavedSearch(searchId, user.id);
      await loadSavedSearches();
    } catch (err: any) {
      setError('Failed to delete saved search. Please try again.');
      console.error('Error deleting saved search:', err);
    } finally {
      setLoading(false);
    }
  };

  if (!user) {
    return (
      <Container>
        <Alert severity="error">Please log in to access the search.</Alert>
      </Container>
    );
  }

  return (
    <Container maxWidth="xl" sx={{ mt: 4 }}>
      <Typography variant="h4" gutterBottom align="center">
        🔍 Intelligent Candidate Search
      </Typography>
      
      <Grid container spacing={3}>
        {/* Search Form */}
        <Grid item xs={12} lg={4}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom color="primary">
              Search Criteria
            </Typography>
            
            {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
            
            {/* Keywords Search */}
            <Box sx={{ mb: 3 }}>
              <TextField
                fullWidth
                label="🔎 Smart Keywords"
                placeholder="e.g., python, react, developer, consulting..."
                value={keywords}
                onChange={(e) => setKeywords(e.target.value)}
                helperText="Searches in skills, experience, job titles, and more!"
                margin="normal"
              />
            </Box>

            {/* Skills Search */}
            <Accordion defaultExpanded>
              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                <Typography variant="subtitle1">⚡ Skills & Experience</Typography>
              </AccordionSummary>
              <AccordionDetails>
                <Box sx={{ mb: 2 }}>
                  <TextField
                    fullWidth
                    label="Skill Name"
                    placeholder="e.g., Python, JavaScript, MySQL..."
                    value={newSkillName}
                    onChange={(e) => setNewSkillName(e.target.value)}
                    size="small"
                  />
                  
                  <Box sx={{ mt: 2 }}>
                    <Typography gutterBottom>Minimum Level (1-5)</Typography>
                    <Slider
                      value={newSkillLevel}
                      onChange={(_, value) => setNewSkillLevel(value as number)}
                      min={1}
                      max={5}
                      marks
                      step={1}
                      valueLabelDisplay="auto"
                    />
                  </Box>
                  
                  <Box sx={{ mt: 2 }}>
                    <Typography gutterBottom>Minimum Duration (months)</Typography>
                    <Slider
                      value={newSkillDuration}
                      onChange={(_, value) => setNewSkillDuration(value as number)}
                      min={0}
                      max={120}
                      step={6}
                      valueLabelDisplay="auto"
                    />
                  </Box>
                  
                  <Button
                    fullWidth
                    variant="outlined"
                    onClick={handleAddSkill}
                    sx={{ mt: 2 }}
                    disabled={!newSkillName.trim()}
                  >
                    Add Skill Requirement
                  </Button>
                </Box>
                
                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                  {skills.map((skill) => (
                    <Chip
                      key={skill.skill_name}
                      label={`${skill.skill_name}${skill.min_level ? ` (L${skill.min_level}+)` : ''}${skill.min_duration ? ` (${skill.min_duration}m+)` : ''}`}
                      onDelete={() => handleRemoveSkill(skill.skill_name)}
                      color="primary"
                      variant="outlined"
                    />
                  ))}
                </Box>
              </AccordionDetails>
            </Accordion>

            {/* Experience Search */}
            <Accordion>
              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                <Typography variant="subtitle1">💼 Work Experience</Typography>
              </AccordionSummary>
              <AccordionDetails>
                <TextField
                  fullWidth
                  label="Job Title"
                  placeholder="e.g., Developer, Manager, Consultant..."
                  value={experienceTitle}
                  onChange={(e) => setExperienceTitle(e.target.value)}
                  margin="normal"
                  size="small"
                />
                
                <TextField
                  fullWidth
                  label="Company"
                  placeholder="e.g., Google, Microsoft, Startup..."
                  value={experienceCompany}
                  onChange={(e) => setExperienceCompany(e.target.value)}
                  margin="normal"
                  size="small"
                />
                
                <Box sx={{ mt: 2 }}>
                  <Typography gutterBottom>Minimum Years of Experience</Typography>
                  <Slider
                    value={experienceMinYears}
                    onChange={(_, value) => setExperienceMinYears(value as number)}
                    min={0}
                    max={20}
                    step={1}
                    marks
                    valueLabelDisplay="auto"
                  />
                </Box>
              </AccordionDetails>
            </Accordion>

            {/* Education Search */}
            <Accordion>
              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                <Typography variant="subtitle1">🎓 Education</Typography>
              </AccordionSummary>
              <AccordionDetails>
                <TextField
                  fullWidth
                  label="Degree Type"
                  placeholder="e.g., Bachelor, Master, PhD..."
                  value={educationDegree}
                  onChange={(e) => setEducationDegree(e.target.value)}
                  margin="normal"
                  size="small"
                />
                
                <TextField
                  fullWidth
                  label="Field of Study"
                  placeholder="e.g., Computer Science, Engineering..."
                  value={educationField}
                  onChange={(e) => setEducationField(e.target.value)}
                  margin="normal"
                  size="small"
                />
              </AccordionDetails>
            </Accordion>

            {/* Sort Options */}
            <Box sx={{ mt: 3 }}>
              <Typography variant="subtitle2" gutterBottom>📊 Sort Results By</Typography>
              <Grid container spacing={2}>
                <Grid item xs={8}>
                  <FormControl fullWidth size="small">
                    <Select
                      value={sortBy}
                      onChange={(e) => setSortBy(e.target.value)}
                    >
                      <MenuItem value="match_score">🎯 Best Match</MenuItem>
                      <MenuItem value="skills">⚡ Most Skills</MenuItem>
                      <MenuItem value="experience">💼 Most Experience</MenuItem>
                      <MenuItem value="created_at">📅 Recently Added</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>
                <Grid item xs={4}>
                  <FormControl fullWidth size="small">
                    <Select
                      value={sortOrder}
                      onChange={(e) => setSortOrder(e.target.value)}
                    >
                      <MenuItem value="desc">↓ High to Low</MenuItem>
                      <MenuItem value="asc">↑ Low to High</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>
              </Grid>
            </Box>

            <Divider sx={{ my: 3 }} />

            {/* Action Buttons */}
            <Grid container spacing={2}>
              <Grid item xs={6}>
                <Button
                  fullWidth
                  variant="contained"
                  onClick={handleSearch}
                  disabled={loading}
                  size="large"
                >
                  {loading ? <CircularProgress size={24} /> : '🔍 Search'}
                </Button>
              </Grid>
              <Grid item xs={6}>
                <Button
                  fullWidth
                  variant="outlined"
                  onClick={clearSearch}
                  disabled={loading}
                  size="large"
                >
                  🗑️ Clear
                </Button>
              </Grid>
            </Grid>
          </Paper>
        </Grid>

        {/* Search Results */}
        <Grid item xs={12} lg={8}>
          <Paper sx={{ p: 3 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
              <Typography variant="h6">
                Search Results
              </Typography>
              {total > 0 && (
                <Chip 
                  label={`${total} candidates found`} 
                  color="success" 
                  variant="outlined"
                />
              )}
            </Box>
            
            {loading ? (
              <Box sx={{ display: 'flex', justifyContent: 'center', p: 5 }}>
                <CircularProgress size={50} />
              </Box>
            ) : candidates.length > 0 ? (
              <List>
                {candidates.map((candidate, index) => (
                  <Box key={candidate.id}>
                    <ListItem alignItems="flex-start" sx={{ px: 0 }}>
                      <ListItemText
                        primary={
                          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                            <Typography variant="h6" component="span">
                              {candidate.first_name} {candidate.last_name}
                            </Typography>
                            <Chip 
                              label={`${candidate.match_score.toFixed(1)}% match`}
                              color={candidate.match_score >= 80 ? 'success' : candidate.match_score >= 60 ? 'warning' : 'default'}
                              size="small"
                            />
                          </Box>
                        }
                        secondary={
                          <Box sx={{ mt: 1 }}>
                            <Typography component="span" variant="body2" color="text.primary">
                              📧 {candidate.email || 'No email provided'}
                            </Typography>
                            
                            {candidate.skills.length > 0 && (
                              <Box sx={{ mt: 1 }}>
                                <Typography variant="body2" color="text.secondary">
                                  ⚡ <strong>Skills:</strong>
                                </Typography>
                                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, mt: 0.5 }}>
                                  {candidate.skills.slice(0, 8).map((skill, i) => (
                                    <Chip
                                      key={i}
                                      label={`${skill.name} (L${skill.level})`}
                                      size="small"
                                      variant="outlined"
                                    />
                                  ))}
                                  {candidate.skills.length > 8 && (
                                    <Chip label={`+${candidate.skills.length - 8} more`} size="small" />
                                  )}
                                </Box>
                              </Box>
                            )}
                            
                            {candidate.experience.length > 0 && (
                              <Box sx={{ mt: 1 }}>
                                <Typography variant="body2" color="text.secondary">
                                  💼 <strong>Experience:</strong>
                                </Typography>
                                {candidate.experience.slice(0, 2).map((exp, i) => (
                                  <Typography key={i} variant="body2" sx={{ ml: 2 }}>
                                    • {exp.title} at {exp.company}
                                  </Typography>
                                ))}
                                {candidate.experience.length > 2 && (
                                  <Typography variant="body2" sx={{ ml: 2, fontStyle: 'italic' }}>
                                    ... and {candidate.experience.length - 2} more positions
                                  </Typography>
                                )}
                              </Box>
                            )}
                          </Box>
                        }
                      />
                    </ListItem>
                    {index < candidates.length - 1 && <Divider />}
                  </Box>
                ))}
              </List>
            ) : (
              <Box sx={{ textAlign: 'center', py: 5 }}>
                <Typography variant="h6" color="text.secondary" gutterBottom>
                  No candidates found
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Try adjusting your search criteria or using different keywords
                </Typography>
                <Box sx={{ mt: 2 }}>
                  <Typography variant="body2" color="text.secondary">
                    💡 <strong>Tips:</strong> Search for skills like "python", "react", "mysql" or job titles like "developer", "manager"
                  </Typography>
                </Box>
              </Box>
            )}
          </Paper>
        </Grid>
      </Grid>
    </Container>
  );
};

export default Search; 