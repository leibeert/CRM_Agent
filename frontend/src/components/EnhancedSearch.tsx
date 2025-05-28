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
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Card,
  CardContent,
  LinearProgress,
  Divider,
  Tab,
  Tabs,
  Badge,
  Tooltip,
  IconButton
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  Psychology as PsychologyIcon,
  TrendingUp as TrendingUpIcon,
  School as SchoolIcon,
  Work as WorkIcon,
  Star as StarIcon,
  Lightbulb as LightbulbIcon,
  Analytics as AnalyticsIcon,
  Refresh as RefreshIcon
} from '@mui/icons-material';
import useAuthStore from '../store/authStore';
import {
  enhancedSearchCandidates,
  getSkillRecommendations,
  getMarketAnalysis,
  parseJobDescription,
  getEnhancedSearchHealth
} from '../utils/api';

interface EnhancedCandidate {
  id: number;
  first_name: string;
  last_name: string;
  email: string;
  skills: string[];
  experience: string;
  education: string;
  match_score: number;
  skill_match?: {
    score: number;
    matched_skills: Array<{
      skill: string;
      candidate_level: string;
      required_level: string;
      similarity: number;
    }>;
    missing_skills: string[];
  };
  experience_match?: {
    score: number;
    years_experience: number;
    required_years: number;
    role_similarity: number;
  };
  education_match?: {
    score: number;
    degree_relevance: number;
    level_match: boolean;
  };
  cultural_fit?: {
    score: number;
    personality_match: number;
    values_alignment: number;
  };
  explanation?: {
    strengths: string[];
    weaknesses: string[];
    recommendations: string[];
  };
  confidence?: number;
}

interface ParsedJob {
  required_skills: Array<{
    skill: string;
    proficiency_level: string;
    importance: string;
  }>;
  preferred_skills: Array<{
    skill: string;
    proficiency_level: string;
  }>;
  experience_requirements: {
    min_years: number;
    max_years: number;
    relevant_roles: string[];
  };
  education_requirements: {
    min_degree_level: string;
    preferred_fields: string[];
    certifications: string[];
  };
  company_culture: {
    work_style: string;
    team_size: string;
    growth_opportunities: string[];
  };
  confidence: number;
}

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`enhanced-search-tabpanel-${index}`}
      aria-labelledby={`enhanced-search-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

const EnhancedSearch: React.FC = () => {
  const [tabValue, setTabValue] = useState(0);
  const [jobDescription, setJobDescription] = useState('');
  const [jobTitle, setJobTitle] = useState('');
  const [companyName, setCompanyName] = useState('');
  const [candidates, setCandidates] = useState<EnhancedCandidate[]>([]);
  const [parsedJob, setParsedJob] = useState<ParsedJob | null>(null);
  const [skillRecommendations, setSkillRecommendations] = useState<any>(null);
  const [marketAnalysis, setMarketAnalysis] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [searchInsights, setSearchInsights] = useState<any>(null);
  const [systemHealth, setSystemHealth] = useState<any>(null);
  const user = useAuthStore((state) => state.user);

  useEffect(() => {
    checkSystemHealth();
  }, []);

  const checkSystemHealth = async () => {
    try {
      const health = await getEnhancedSearchHealth();
      setSystemHealth(health);
    } catch (err) {
      console.error('Error checking system health:', err);
    }
  };

  const handleParseJob = async () => {
    if (!jobDescription.trim()) {
      setError('Please enter a job description to parse.');
      return;
    }

    try {
      setLoading(true);
      setError('');
      const parsed = await parseJobDescription(jobDescription, jobTitle, companyName);
      setParsedJob(parsed);
    } catch (err) {
      setError('Failed to parse job description. Please try again.');
      console.error('Error parsing job:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleEnhancedSearch = async () => {
    if (!jobDescription.trim()) {
      setError('Please enter a job description to search.');
      return;
    }

    try {
      setLoading(true);
      setError('');
      
      const searchData = {
        job_description: jobDescription,
        job_title: jobTitle,
        company_name: companyName,
        limit: 20,
        min_score: 0.5
      };

      const results = await enhancedSearchCandidates(searchData);
      setCandidates(results.candidates || []);
      setSearchInsights(results.search_insights);
      
      if (results.parsed_job) {
        setParsedJob(results.parsed_job);
      }
    } catch (err) {
      setError('Failed to search candidates. Please try again.');
      console.error('Error searching candidates:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleGetSkillRecommendations = async () => {
    if (!jobDescription.trim()) {
      setError('Please enter a job description to get skill recommendations.');
      return;
    }

    try {
      setLoading(true);
      setError('');
      
      // Extract current skills from parsed job or use empty array
      const currentSkills = parsedJob?.required_skills.map(s => s.skill) || [];
      
      const recommendations = await getSkillRecommendations(jobDescription, currentSkills);
      setSkillRecommendations(recommendations);
    } catch (err) {
      setError('Failed to get skill recommendations. Please try again.');
      console.error('Error getting recommendations:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleGetMarketAnalysis = async () => {
    if (!parsedJob?.required_skills.length) {
      setError('Please parse a job description first to analyze market data.');
      return;
    }

    try {
      setLoading(true);
      setError('');
      
      const skills = parsedJob.required_skills.map(s => s.skill);
      const analysis = await getMarketAnalysis(skills);
      setMarketAnalysis(analysis);
    } catch (err) {
      setError('Failed to get market analysis. Please try again.');
      console.error('Error getting market analysis:', err);
    } finally {
      setLoading(false);
    }
  };

  const getScoreColor = (score: number) => {
    if (score >= 0.8) return 'success';
    if (score >= 0.6) return 'warning';
    return 'error';
  };

  const getScoreLabel = (score: number) => {
    if (score >= 0.9) return 'Excellent';
    if (score >= 0.8) return 'Very Good';
    if (score >= 0.7) return 'Good';
    if (score >= 0.6) return 'Fair';
    return 'Poor';
  };

  if (!user) {
    return (
      <Container>
        <Alert severity="error">Please log in to access the enhanced search.</Alert>
      </Container>
    );
  }

  return (
    <Container maxWidth="xl" sx={{ mt: 4 }}>
      {/* System Health Status */}
      {systemHealth && (
        <Alert 
          severity={systemHealth.status === 'healthy' ? 'success' : 'warning'} 
          sx={{ mb: 2 }}
          action={
            <IconButton size="small" onClick={checkSystemHealth}>
              <RefreshIcon />
            </IconButton>
          }
        >
          Enhanced Search System: {systemHealth.status} 
          {systemHealth.semantic_matcher && ` | AI Matching: ${systemHealth.semantic_matcher}`}
        </Alert>
      )}

      <Typography variant="h4" gutterBottom>
        🧠 AI-Powered Candidate Search
      </Typography>

      <Grid container spacing={3}>
        {/* Input Section */}
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Job Requirements
            </Typography>
            {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
            
            <TextField
              fullWidth
              label="Job Title"
              value={jobTitle}
              onChange={(e) => setJobTitle(e.target.value)}
              margin="normal"
              placeholder="e.g., Senior Python Developer"
            />
            
            <TextField
              fullWidth
              label="Company Name"
              value={companyName}
              onChange={(e) => setCompanyName(e.target.value)}
              margin="normal"
              placeholder="e.g., TechCorp Inc."
            />
            
            <TextField
              fullWidth
              multiline
              rows={8}
              label="Job Description"
              value={jobDescription}
              onChange={(e) => setJobDescription(e.target.value)}
              margin="normal"
              placeholder="Paste the complete job description here..."
            />

            <Box sx={{ mt: 2, display: 'flex', flexDirection: 'column', gap: 1 }}>
              <Button
                fullWidth
                variant="outlined"
                onClick={handleParseJob}
                disabled={loading}
                startIcon={<PsychologyIcon />}
              >
                Parse Job Description
              </Button>
              
              <Button
                fullWidth
                variant="contained"
                onClick={handleEnhancedSearch}
                disabled={loading}
                startIcon={<AnalyticsIcon />}
              >
                AI-Powered Search
              </Button>
              
              <Button
                fullWidth
                variant="outlined"
                onClick={handleGetSkillRecommendations}
                disabled={loading}
                startIcon={<LightbulbIcon />}
              >
                Get Skill Insights
              </Button>
              
              <Button
                fullWidth
                variant="outlined"
                onClick={handleGetMarketAnalysis}
                disabled={loading || !parsedJob}
                startIcon={<TrendingUpIcon />}
              >
                Market Analysis
              </Button>
            </Box>
          </Paper>
        </Grid>

        {/* Results Section */}
        <Grid item xs={12} md={8}>
          <Paper sx={{ p: 3 }}>
            <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
              <Tabs value={tabValue} onChange={(e, newValue) => setTabValue(newValue)}>
                <Tab 
                  label={
                    <Badge badgeContent={candidates.length} color="primary">
                      Candidates
                    </Badge>
                  } 
                />
                <Tab label="Job Analysis" />
                <Tab label="Skill Insights" />
                <Tab label="Market Data" />
              </Tabs>
            </Box>

            {loading && (
              <Box sx={{ mt: 2 }}>
                <LinearProgress />
                <Typography variant="body2" sx={{ mt: 1, textAlign: 'center' }}>
                  AI is analyzing candidates...
                </Typography>
              </Box>
            )}

            {/* Candidates Tab */}
            <TabPanel value={tabValue} index={0}>
              {candidates.length > 0 ? (
                <>
                  {searchInsights && (
                    <Card sx={{ mb: 2, bgcolor: 'primary.50' }}>
                      <CardContent>
                        <Typography variant="h6" gutterBottom>
                          Search Insights
                        </Typography>
                        <Grid container spacing={2}>
                          <Grid item xs={6} md={3}>
                            <Typography variant="body2" color="text.secondary">
                              Candidates Evaluated
                            </Typography>
                            <Typography variant="h6">
                              {searchInsights.total_candidates_evaluated || 0}
                            </Typography>
                          </Grid>
                          <Grid item xs={6} md={3}>
                            <Typography variant="body2" color="text.secondary">
                              Processing Time
                            </Typography>
                            <Typography variant="h6">
                              {searchInsights.processing_time_seconds?.toFixed(1) || 0}s
                            </Typography>
                          </Grid>
                          <Grid item xs={6} md={3}>
                            <Typography variant="body2" color="text.secondary">
                              Parsing Confidence
                            </Typography>
                            <Typography variant="h6">
                              {((searchInsights.parsing_confidence || 0) * 100).toFixed(0)}%
                            </Typography>
                          </Grid>
                          <Grid item xs={6} md={3}>
                            <Typography variant="body2" color="text.secondary">
                              Top Matches
                            </Typography>
                            <Typography variant="h6">
                              {candidates.filter(c => c.match_score > 0.8).length}
                            </Typography>
                          </Grid>
                        </Grid>
                      </CardContent>
                    </Card>
                  )}

                  <List>
                    {candidates.map((candidate) => (
                      <Accordion key={candidate.id} sx={{ mb: 1 }}>
                        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                          <Box sx={{ display: 'flex', alignItems: 'center', width: '100%' }}>
                            <Box sx={{ flexGrow: 1 }}>
                              <Typography variant="h6">
                                {candidate.first_name} {candidate.last_name}
                              </Typography>
                              <Typography variant="body2" color="text.secondary">
                                {candidate.email}
                              </Typography>
                            </Box>
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                              <Chip
                                label={`${(candidate.match_score * 100).toFixed(0)}%`}
                                color={getScoreColor(candidate.match_score)}
                                size="small"
                              />
                              <Chip
                                label={getScoreLabel(candidate.match_score)}
                                variant="outlined"
                                size="small"
                              />
                              {candidate.confidence && (
                                <Tooltip title={`Confidence: ${(candidate.confidence * 100).toFixed(0)}%`}>
                                  <StarIcon 
                                    color={candidate.confidence > 0.8 ? 'primary' : 'disabled'} 
                                    fontSize="small" 
                                  />
                                </Tooltip>
                              )}
                            </Box>
                          </Box>
                        </AccordionSummary>
                        <AccordionDetails>
                          <Grid container spacing={2}>
                            {/* Skills Match */}
                            {candidate.skill_match && (
                              <Grid item xs={12} md={6}>
                                <Card variant="outlined">
                                  <CardContent>
                                    <Typography variant="subtitle1" gutterBottom>
                                      <WorkIcon fontSize="small" sx={{ mr: 1, verticalAlign: 'middle' }} />
                                      Skills Match ({(candidate.skill_match.score * 100).toFixed(0)}%)
                                    </Typography>
                                    <LinearProgress 
                                      variant="determinate" 
                                      value={candidate.skill_match.score * 100}
                                      color={getScoreColor(candidate.skill_match.score)}
                                      sx={{ mb: 1 }}
                                    />
                                    {candidate.skill_match.matched_skills.slice(0, 3).map((skill, idx) => (
                                      <Chip
                                        key={idx}
                                        label={`${skill.skill} (${(skill.similarity * 100).toFixed(0)}%)`}
                                        size="small"
                                        sx={{ mr: 0.5, mb: 0.5 }}
                                        color={skill.similarity > 0.8 ? 'success' : 'default'}
                                      />
                                    ))}
                                    {candidate.skill_match.missing_skills.length > 0 && (
                                      <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                                        Missing: {candidate.skill_match.missing_skills.slice(0, 3).join(', ')}
                                      </Typography>
                                    )}
                                  </CardContent>
                                </Card>
                              </Grid>
                            )}

                            {/* Experience Match */}
                            {candidate.experience_match && (
                              <Grid item xs={12} md={6}>
                                <Card variant="outlined">
                                  <CardContent>
                                    <Typography variant="subtitle1" gutterBottom>
                                      <TrendingUpIcon fontSize="small" sx={{ mr: 1, verticalAlign: 'middle' }} />
                                      Experience ({(candidate.experience_match.score * 100).toFixed(0)}%)
                                    </Typography>
                                    <LinearProgress 
                                      variant="determinate" 
                                      value={candidate.experience_match.score * 100}
                                      color={getScoreColor(candidate.experience_match.score)}
                                      sx={{ mb: 1 }}
                                    />
                                    <Typography variant="body2">
                                      {candidate.experience_match.years_experience} years experience
                                    </Typography>
                                    <Typography variant="body2" color="text.secondary">
                                      Role similarity: {(candidate.experience_match.role_similarity * 100).toFixed(0)}%
                                    </Typography>
                                  </CardContent>
                                </Card>
                              </Grid>
                            )}

                            {/* Education Match */}
                            {candidate.education_match && (
                              <Grid item xs={12} md={6}>
                                <Card variant="outlined">
                                  <CardContent>
                                    <Typography variant="subtitle1" gutterBottom>
                                      <SchoolIcon fontSize="small" sx={{ mr: 1, verticalAlign: 'middle' }} />
                                      Education ({(candidate.education_match.score * 100).toFixed(0)}%)
                                    </Typography>
                                    <LinearProgress 
                                      variant="determinate" 
                                      value={candidate.education_match.score * 100}
                                      color={getScoreColor(candidate.education_match.score)}
                                      sx={{ mb: 1 }}
                                    />
                                    <Typography variant="body2">
                                      Degree relevance: {(candidate.education_match.degree_relevance * 100).toFixed(0)}%
                                    </Typography>
                                    <Typography variant="body2" color="text.secondary">
                                      Level match: {candidate.education_match.level_match ? 'Yes' : 'No'}
                                    </Typography>
                                  </CardContent>
                                </Card>
                              </Grid>
                            )}

                            {/* Cultural Fit */}
                            {candidate.cultural_fit && (
                              <Grid item xs={12} md={6}>
                                <Card variant="outlined">
                                  <CardContent>
                                    <Typography variant="subtitle1" gutterBottom>
                                      <PsychologyIcon fontSize="small" sx={{ mr: 1, verticalAlign: 'middle' }} />
                                      Cultural Fit ({(candidate.cultural_fit.score * 100).toFixed(0)}%)
                                    </Typography>
                                    <LinearProgress 
                                      variant="determinate" 
                                      value={candidate.cultural_fit.score * 100}
                                      color={getScoreColor(candidate.cultural_fit.score)}
                                      sx={{ mb: 1 }}
                                    />
                                    <Typography variant="body2">
                                      Personality: {(candidate.cultural_fit.personality_match * 100).toFixed(0)}%
                                    </Typography>
                                    <Typography variant="body2" color="text.secondary">
                                      Values: {(candidate.cultural_fit.values_alignment * 100).toFixed(0)}%
                                    </Typography>
                                  </CardContent>
                                </Card>
                              </Grid>
                            )}

                            {/* AI Explanation */}
                            {candidate.explanation && (
                              <Grid item xs={12}>
                                <Card variant="outlined" sx={{ bgcolor: 'grey.50' }}>
                                  <CardContent>
                                    <Typography variant="subtitle1" gutterBottom>
                                      🤖 AI Analysis
                                    </Typography>
                                    
                                    {candidate.explanation.strengths.length > 0 && (
                                      <Box sx={{ mb: 2 }}>
                                        <Typography variant="body2" color="success.main" fontWeight="bold">
                                          Strengths:
                                        </Typography>
                                        <ul style={{ margin: '4px 0', paddingLeft: '20px' }}>
                                          {candidate.explanation.strengths.map((strength, idx) => (
                                            <li key={idx}>
                                              <Typography variant="body2">{strength}</Typography>
                                            </li>
                                          ))}
                                        </ul>
                                      </Box>
                                    )}

                                    {candidate.explanation.weaknesses.length > 0 && (
                                      <Box sx={{ mb: 2 }}>
                                        <Typography variant="body2" color="warning.main" fontWeight="bold">
                                          Areas for Consideration:
                                        </Typography>
                                        <ul style={{ margin: '4px 0', paddingLeft: '20px' }}>
                                          {candidate.explanation.weaknesses.map((weakness, idx) => (
                                            <li key={idx}>
                                              <Typography variant="body2">{weakness}</Typography>
                                            </li>
                                          ))}
                                        </ul>
                                      </Box>
                                    )}

                                    {candidate.explanation.recommendations.length > 0 && (
                                      <Box>
                                        <Typography variant="body2" color="primary.main" fontWeight="bold">
                                          Recommendations:
                                        </Typography>
                                        <ul style={{ margin: '4px 0', paddingLeft: '20px' }}>
                                          {candidate.explanation.recommendations.map((rec, idx) => (
                                            <li key={idx}>
                                              <Typography variant="body2">{rec}</Typography>
                                            </li>
                                          ))}
                                        </ul>
                                      </Box>
                                    )}
                                  </CardContent>
                                </Card>
                              </Grid>
                            )}
                          </Grid>
                        </AccordionDetails>
                      </Accordion>
                    ))}
                  </List>
                </>
              ) : (
                <Typography color="text.secondary" align="center" sx={{ mt: 4 }}>
                  No candidates found. Try running an AI-powered search.
                </Typography>
              )}
            </TabPanel>

            {/* Job Analysis Tab */}
            <TabPanel value={tabValue} index={1}>
              {parsedJob ? (
                <Grid container spacing={3}>
                  <Grid item xs={12}>
                    <Alert severity="info" sx={{ mb: 2 }}>
                      Parsing Confidence: {(parsedJob.confidence * 100).toFixed(0)}%
                    </Alert>
                  </Grid>

                  <Grid item xs={12} md={6}>
                    <Card>
                      <CardContent>
                        <Typography variant="h6" gutterBottom>
                          Required Skills
                        </Typography>
                        {parsedJob.required_skills.map((skill, idx) => (
                          <Chip
                            key={idx}
                            label={`${skill.skill} (${skill.proficiency_level})`}
                            color={skill.importance === 'high' ? 'primary' : 'default'}
                            sx={{ mr: 0.5, mb: 0.5 }}
                          />
                        ))}
                      </CardContent>
                    </Card>
                  </Grid>

                  <Grid item xs={12} md={6}>
                    <Card>
                      <CardContent>
                        <Typography variant="h6" gutterBottom>
                          Preferred Skills
                        </Typography>
                        {parsedJob.preferred_skills.map((skill, idx) => (
                          <Chip
                            key={idx}
                            label={`${skill.skill} (${skill.proficiency_level})`}
                            variant="outlined"
                            sx={{ mr: 0.5, mb: 0.5 }}
                          />
                        ))}
                      </CardContent>
                    </Card>
                  </Grid>

                  <Grid item xs={12} md={6}>
                    <Card>
                      <CardContent>
                        <Typography variant="h6" gutterBottom>
                          Experience Requirements
                        </Typography>
                        <Typography variant="body2">
                          Years: {parsedJob.experience_requirements.min_years}-{parsedJob.experience_requirements.max_years}
                        </Typography>
                        <Typography variant="body2" sx={{ mt: 1 }}>
                          Relevant Roles: {parsedJob.experience_requirements.relevant_roles.join(', ')}
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>

                  <Grid item xs={12} md={6}>
                    <Card>
                      <CardContent>
                        <Typography variant="h6" gutterBottom>
                          Education & Culture
                        </Typography>
                        <Typography variant="body2">
                          Min Degree: {parsedJob.education_requirements.min_degree_level}
                        </Typography>
                        <Typography variant="body2">
                          Work Style: {parsedJob.company_culture.work_style}
                        </Typography>
                        <Typography variant="body2">
                          Team Size: {parsedJob.company_culture.team_size}
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                </Grid>
              ) : (
                <Typography color="text.secondary" align="center" sx={{ mt: 4 }}>
                  Parse a job description to see detailed analysis.
                </Typography>
              )}
            </TabPanel>

            {/* Skill Insights Tab */}
            <TabPanel value={tabValue} index={2}>
              {skillRecommendations ? (
                <Grid container spacing={3}>
                  <Grid item xs={12}>
                    <Card>
                      <CardContent>
                        <Typography variant="h6" gutterBottom>
                          Skill Gap Analysis
                        </Typography>
                        {skillRecommendations.skill_gaps?.map((gap: any, idx: number) => (
                          <Box key={idx} sx={{ mb: 2 }}>
                            <Typography variant="subtitle1">{gap.skill}</Typography>
                            <Typography variant="body2" color="text.secondary">
                              Priority: {gap.priority} | Market Demand: {(gap.market_demand * 100).toFixed(0)}%
                            </Typography>
                            <Typography variant="body2">
                              {gap.learning_path}
                            </Typography>
                          </Box>
                        ))}
                      </CardContent>
                    </Card>
                  </Grid>

                  <Grid item xs={12}>
                    <Card>
                      <CardContent>
                        <Typography variant="h6" gutterBottom>
                          Learning Recommendations
                        </Typography>
                        {skillRecommendations.learning_path?.map((item: any, idx: number) => (
                          <Box key={idx} sx={{ mb: 1 }}>
                            <Chip
                              label={`${item.skill} (${item.estimated_time})`}
                              color={item.priority === 'high' ? 'primary' : 'default'}
                              sx={{ mr: 0.5 }}
                            />
                          </Box>
                        ))}
                      </CardContent>
                    </Card>
                  </Grid>
                </Grid>
              ) : (
                <Typography color="text.secondary" align="center" sx={{ mt: 4 }}>
                  Get skill insights to see recommendations.
                </Typography>
              )}
            </TabPanel>

            {/* Market Data Tab */}
            <TabPanel value={tabValue} index={3}>
              {marketAnalysis ? (
                <Grid container spacing={3}>
                  <Grid item xs={12}>
                    <Card>
                      <CardContent>
                        <Typography variant="h6" gutterBottom>
                          Market Demand Analysis
                        </Typography>
                        {marketAnalysis.skill_analysis?.map((skill: any, idx: number) => (
                          <Box key={idx} sx={{ mb: 2 }}>
                            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                              <Typography variant="subtitle1">{skill.skill}</Typography>
                              <Chip
                                label={`${(skill.demand_score * 100).toFixed(0)}% demand`}
                                color={skill.demand_score > 0.8 ? 'success' : skill.demand_score > 0.6 ? 'warning' : 'default'}
                                size="small"
                              />
                            </Box>
                            <LinearProgress
                              variant="determinate"
                              value={skill.demand_score * 100}
                              color={skill.demand_score > 0.8 ? 'success' : skill.demand_score > 0.6 ? 'warning' : 'primary'}
                              sx={{ mt: 1, mb: 1 }}
                            />
                            <Typography variant="body2" color="text.secondary">
                              Growth: {skill.growth_trend} | Salary Impact: +{((skill.salary_impact - 1) * 100).toFixed(0)}%
                            </Typography>
                          </Box>
                        ))}
                      </CardContent>
                    </Card>
                  </Grid>

                  {marketAnalysis.market_trends && (
                    <Grid item xs={12}>
                      <Card>
                        <CardContent>
                          <Typography variant="h6" gutterBottom>
                            Market Trends
                          </Typography>
                          {marketAnalysis.market_trends.map((trend: any, idx: number) => (
                            <Box key={idx} sx={{ mb: 2 }}>
                              <Typography variant="subtitle1">{trend.trend_name}</Typography>
                              <Typography variant="body2" color="text.secondary">
                                {trend.description}
                              </Typography>
                              <Box sx={{ mt: 1 }}>
                                {trend.affected_skills.slice(0, 5).map((skill: string, skillIdx: number) => (
                                  <Chip
                                    key={skillIdx}
                                    label={skill}
                                    size="small"
                                    sx={{ mr: 0.5, mb: 0.5 }}
                                  />
                                ))}
                              </Box>
                            </Box>
                          ))}
                        </CardContent>
                      </Card>
                    </Grid>
                  )}
                </Grid>
              ) : (
                <Typography color="text.secondary" align="center" sx={{ mt: 4 }}>
                  Run market analysis to see demand trends.
                </Typography>
              )}
            </TabPanel>
          </Paper>
        </Grid>
      </Grid>
    </Container>
  );
};

export default EnhancedSearch; 