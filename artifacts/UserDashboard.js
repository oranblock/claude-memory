// UserDashboard.jsx - React component with hooks and context
import React, { useState, useEffect, useContext } from 'react';
import { fetchUserData, updateUserPreferences } from '../api/userService';
import { ThemeContext } from '../contexts/ThemeContext';
import DashboardLayout from '../layouts/DashboardLayout';
import UserStatistics from './UserStatistics';
import PreferencesPanel from './PreferencesPanel';
import { Alert, Button, Spinner } from '../components/ui';

const UserDashboard = ({ userId }) => {
  const { theme, toggleTheme } = useContext(ThemeContext);
  const [userData, setUserData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('overview');
  
  useEffect(() => {
    async function loadUserData() {
      try {
        setLoading(true);
        const data = await fetchUserData(userId);
        setUserData(data);
        setLoading(false);
      } catch (err) {
        setError('Failed to load user data');
        setLoading(false);
        console.error(err);
      }
    }
    
    loadUserData();
  }, [userId]);
  
  const handleTabChange = (tab) => {
    setActiveTab(tab);
  };
  
  const handlePreferenceUpdate = async (preferences) => {
    try {
      await updateUserPreferences(userId, preferences);
      // Update local user data
      setUserData({
        ...userData,
        preferences
      });
    } catch (err) {
      setError('Failed to update preferences');
      console.error(err);
    }
  };
  
  if (loading) return (
    <div className="loading-container">
      <Spinner size="large" />
      <p>Loading user dashboard...</p>
    </div>
  );
  
  if (error) return (
    <Alert type="error" title="Error Loading Data">
      {error}
      <Button onClick={() => window.location.reload()}>
        Try Again
      </Button>
    </Alert>
  );
  
  return (
    <DashboardLayout>
      <div className={`user-dashboard theme-${theme}`}>
        <header className="dashboard-header">
          <div className="header-left">
            <h1>Welcome back, {userData.name}</h1>
            <p className="last-login">Last login: {new Date(userData.lastLogin).toLocaleString()}</p>
          </div>
          <div className="header-right">
            <Button onClick={toggleTheme} variant="outline">
              {theme === 'dark' ? '☀️ Light Mode' : '🌙 Dark Mode'}
            </Button>
          </div>
        </header>
        
        <nav className="tab-navigation">
          <Button 
            className={activeTab === 'overview' ? 'active' : ''} 
            onClick={() => handleTabChange('overview')}
          >
            Overview
          </Button>
          <Button 
            className={activeTab === 'statistics' ? 'active' : ''} 
            onClick={() => handleTabChange('statistics')}
          >
            Statistics
          </Button>
          <Button 
            className={activeTab === 'preferences' ? 'active' : ''} 
            onClick={() => handleTabChange('preferences')}
          >
            Preferences
          </Button>
          <Button 
            className={activeTab === 'reports' ? 'active' : ''} 
            onClick={() => handleTabChange('reports')}
          >
            Reports
          </Button>
        </nav>
        
        <main className="dashboard-content">
          {activeTab === 'overview' && (
            <div className="overview-panel">
              <h2>Account Overview</h2>
              <div className="account-info">
                <p><strong>Member since:</strong> {new Date(userData.joinDate).toLocaleDateString()}</p>
                <p><strong>Subscription:</strong> {userData.subscription.plan}</p>
                <p><strong>Status:</strong> {userData.status}</p>
                <p><strong>Next billing date:</strong> {new Date(userData.subscription.nextBillingDate).toLocaleDateString()}</p>
              </div>
              
              <div className="quick-stats">
                <div className="stat-card">
                  <div className="stat-value">{userData.statistics.totalLogins}</div>
                  <div className="stat-label">Total Logins</div>
                </div>
                <div className="stat-card">
                  <div className="stat-value">{userData.statistics.projectsCreated}</div>
                  <div className="stat-label">Projects</div>
                </div>
                <div className="stat-card">
                  <div className="stat-value">{userData.statistics.reportsGenerated}</div>
                  <div className="stat-label">Reports</div>
                </div>
                <div className="stat-card">
                  <div className="stat-value">{userData.storage.used} / {userData.storage.total} GB</div>
                  <div className="stat-label">Storage</div>
                </div>
              </div>
            </div>
          )}
          
          {activeTab === 'statistics' && (
            <UserStatistics stats={userData.statistics} />
          )}
          
          {activeTab === 'preferences' && (
            <PreferencesPanel 
              preferences={userData.preferences}
              onUpdate={handlePreferenceUpdate}
            />
          )}
          
          {activeTab === 'reports' && (
            <div className="reports-panel">
              <h2>Saved Reports</h2>
              {userData.reports.length === 0 ? (
                <p>No reports saved yet.</p>
              ) : (
                <ul className="reports-list">
                  {userData.reports.map(report => (
                    <li key={report.id} className="report-item">
                      <div className="report-title">{report.title}</div>
                      <div className="report-date">{new Date(report.createdAt).toLocaleDateString()}</div>
                      <div className="report-actions">
                        <Button size="small" variant="text">View</Button>
                        <Button size="small" variant="text">Download</Button>
                      </div>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          )}
        </main>
      </div>
    </DashboardLayout>
  );
};

export default UserDashboard;