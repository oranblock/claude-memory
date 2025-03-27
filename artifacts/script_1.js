javascriptCopy// UserDashboard.js - React component with hooks
import React, { useState, useEffect } from 'react';
import { fetchUserData, updateUserPreferences } from '../api/userService';
import DashboardLayout from '../layouts/DashboardLayout';
import UserStatistics from './UserStatistics';
import PreferencesPanel from './PreferencesPanel';

const UserDashboard = ({ userId, theme }) => {
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
  
  if (loading) return <div className="loading-spinner">Loading...</div>;
  if (error) return <div className="error-message">{error}</div>;
  
  return (
    <DashboardLayout>
      <div className="user-dashboard">
        <header className="dashboard-header">
          <h1>Welcome back, {userData.name}</h1>
          <div className="tab-navigation">
            <button 
              className={activeTab === 'overview' ? 'active' : ''} 
              onClick={() => handleTabChange('overview')}
            >
              Overview
            </button>
            <button 
              className={activeTab === 'statistics' ? 'active' : ''} 
              onClick={() => handleTabChange('statistics')}
            >
              Statistics
            </button>
            <button 
              className={activeTab === 'preferences' ? 'active' : ''} 
              onClick={() => handleTabChange('preferences')}
            >
              Preferences
            </button>
          </div>
        </header>
        
        <main className="dashboard-content">
          {activeTab === 'overview' && (
            <div className="overview-panel">
              <h2>Account Overview</h2>
              <div className="account-info">
                <p><strong>Member since:</strong> {new Date(userData.joinDate).toLocaleDateString()}</p>
                <p><strong>Subscription:</strong> {userData.subscription.plan}</p>
                <p><strong>Status:</strong> {userData.status}</p>
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
        </main>
      </div>
    </DashboardLayout>
  );
};

export default UserDashboard;