// UserDashboard.js
import React, { useState, useEffect } from 'react';
import { fetchUserData } from '../services/api';

const UserDashboard = ({ userId }) => {
  const [userData, setUserData] = useState(null);
  
  useEffect(() => {
    fetchUserData(userId).then(data => setUserData(data));
  }, [userId]);
  
  return userData ? <div className="dashboard">{userData.name}'s Dashboard</div> : <div>Loading...</div>;
};

export default UserDashboard;