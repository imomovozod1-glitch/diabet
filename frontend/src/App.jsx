import React from 'react';
import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom';
import './index.css';
import Dashboard from './pages/Dashboard';
import Patients  from './pages/Patients';
import Predict   from './pages/Predict';

function Sidebar() {
  return (
    <aside className="sidebar">
      <div className="sidebar-logo">
        <h1>🩺 DiabetRisk</h1>
        <span>v1.0 · ML Pipeline</span>
      </div>
      <nav>
        <NavLink to="/"         end>📊 Dashboard</NavLink>
        <NavLink to="/patients">👥 Bemorlar</NavLink>
        <NavLink to="/predict"> 🔬 Bashorat</NavLink>
      </nav>
      <div className="sidebar-footer">
        <span className="dot" />
        <span style={{ fontSize: 12, color: 'var(--text3)' }}>Airflow · har soat</span>
      </div>
    </aside>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <div className="layout">
        <Sidebar />
        <main className="main">
          <Routes>
            <Route path="/"         element={<Dashboard />} />
            <Route path="/patients" element={<Patients />}  />
            <Route path="/predict"  element={<Predict />}   />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}
