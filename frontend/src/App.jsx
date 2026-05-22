import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom';
import './index.css';
import Dashboard from './pages/Dashboard';
import Patients  from './pages/Patients';
import Predict   from './pages/Predict';
import { LangProvider, useLang, LANGUAGES } from './i18n';

function Sidebar() {
  const { t, lang, changeLang } = useLang();
  return (
    <aside className="sidebar">
      <div className="sidebar-logo">
        <h1>🩺 {t('app.title')}</h1>
        <span>{t('app.version')}</span>
      </div>
      <nav>
        <NavLink to="/"         end>📊 {t('nav.dashboard')}</NavLink>
        <NavLink to="/patients">👥 {t('nav.patients')}</NavLink>
        <NavLink to="/predict"> 🔬 {t('nav.predict')}</NavLink>
      </nav>

      <div className="lang-switcher">
        {LANGUAGES.map(({ code, label, flag }) => (
          <button
            key={code}
            className={`lang-btn${lang === code ? ' active' : ''}`}
            onClick={() => changeLang(code)}
            title={label}
          >
            {flag} {label}
          </button>
        ))}
      </div>

      <div className="sidebar-footer">
        <span className="dot" />
        <span style={{ fontSize: 12, color: 'var(--text3)' }}>{t('nav.airflow')}</span>
      </div>
    </aside>
  );
}

function AppInner() {
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

export default function App() {
  return (
    <LangProvider>
      <AppInner />
    </LangProvider>
  );
}
