import React, { useEffect, useState } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, LineChart, Line, Legend,
} from 'recharts';
import { api } from '../api';

const RISK_COLORS = {
  very_low:  '#3fb950',
  low:       '#58a6ff',
  moderate:  '#d29922',
  high:      '#f0883e',
  very_high: '#f85149',
};

const RISK_LABELS = {
  very_low:  'Juda past',
  low:       'Past',
  moderate:  "O'rta",
  high:      'Yuqori',
  very_high: 'Juda yuqori',
};

export default function Dashboard() {
  const [stats, setStats]   = useState(null);
  const [trend, setTrend]   = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError]   = useState(null);

  useEffect(() => {
    Promise.all([api.getStats(), api.getTrend()])
      .then(([s, t]) => {
        setStats(s.data);
        // Trend ma'lumotlarini yillar bo'yicha pivot
        const map = {};
        t.data.trend.forEach(({ year, risk_level, count }) => {
          if (!map[year]) map[year] = { year };
          map[year][risk_level] = count;
        });
        setTrend(Object.values(map).sort((a, b) => a.year - b.year));
      })
      .catch(() => setError('API ga ulanib bo\'lmadi'))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="loading">Yuklanmoqda...</div>;
  if (error)   return <div className="loading" style={{ color: 'var(--red)' }}>{error}</div>;

  const pieData = stats.risk_distribution.map(r => ({
    name:  RISK_LABELS[r.risk_level] || r.risk_level,
    value: r.count,
    color: RISK_COLORS[r.risk_level],
    pct:   r.pct,
  }));

  return (
    <>
      <div className="page-header">
        <h2>Dashboard</h2>
        <p>Diabet xavfi monitoring tizimi</p>
      </div>

      {/* Stat cards */}
      <div className="stats-grid">
        <div className="stat-card">
          <div className="label">Jami bemorlar</div>
          <div className="value" style={{ color: 'var(--accent)' }}>
            {stats.total_patients.toLocaleString()}
          </div>
          <div className="sub">2020–2026</div>
        </div>
        <div className="stat-card">
          <div className="label">Risk baholashlar</div>
          <div className="value" style={{ color: 'var(--green)' }}>
            {stats.total_assessments.toLocaleString()}
          </div>
          <div className="sub">Jami tekshiruvlar</div>
        </div>
        <div className="stat-card">
          <div className="label">O'rtacha HbA1c</div>
          <div className="value" style={{ color: 'var(--yellow)' }}>
            {stats.averages.avg_hba1c}
            <span style={{ fontSize: 16, fontWeight: 400, color: 'var(--text3)' }}>%</span>
          </div>
          <div className="sub">Normal: 4.0–5.6%</div>
        </div>
        <div className="stat-card">
          <div className="label">Bildirishnomalar</div>
          <div className="value" style={{ color: 'var(--red)' }}>
            {stats.unsent_notifications.toLocaleString()}
          </div>
          <div className="sub">Yuborilmagan</div>
        </div>
      </div>

      {/* Charts */}
      <div className="charts-grid">

        {/* Pie chart */}
        <div className="card">
          <div className="card-title">Risk darajalari taqsimoti</div>
          <ResponsiveContainer width="100%" height={220}>
            <PieChart>
              <Pie
                data={pieData}
                cx="50%" cy="50%"
                innerRadius={55} outerRadius={85}
                paddingAngle={3}
                dataKey="value"
              >
                {pieData.map((entry, i) => (
                  <Cell key={i} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip
                contentStyle={{ background: 'var(--bg3)', border: '1px solid var(--border)', borderRadius: 8 }}
                formatter={(v, n, p) => [`${v.toLocaleString()} (${p.payload.pct}%)`, p.payload.name]}
              />
            </PieChart>
          </ResponsiveContainer>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px 14px', marginTop: 8 }}>
            {pieData.map((d, i) => (
              <span key={i} style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, color: 'var(--text2)' }}>
                <span style={{ width: 8, height: 8, borderRadius: '50%', background: d.color }} />
                {d.name} {d.pct}%
              </span>
            ))}
          </div>
        </div>

        {/* Bar chart - average biomarkers */}
        <div className="card">
          <div className="card-title">O'rtacha ko'rsatkichlar</div>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={[
              { name: 'HbA1c (%)',      value: stats.averages.avg_hba1c,   normal: 5.6  },
              { name: 'Glucose/10',     value: +(stats.averages.avg_glucose / 10).toFixed(1), normal: 10 },
              { name: 'BMI',            value: stats.averages.avg_bmi,     normal: 24.9 },
            ]} barSize={28}>
              <XAxis dataKey="name" tick={{ fill: 'var(--text3)', fontSize: 12 }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fill: 'var(--text3)', fontSize: 11 }} axisLine={false} tickLine={false} />
              <Tooltip
                contentStyle={{ background: 'var(--bg3)', border: '1px solid var(--border)', borderRadius: 8 }}
              />
              <Bar dataKey="value" fill="var(--accent)" radius={[4, 4, 0, 0]} />
              <Bar dataKey="normal" fill="var(--border)" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Trend line chart */}
      <div className="card">
        <div className="card-title">Yillik trend — risk taqsimoti</div>
        <ResponsiveContainer width="100%" height={240}>
          <LineChart data={trend} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
            <XAxis dataKey="year" tick={{ fill: 'var(--text3)', fontSize: 12 }} axisLine={false} tickLine={false} />
            <YAxis tick={{ fill: 'var(--text3)', fontSize: 11 }} axisLine={false} tickLine={false} />
            <Tooltip
              contentStyle={{ background: 'var(--bg3)', border: '1px solid var(--border)', borderRadius: 8 }}
            />
            <Legend wrapperStyle={{ fontSize: 12, color: 'var(--text2)' }} />
            {Object.keys(RISK_COLORS).map(level => (
              <Line
                key={level}
                type="monotone"
                dataKey={level}
                name={RISK_LABELS[level]}
                stroke={RISK_COLORS[level]}
                strokeWidth={2}
                dot={false}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </>
  );
}
