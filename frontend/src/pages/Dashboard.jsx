import { useEffect, useState } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, LineChart, Line, Legend,
} from 'recharts';
import { api } from '../api';
import { useLang } from '../i18n';

const RISK_COLORS = {
  very_low:  '#3fb950',
  low:       '#58a6ff',
  moderate:  '#d29922',
  high:      '#f0883e',
  very_high: '#f85149',
};

export default function Dashboard() {
  const { t } = useLang();
  const [stats, setStats]     = useState(null);
  const [trend, setTrend]     = useState([]);
  const [loading, setLoading] = useState(true);
  const [hasError, setError]  = useState(false);

  useEffect(() => {
    Promise.all([api.getStats(), api.getTrend()])
      .then(([s, tr]) => {
        setStats(s.data);
        const map = {};
        tr.data.trend.forEach(({ year, risk_level, count }) => {
          if (!map[year]) map[year] = { year };
          map[year][risk_level] = count;
        });
        setTrend(Object.values(map).sort((a, b) => a.year - b.year));
      })
      .catch(() => setError(true))
      .finally(() => setLoading(false));
  }, []);

  if (loading)  return <div className="loading">{t('common.loading')}</div>;
  if (hasError) return <div className="loading" style={{ color: 'var(--red)' }}>{t('common.errorApi')}</div>;

  const pieData = stats.risk_distribution.map(r => ({
    name:  t(`risk.${r.risk_level}`),
    value: r.count,
    color: RISK_COLORS[r.risk_level],
    pct:   r.pct,
  }));

  return (
    <>
      <div className="page-header">
        <h2>{t('dashboard.title')}</h2>
        <p>{t('dashboard.subtitle')}</p>
      </div>

      <div className="stats-grid">
        <div className="stat-card">
          <div className="label">{t('dashboard.totalPatients')}</div>
          <div className="value" style={{ color: 'var(--accent)' }}>
            {stats.total_patients.toLocaleString()}
          </div>
          <div className="sub">{t('dashboard.period')}</div>
        </div>
        <div className="stat-card">
          <div className="label">{t('dashboard.riskAssessments')}</div>
          <div className="value" style={{ color: 'var(--green)' }}>
            {stats.total_assessments.toLocaleString()}
          </div>
          <div className="sub">{t('dashboard.totalExams')}</div>
        </div>
        <div className="stat-card">
          <div className="label">{t('dashboard.avgHba1c')}</div>
          <div className="value" style={{ color: 'var(--yellow)' }}>
            {stats.averages.avg_hba1c}
            <span style={{ fontSize: 16, fontWeight: 400, color: 'var(--text3)' }}>%</span>
          </div>
          <div className="sub">{t('dashboard.normalHba1c')}</div>
        </div>
        <div className="stat-card">
          <div className="label">{t('dashboard.notifications')}</div>
          <div className="value" style={{ color: 'var(--red)' }}>
            {stats.unsent_notifications.toLocaleString()}
          </div>
          <div className="sub">{t('dashboard.unsent')}</div>
        </div>
      </div>

      <div className="charts-grid">
        <div className="card">
          <div className="card-title">{t('dashboard.riskDistribution')}</div>
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
                formatter={(v, _, p) => [`${v.toLocaleString()} (${p.payload.pct}%)`, p.payload.name]}
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

        <div className="card">
          <div className="card-title">{t('dashboard.avgIndicators')}</div>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={[
              { name: 'HbA1c (%)',  value: stats.averages.avg_hba1c,   normal: 5.6  },
              { name: 'Glucose/10', value: +(stats.averages.avg_glucose / 10).toFixed(1), normal: 10 },
              { name: 'BMI',        value: stats.averages.avg_bmi,     normal: 24.9 },
            ]} barSize={28}>
              <XAxis dataKey="name" tick={{ fill: 'var(--text3)', fontSize: 12 }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fill: 'var(--text3)', fontSize: 11 }} axisLine={false} tickLine={false} />
              <Tooltip contentStyle={{ background: 'var(--bg3)', border: '1px solid var(--border)', borderRadius: 8 }} />
              <Bar dataKey="value" fill="var(--accent)" radius={[4, 4, 0, 0]} />
              <Bar dataKey="normal" fill="var(--border)" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="card">
        <div className="card-title">{t('dashboard.annualTrend')}</div>
        <ResponsiveContainer width="100%" height={240}>
          <LineChart data={trend} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
            <XAxis dataKey="year" tick={{ fill: 'var(--text3)', fontSize: 12 }} axisLine={false} tickLine={false} />
            <YAxis tick={{ fill: 'var(--text3)', fontSize: 11 }} axisLine={false} tickLine={false} />
            <Tooltip contentStyle={{ background: 'var(--bg3)', border: '1px solid var(--border)', borderRadius: 8 }} />
            <Legend wrapperStyle={{ fontSize: 12, color: 'var(--text2)' }} />
            {Object.keys(RISK_COLORS).map(level => (
              <Line
                key={level}
                type="monotone"
                dataKey={level}
                name={t(`risk.${level}`)}
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
