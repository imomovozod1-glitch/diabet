import React, { useEffect, useState, useCallback } from 'react';
import { api } from '../api';

const RISK_LABELS = {
  very_low: 'Juda past', low: 'Past', moderate: "O'rta",
  high: 'Yuqori', very_high: 'Juda yuqori',
};

function RiskBadge({ level }) {
  return (
    <span className={`risk-badge risk-${level}`}>
      {RISK_LABELS[level] || level}
    </span>
  );
}

function PatientModal({ patient, onClose }) {
  const [detail, setDetail] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.getPatient(patient.id)
      .then(r => setDetail(r.data))
      .finally(() => setLoading(false));
  }, [patient.id]);

  return (
    <div
      style={{
        position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.7)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        zIndex: 1000, padding: 20,
      }}
      onClick={onClose}
    >
      <div
        style={{
          background: 'var(--bg2)', border: '1px solid var(--border)',
          borderRadius: 12, padding: 24, width: '100%', maxWidth: 640,
          maxHeight: '85vh', overflowY: 'auto',
        }}
        onClick={e => e.stopPropagation()}
      >
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 20 }}>
          <div>
            <h3 style={{ fontSize: 18, fontWeight: 600 }}>
              {patient.first_name} {patient.last_name}
            </h3>
            <span style={{ fontSize: 12, color: 'var(--text3)' }}>
              {patient.age} yosh · {patient.gender === 'male' ? 'Erkak' : 'Ayol'} · {patient.address}
            </span>
          </div>
          <button
            onClick={onClose}
            style={{ background: 'none', border: 'none', color: 'var(--text3)', fontSize: 20, cursor: 'pointer' }}
          >✕</button>
        </div>

        {loading ? (
          <div className="loading">Yuklanmoqda...</div>
        ) : (
          <>
            {/* Risk tarixi */}
            {detail.risk_history?.length > 0 && (
              <div style={{ marginBottom: 20 }}>
                <div className="card-title">Risk baholash</div>
                {detail.risk_history.slice(0, 1).map((r, i) => (
                  <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 12, padding: '12px 0' }}>
                    <RiskBadge level={r.risk_level} />
                    <span style={{ fontSize: 13, color: 'var(--text2)' }}>
                      Risk skori: <b style={{ color: 'var(--text)' }}>{(r.risk_score * 100).toFixed(1)}%</b>
                    </span>
                  </div>
                ))}
                <p style={{ fontSize: 12.5, color: 'var(--text2)', lineHeight: 1.6 }}>
                  {detail.risk_history[0]?.recommendation}
                </p>
              </div>
            )}

            {/* Tahlil natijalari */}
            {detail.test_results?.length > 0 && (
              <div>
                <div className="card-title" style={{ marginBottom: 10 }}>Tahlil natijalari</div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                  {detail.test_results.slice(0, 12).map((r, i) => (
                    <div key={i} style={{
                      background: 'var(--bg3)', borderRadius: 7, padding: '8px 12px',
                      display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                    }}>
                      <span style={{ fontSize: 12, color: 'var(--text2)' }}>{r.biomarker}</span>
                      <span style={{
                        fontSize: 13, fontFamily: 'DM Mono, monospace', fontWeight: 500,
                        color: r.status === 'normal' ? 'var(--text)' :
                               r.status === 'high' || r.status === 'critical_high' ? 'var(--red)' : 'var(--yellow)',
                      }}>
                        {Number(r.value).toFixed(1)} {r.unit}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

export default function Patients() {
  const [data, setData]       = useState({ data: [], total: 0, pages: 1 });
  const [loading, setLoading] = useState(true);
  const [page, setPage]       = useState(1);
  const [search, setSearch]   = useState('');
  const [gender, setGender]   = useState('');
  const [riskLevel, setRisk]  = useState('');
  const [selected, setSelected] = useState(null);

  const load = useCallback(() => {
    setLoading(true);
    api.getPatients({ page, limit: 20, search, gender, risk_level: riskLevel })
      .then(r => setData(r.data))
      .finally(() => setLoading(false));
  }, [page, search, gender, riskLevel]);

  useEffect(() => { load(); }, [load]);

  // Qidiruv kiritilganda 1-sahifaga qaytish
  const handleSearch = e => { setSearch(e.target.value); setPage(1); };

  return (
    <>
      <div className="page-header">
        <h2>Bemorlar</h2>
        <p>Jami {data.total.toLocaleString()} ta bemor</p>
      </div>

      {/* Filterlar */}
      <div className="filter-bar">
        <input
          placeholder="🔍 Ism yoki familiya..."
          value={search}
          onChange={handleSearch}
          style={{ flex: 1, minWidth: 200 }}
        />
        <select value={gender} onChange={e => { setGender(e.target.value); setPage(1); }}>
          <option value="">Barcha jinslar</option>
          <option value="male">Erkak</option>
          <option value="female">Ayol</option>
        </select>
        <select value={riskLevel} onChange={e => { setRisk(e.target.value); setPage(1); }}>
          <option value="">Barcha risk</option>
          <option value="very_low">Juda past</option>
          <option value="low">Past</option>
          <option value="moderate">O'rta</option>
          <option value="high">Yuqori</option>
          <option value="very_high">Juda yuqori</option>
        </select>
        <button className="btn btn-outline" onClick={load}>↺ Yangilash</button>
      </div>

      {/* Jadval */}
      <div className="table-wrap">
        {loading ? (
          <div className="loading">Yuklanmoqda...</div>
        ) : data.data.length === 0 ? (
          <div className="empty">Bemor topilmadi</div>
        ) : (
          <>
            <table>
              <thead>
                <tr>
                  <th>Ism Familiya</th>
                  <th>Yosh</th>
                  <th>Jins</th>
                  <th>Manzil</th>
                  <th>Risk darajasi</th>
                  <th>Risk skori</th>
                  <th>Sana</th>
                </tr>
              </thead>
              <tbody>
                {data.data.map(p => (
                  <tr key={p.id} onClick={() => setSelected(p)}>
                    <td style={{ fontWeight: 500 }}>{p.first_name} {p.last_name}</td>
                    <td>{p.age}</td>
                    <td>{p.gender === 'male' ? 'Erkak' : 'Ayol'}</td>
                    <td style={{ color: 'var(--text2)' }}>{p.address}</td>
                    <td>
                      {p.risk_level
                        ? <RiskBadge level={p.risk_level} />
                        : <span style={{ color: 'var(--text3)' }}>—</span>}
                    </td>
                    <td style={{ fontFamily: 'DM Mono, monospace', fontSize: 13 }}>
                      {p.risk_score != null
                        ? `${(p.risk_score * 100).toFixed(1)}%`
                        : '—'}
                    </td>
                    <td style={{ color: 'var(--text3)', fontSize: 12.5 }}>
                      {p.created_at ? new Date(p.created_at).toLocaleDateString('uz-UZ') : '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>

            {/* Pagination */}
            <div className="pagination">
              <button disabled={page === 1} onClick={() => setPage(p => p - 1)}>← Oldingi</button>
              <span>{page} / {data.pages}</span>
              <button disabled={page >= data.pages} onClick={() => setPage(p => p + 1)}>Keyingi →</button>
            </div>
          </>
        )}
      </div>

      {selected && <PatientModal patient={selected} onClose={() => setSelected(null)} />}
    </>
  );
}
