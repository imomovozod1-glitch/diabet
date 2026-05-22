import { useEffect, useState, useCallback } from 'react';
import { api } from '../api';
import { useLang } from '../i18n';

function RiskBadge({ level }) {
  const { t } = useLang();
  return (
    <span className={`risk-badge risk-${level}`}>
      {t(`risk.${level}`) || level}
    </span>
  );
}

function PatientModal({ patient, onClose }) {
  const { t } = useLang();
  const [detail, setDetail]   = useState(null);
  const [loading, setLoading] = useState(true);
  const [hasError, setError]  = useState(false);

  useEffect(() => {
    api.getPatient(patient.id)
      .then(r => setDetail(r.data))
      .catch(() => setError(true))
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
              {patient.age} {t('common.years')} · {patient.gender === 'male' ? t('common.male') : t('common.female')} · {patient.address}
            </span>
          </div>
          <button
            onClick={onClose}
            style={{ background: 'none', border: 'none', color: 'var(--text3)', fontSize: 20, cursor: 'pointer' }}
          >✕</button>
        </div>

        {loading ? (
          <div className="loading">{t('common.loading')}</div>
        ) : hasError ? (
          <div className="loading" style={{ color: 'var(--red)' }}>{t('patients.modal.errorLoad')}</div>
        ) : (
          <>
            {detail.risk_history?.length > 0 && (
              <div style={{ marginBottom: 20 }}>
                <div className="card-title">{t('patients.modal.riskAssessment')}</div>
                {detail.risk_history.slice(0, 1).map((r, i) => (
                  <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 12, padding: '12px 0' }}>
                    <RiskBadge level={r.risk_level} />
                    <span style={{ fontSize: 13, color: 'var(--text2)' }}>
                      {t('patients.modal.riskScore')}: <b style={{ color: 'var(--text)' }}>{(r.risk_score * 100).toFixed(1)}%</b>
                    </span>
                  </div>
                ))}
                <p style={{ fontSize: 12.5, color: 'var(--text2)', lineHeight: 1.6 }}>
                  {t(`predict.recs.${detail.risk_history[0]?.risk_level}`)}
                </p>
              </div>
            )}

            {detail.test_results?.length > 0 && (
              <div>
                <div className="card-title" style={{ marginBottom: 10 }}>{t('patients.modal.testResults')}</div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                  {detail.test_results.slice(0, 12).map((r, i) => (
                    <div key={i} style={{
                      background: 'var(--bg3)', borderRadius: 7, padding: '8px 12px',
                      display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                    }}>
                      <span style={{ fontSize: 12, color: 'var(--text2)' }}>
                        {t(`biomarkers.${r.code?.toLowerCase()}`, r.biomarker)}
                      </span>
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
  const { t } = useLang();
  const [data, setData]         = useState({ data: [], total: 0, pages: 1 });
  const [loading, setLoading]   = useState(true);
  const [hasError, setHasError] = useState(false);
  const [page, setPage]         = useState(1);
  const [search, setSearch]     = useState('');
  const [gender, setGender]     = useState('');
  const [riskLevel, setRisk]    = useState('');
  const [selected, setSelected] = useState(null);

  const load = useCallback(() => {
    setLoading(true);
    setHasError(false);
    api.getPatients({ page, limit: 20, search, gender, risk_level: riskLevel })
      .then(r => setData(r.data))
      .catch(() => setHasError(true))
      .finally(() => setLoading(false));
  }, [page, search, gender, riskLevel]);

  useEffect(() => { load(); }, [load]);

  const handleSearch = e => { setSearch(e.target.value); setPage(1); };

  return (
    <>
      <div className="page-header">
        <h2>{t('patients.title')}</h2>
        <p>{t('patients.subtitle', { count: data.total.toLocaleString() })}</p>
      </div>

      <div className="filter-bar">
        <input
          placeholder={t('patients.searchPlaceholder')}
          value={search}
          onChange={handleSearch}
          style={{ flex: 1, minWidth: 200 }}
        />
        <select value={gender} onChange={e => { setGender(e.target.value); setPage(1); }}>
          <option value="">{t('common.allGenders')}</option>
          <option value="male">{t('common.male')}</option>
          <option value="female">{t('common.female')}</option>
        </select>
        <select value={riskLevel} onChange={e => { setRisk(e.target.value); setPage(1); }}>
          <option value="">{t('common.allRisk')}</option>
          <option value="very_low">{t('risk.very_low')}</option>
          <option value="low">{t('risk.low')}</option>
          <option value="moderate">{t('risk.moderate')}</option>
          <option value="high">{t('risk.high')}</option>
          <option value="very_high">{t('risk.very_high')}</option>
        </select>
        <button className="btn btn-outline" onClick={load}>{t('common.refresh')}</button>
      </div>

      <div className="table-wrap">
        {loading ? (
          <div className="loading">{t('common.loading')}</div>
        ) : hasError ? (
          <div className="loading" style={{ color: 'var(--red)' }}>{t('common.errorApi')}</div>
        ) : data.data.length === 0 ? (
          <div className="empty">{t('patients.noResults')}</div>
        ) : (
          <>
            <table>
              <thead>
                <tr>
                  <th>{t('patients.colName')}</th>
                  <th>{t('patients.colAge')}</th>
                  <th>{t('patients.colGender')}</th>
                  <th>{t('patients.colAddress')}</th>
                  <th>{t('patients.colRiskLevel')}</th>
                  <th>{t('patients.colRiskScore')}</th>
                  <th>{t('patients.colDate')}</th>
                </tr>
              </thead>
              <tbody>
                {data.data.map(p => (
                  <tr key={p.id} onClick={() => setSelected(p)}>
                    <td style={{ fontWeight: 500 }}>{p.first_name} {p.last_name}</td>
                    <td>{p.age}</td>
                    <td>{p.gender === 'male' ? t('common.male') : t('common.female')}</td>
                    <td style={{ color: 'var(--text2)' }}>{p.address}</td>
                    <td>
                      {p.risk_level
                        ? <RiskBadge level={p.risk_level} />
                        : <span style={{ color: 'var(--text3)' }}>{t('common.noData')}</span>}
                    </td>
                    <td style={{ fontFamily: 'DM Mono, monospace', fontSize: 13 }}>
                      {p.risk_score != null ? `${(p.risk_score * 100).toFixed(1)}%` : t('common.noData')}
                    </td>
                    <td style={{ color: 'var(--text3)', fontSize: 12.5 }}>
                      {p.created_at ? new Date(p.created_at).toLocaleDateString() : t('common.noData')}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>

            <div className="pagination">
              <button disabled={page === 1} onClick={() => setPage(p => p - 1)}>
                {t('common.prevPage')}
              </button>
              <span>{page} / {data.pages}</span>
              <button disabled={page >= data.pages} onClick={() => setPage(p => p + 1)}>
                {t('common.nextPage')}
              </button>
            </div>
          </>
        )}
      </div>

      {selected && <PatientModal patient={selected} onClose={() => setSelected(null)} />}
    </>
  );
}
