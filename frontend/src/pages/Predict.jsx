import React, { useState } from 'react';
import { api } from '../api';

const RISK_LABELS = {
  very_low: 'Juda past', low: 'Past', moderate: "O'rta",
  high: 'Yuqori', very_high: 'Juda yuqori',
};

const RISK_COLORS = {
  very_low: 'var(--green)', low: 'var(--accent)',
  moderate: 'var(--yellow)', high: 'var(--orange)', very_high: 'var(--red)',
};

export default function Predict() {
  const [form, setForm] = useState({
    age: '', gender: 'male',
    hba1c: '', fasting_glucose: '', bmi: '',
    systolic_bp: '', diastolic_bp: '',
    triglycerides: '', hdl: '',
    homa_ir: '', crp: '',
    family_history: false,
  });

  const [result, setResult]   = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState(null);

  const set = (k, v) => setForm(f => ({ ...f, [k]: v }));

  const handleSubmit = () => {
    setError(null);
    if (!form.age || !form.hba1c || !form.fasting_glucose || !form.bmi) {
      setError("Majburiy maydonlarni to'ldiring: Yosh, HbA1c, Glucose, BMI");
      return;
    }
    const payload = {
      age:              Number(form.age),
      gender:           form.gender,
      hba1c:            Number(form.hba1c),
      fasting_glucose:  Number(form.fasting_glucose),
      bmi:              Number(form.bmi),
      family_history:   form.family_history,
      ...(form.systolic_bp    && { systolic_bp:   Number(form.systolic_bp) }),
      ...(form.diastolic_bp   && { diastolic_bp:  Number(form.diastolic_bp) }),
      ...(form.triglycerides  && { triglycerides: Number(form.triglycerides) }),
      ...(form.hdl            && { hdl:           Number(form.hdl) }),
      ...(form.homa_ir        && { homa_ir:       Number(form.homa_ir) }),
      ...(form.crp            && { crp:           Number(form.crp) }),
    };
    setLoading(true);
    api.predict(payload)
      .then(r => setResult(r.data))
      .catch(() => setError("API xatosi. FastAPI ishlamoqdami?"))
      .finally(() => setLoading(false));
  };

  return (
    <>
      <div className="page-header">
        <h2>Diabet xavfini bashorat qilish</h2>
        <p>Bemor tahlil natijalarini kiriting</p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, alignItems: 'start' }}>

        {/* Form */}
        <div className="card">
          <div className="card-title" style={{ marginBottom: 16 }}>Bemor ma'lumotlari</div>

          <div className="form-grid">
            <div className="form-group">
              <label>Yosh *</label>
              <input type="number" placeholder="45" value={form.age}
                onChange={e => set('age', e.target.value)} min={1} max={120} />
            </div>
            <div className="form-group">
              <label>Jins *</label>
              <select value={form.gender} onChange={e => set('gender', e.target.value)}>
                <option value="male">Erkak</option>
                <option value="female">Ayol</option>
              </select>
            </div>
          </div>

          <div style={{ margin: '14px 0 4px', fontSize: 11, color: 'var(--text3)', textTransform: 'uppercase', letterSpacing: '0.6px', fontWeight: 600 }}>
            Asosiy ko'rsatkichlar *
          </div>
          <div className="form-grid" style={{ marginBottom: 14 }}>
            <div className="form-group">
              <label>HbA1c (%)</label>
              <input type="number" placeholder="5.8" step="0.1"
                value={form.hba1c} onChange={e => set('hba1c', e.target.value)} />
            </div>
            <div className="form-group">
              <label>Fasting Glucose (mg/dL)</label>
              <input type="number" placeholder="108"
                value={form.fasting_glucose} onChange={e => set('fasting_glucose', e.target.value)} />
            </div>
            <div className="form-group">
              <label>BMI (kg/m²)</label>
              <input type="number" placeholder="28.5" step="0.1"
                value={form.bmi} onChange={e => set('bmi', e.target.value)} />
            </div>
            <div className="form-group">
              <label>HOMA-IR</label>
              <input type="number" placeholder="3.1" step="0.1"
                value={form.homa_ir} onChange={e => set('homa_ir', e.target.value)} />
            </div>
          </div>

          <div style={{ margin: '0 0 4px', fontSize: 11, color: 'var(--text3)', textTransform: 'uppercase', letterSpacing: '0.6px', fontWeight: 600 }}>
            Qo'shimcha ko'rsatkichlar (ixtiyoriy)
          </div>
          <div className="form-grid" style={{ marginBottom: 14 }}>
            <div className="form-group">
              <label>Sistolik BP (mmHg)</label>
              <input type="number" placeholder="128"
                value={form.systolic_bp} onChange={e => set('systolic_bp', e.target.value)} />
            </div>
            <div className="form-group">
              <label>Diastolik BP (mmHg)</label>
              <input type="number" placeholder="82"
                value={form.diastolic_bp} onChange={e => set('diastolic_bp', e.target.value)} />
            </div>
            <div className="form-group">
              <label>Trigliseridlar (mg/dL)</label>
              <input type="number" placeholder="165"
                value={form.triglycerides} onChange={e => set('triglycerides', e.target.value)} />
            </div>
            <div className="form-group">
              <label>HDL (mg/dL)</label>
              <input type="number" placeholder="46"
                value={form.hdl} onChange={e => set('hdl', e.target.value)} />
            </div>
            <div className="form-group">
              <label>CRP (mg/L)</label>
              <input type="number" placeholder="1.8" step="0.1"
                value={form.crp} onChange={e => set('crp', e.target.value)} />
            </div>
            <div className="form-group" style={{ justifyContent: 'flex-end' }}>
              <label>Oilaviy tarix</label>
              <label style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer',
                background: 'var(--bg3)', border: '1px solid var(--border)', borderRadius: 7,
                padding: '9px 12px', fontSize: 13.5 }}>
                <input type="checkbox" checked={form.family_history}
                  onChange={e => set('family_history', e.target.checked)} />
                Oilada diabet bor
              </label>
            </div>
          </div>

          {error && (
            <div style={{ background: 'rgba(248,81,73,0.1)', border: '1px solid rgba(248,81,73,0.3)',
              borderRadius: 7, padding: '10px 14px', fontSize: 13, color: 'var(--red)', marginBottom: 14 }}>
              ⚠ {error}
            </div>
          )}

          <button
            className="btn btn-primary"
            style={{ width: '100%' }}
            onClick={handleSubmit}
            disabled={loading}
          >
            {loading ? 'Hisoblanmoqda...' : '🔬 Xavfni baholash'}
          </button>
        </div>

        {/* Result */}
        <div>
          {!result ? (
            <div className="card" style={{ textAlign: 'center', padding: '40px 24px' }}>
              <div style={{ fontSize: 40, marginBottom: 12 }}>🩺</div>
              <p style={{ color: 'var(--text3)', fontSize: 13.5 }}>
                Ma'lumotlarni kiriting va<br />natijani ko'ring
              </p>
            </div>
          ) : (
            <div className="result-card">
              {/* Asosiy natija */}
              <div style={{ textAlign: 'center', marginBottom: 24 }}>
                <div className="risk-score-big" style={{ color: RISK_COLORS[result.risk_level] }}>
                  {(result.risk_score * 100).toFixed(1)}%
                </div>
                <div style={{ marginTop: 10 }}>
                  <span className={`risk-badge risk-${result.risk_level}`} style={{ fontSize: 13 }}>
                    {RISK_LABELS[result.risk_level]}
                  </span>
                </div>
              </div>

              {/* Tavsiya */}
              <div style={{
                background: 'var(--bg3)', borderRadius: 8, padding: '12px 16px',
                marginBottom: 20, fontSize: 13.5, lineHeight: 1.7,
                borderLeft: `3px solid ${RISK_COLORS[result.risk_level]}`,
              }}>
                {result.recommendation}
              </div>

              {/* Top omillar */}
              {result.top_factors?.length > 0 && (
                <>
                  <div className="card-title" style={{ marginBottom: 8 }}>Ta'sir qilgan omillar</div>
                  {result.top_factors.map((f, i) => (
                    <div key={i} className="factor-bar">
                      <span className="factor-name">{f.name}</span>
                      <span style={{ fontSize: 13, fontFamily: 'DM Mono, monospace',
                        color: 'var(--text2)', marginRight: 8 }}>
                        {typeof f.value === 'number' ? f.value.toFixed(1) : f.value}
                      </span>
                      <span className={`factor-impact impact-${f.impact.replace("'", "")}`}>
                        {f.impact}
                      </span>
                      <div className="bar-track">
                        <div className="bar-fill"
                          style={{ width: `${Math.min(f.weight * 300, 100)}%` }} />
                      </div>
                    </div>
                  ))}
                </>
              )}

              <button
                className="btn btn-outline"
                style={{ width: '100%', marginTop: 16 }}
                onClick={() => setResult(null)}
              >
                Yangi tekshiruv
              </button>
            </div>
          )}
        </div>
      </div>
    </>
  );
}
