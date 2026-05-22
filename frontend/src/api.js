import axios from 'axios';

const API = axios.create({
  baseURL: process.env.REACT_APP_API_URL || 'http://localhost:8001',
  timeout: 15000,
});

export const api = {
  // Health
  health: () => API.get('/health'),

  // Dashboard
  getStats:   () => API.get('/dashboard/stats'),
  getTrend:   () => API.get('/dashboard/trend'),

  // Patients
  getPatients: (params) => API.get('/patients', { params }),
  getPatient:  (id)     => API.get(`/patients/${id}`),

  // Risk
  getRisk:     (patientId) => API.get(`/risk/${patientId}`),

  // Predict
  predict:     (data)      => API.post('/predict', data),
};
