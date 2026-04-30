/**
 * AstraMed Assist - API Utilities
 * Communicates with FastAPI backend at /predict and /generate-report
 */

const BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

/**
 * Analyze a chest X-ray image.
 * @param {File} file - The image file to analyze
 * @param {number} mcPasses - Monte Carlo Dropout passes (default 20)
 * @param {number} threshold - Detection threshold (default 0.5)
 * @returns {Promise<Object>} Analysis results
 */
export async function analyzeXray(file, mcPasses = 20, threshold = 0.5) {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('mc_passes', mcPasses);
  formData.append('threshold', threshold);

  const response = await fetch(`${BASE_URL}/predict`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.detail || `Server error: ${response.status}`);
  }

  return response.json();
}

/**
 * Generate a clinical PDF report.
 * @param {File} file - The X-ray image file
 * @param {Object} patientInfo - Patient metadata
 * @returns {Promise<Blob>} PDF blob
 */
export async function generateReport(file, patientInfo = {}) {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('patient_name', patientInfo.name || 'Anonymous');
  formData.append('patient_id', patientInfo.patient_id || '');
  formData.append('age', patientInfo.age || '');
  formData.append('gender', patientInfo.gender || '');
  formData.append('referring_physician', patientInfo.referring_physician || '');
  formData.append('clinical_indication', patientInfo.clinical_indication || 'Routine screening');

  const response = await fetch(`${BASE_URL}/generate-report`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.detail || `Report generation failed: ${response.status}`);
  }

  return response.blob();
}

/**
 * Check API health.
 */
export async function checkHealth() {
  const response = await fetch(`${BASE_URL}/health`);
  return response.json();
}
