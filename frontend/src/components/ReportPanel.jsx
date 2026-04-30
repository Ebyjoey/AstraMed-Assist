import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { FileText, Download, RefreshCw, CheckCircle, Clock } from 'lucide-react';
import TriageBadge from './TriageBadge';

export default function ReportPanel({ results, patientInfo, reportUrl, isGenerating, onGenerateReport }) {
  const hasResults = !!results;
  const hasReport  = !!reportUrl;

  return (
    <div>
      <div className="panel-title">
        <FileText size={13} />
        Clinical Report
      </div>

      {/* ── Report Preview ── */}
      <AnimatePresence mode="wait">
        {!hasResults ? (
          <motion.div
            key="empty"
            className="report-preview-placeholder"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
          >
            <FileText size={36} color="var(--gray-300)" />
            <p style={{ fontSize: 13, color: 'var(--gray-400)', fontWeight: 500 }}>No results yet</p>
            <p style={{ fontSize: 11, color: 'var(--gray-400)', textAlign: 'center', maxWidth: 180 }}>
              Complete analysis to enable report generation
            </p>
          </motion.div>
        ) : (
          <motion.div
            key="preview"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="report-preview"
          >
            {/* Header */}
            <div className="report-section">
              <div className="report-section-title">PATIENT INFORMATION</div>
              <ReportRow k="Name"      v={patientInfo.name        || '—'} />
              <ReportRow k="ID"        v={patientInfo.patient_id  || '—'} />
              <ReportRow k="Age"       v={patientInfo.age         || '—'} />
              <ReportRow k="Gender"    v={patientInfo.gender      || '—'} />
              <ReportRow k="Physician" v={patientInfo.referring_physician || '—'} />
            </div>

            {/* Triage */}
            <div className="report-section">
              <div className="report-section-title">TRIAGE ASSESSMENT</div>
              <div style={{ marginBottom: 8 }}>
                <TriageBadge level={results.triage?.triage_level} score={results.triage?.triage_score} />
              </div>
              <ReportRow k="Primary Finding" v={results.triage?.primary_finding || '—'} />
              <ReportRow k="Severity"        v={results.triage?.severity_label  || '—'} />
              <ReportRow k="Confidence"      v={`${((results.triage?.confidence || 0) * 100).toFixed(1)}%`} />
            </div>

            {/* Probabilities */}
            <div className="report-section">
              <div className="report-section-title">DISEASE PROBABILITIES</div>
              {Object.entries(results.probabilities || {}).map(([name, prob]) => (
                <ReportRow
                  key={name}
                  k={name}
                  v={`${(prob * 100).toFixed(1)}%`}
                  highlight={prob >= 0.5 && name !== 'Normal'}
                />
              ))}
            </div>

            {/* Severity */}
            <div className="report-section">
              <div className="report-section-title">SEVERITY SCORES</div>
              {Object.entries(results.severities || {}).map(([name, sev]) => (
                <ReportRow key={name} k={name} v={`${(sev * 100).toFixed(1)}%`} />
              ))}
              <ReportRow k="Uncertainty" v={(results.uncertainty?.value || 0).toFixed(4)} />
            </div>

            {/* Model */}
            <div className="report-section">
              <div className="report-section-title">MODEL INFORMATION</div>
              <ReportRow k="Backbone"   v={results.metadata?.model_backbone || 'DenseNet-121'} />
              <ReportRow k="Device"     v={results.metadata?.device || 'CPU'} />
              <ReportRow k="MC Passes"  v={results.metadata?.mc_passes || 20} />
              <ReportRow k="Timestamp"  v={results.metadata?.timestamp ? new Date(results.metadata.timestamp).toLocaleTimeString() : '—'} />
            </div>

            {/* Disclaimer */}
            <div style={{ background: 'var(--amber-50)', border: '1px solid #FCD34D', borderRadius: 6, padding: '8px 10px', fontSize: 10, color: 'var(--amber-600)', lineHeight: 1.5 }}>
              ⚠ AI-generated. Not a medical diagnosis. Requires radiologist review.
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <div className="divider" />

      {/* ── Generate Button ── */}
      <motion.button
        className="btn btn-primary"
        style={{ marginBottom: 8 }}
        onClick={onGenerateReport}
        disabled={!hasResults || isGenerating}
        whileTap={{ scale: 0.98 }}
      >
        {isGenerating ? (
          <>
            <RefreshCw size={14} style={{ animation: 'spin 1s linear infinite' }} />
            Generating PDF...
          </>
        ) : (
          <>
            <FileText size={14} />
            Generate PDF Report
          </>
        )}
      </motion.button>

      {/* ── Download Button ── */}
      <AnimatePresence>
        {hasReport && (
          <motion.div
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
          >
            <a
              href={reportUrl}
              download={`AstraMed_Report_${patientInfo.name || 'Patient'}_${new Date().toISOString().slice(0, 10)}.pdf`}
              style={{ textDecoration: 'none', display: 'block' }}
            >
              <button className="btn btn-outline" style={{ width: '100%', color: 'var(--green-600)', borderColor: 'var(--green-500)', background: 'var(--green-50)' }}>
                <Download size={14} />
                Download PDF Report
              </button>
            </a>

            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: 10, fontSize: 11, color: 'var(--green-600)' }}
            >
              <CheckCircle size={13} />
              Report ready for download
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ── Report Info ── */}
      <div className="divider" />
      <div style={{ fontSize: 11, color: 'var(--gray-500)', lineHeight: 1.7 }}>
        <div style={{ display: 'flex', gap: 6, marginBottom: 4 }}>
          <Clock size={12} style={{ marginTop: 2, flexShrink: 0 }} />
          <span>Report includes: predictions, heatmap, severity, triage, clinical summary & disclaimer</span>
        </div>
        <div style={{ display: 'flex', gap: 6 }}>
          <FileText size={12} style={{ marginTop: 2, flexShrink: 0 }} />
          <span>Format: A4 PDF · DICOM SR compatible</span>
        </div>
      </div>
    </div>
  );
}

function ReportRow({ k, v, highlight }) {
  return (
    <div className="report-row">
      <span className="rk">{k}</span>
      <span className="rv" style={{ color: highlight ? 'var(--red-600)' : undefined }}>{v}</span>
    </div>
  );
}
