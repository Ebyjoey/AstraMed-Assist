import React, { useState, useCallback } from 'react';
import { Toaster } from 'react-hot-toast';
import { motion, AnimatePresence } from 'framer-motion';
import UploadPanel from './components/UploadPanel';
import AnalysisPanel from './components/AnalysisPanel';
import ReportPanel from './components/ReportPanel';
import Loader from './components/Loader';
import { analyzeXray, generateReport } from './utils/api';
import toast from 'react-hot-toast';
import './index.css';

export default function App() {
  const [imageFile, setImageFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [patientInfo, setPatientInfo] = useState({
    name: '', age: '', gender: '', patient_id: '',
    referring_physician: '', clinical_indication: 'Routine chest X-ray screening'
  });
  const [results, setResults] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isGeneratingReport, setIsGeneratingReport] = useState(false);
  const [showHeatmap, setShowHeatmap] = useState(true);
  const [reportUrl, setReportUrl] = useState(null);

  const handleImageDrop = useCallback((file) => {
    setImageFile(file);
    setResults(null);
    setReportUrl(null);
    const url = URL.createObjectURL(file);
    setImagePreview(url);
  }, []);

  const handleAnalyze = async () => {
    if (!imageFile) {
      toast.error('Please upload a chest X-ray image first');
      return;
    }
    setIsAnalyzing(true);
    setResults(null);
    try {
      const data = await analyzeXray(imageFile);
      setResults(data);
      toast.success('Analysis complete!');
    } catch (err) {
      toast.error(err.message || 'Analysis failed. Please try again.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleGenerateReport = async () => {
    if (!imageFile || !results) {
      toast.error('Please run analysis first');
      return;
    }
    setIsGeneratingReport(true);
    try {
      const blob = await generateReport(imageFile, patientInfo);
      const url = URL.createObjectURL(blob);
      setReportUrl(url);
      toast.success('Report generated!');
    } catch (err) {
      toast.error(err.message || 'Report generation failed');
    } finally {
      setIsGeneratingReport(false);
    }
  };

  const handleReset = () => {
    setImageFile(null);
    setImagePreview(null);
    setResults(null);
    setReportUrl(null);
  };

  return (
    <div className="app-shell">
      <Toaster position="top-right" toastOptions={{ duration: 3500 }} />

      {/* ── Top Bar ── */}
      <header className="topbar">
        <div className="topbar-brand">
          <div className="brand-icon">
            <svg width="28" height="28" viewBox="0 0 28 28" fill="none">
              <rect width="28" height="28" rx="8" fill="#1A56DB"/>
              <path d="M14 5v18M5 14h18" stroke="white" strokeWidth="2.5" strokeLinecap="round"/>
              <circle cx="14" cy="14" r="4" fill="none" stroke="white" strokeWidth="1.5"/>
            </svg>
          </div>
          <div>
            <h1 className="brand-title">AstraMed Assist</h1>
            <p className="brand-sub">AI Chest X-Ray Analysis · Triage · Reporting</p>
          </div>
        </div>

        <div className="topbar-stats">
          <Stat label="Accuracy" value="90.8%" />
          <Stat label="AUC" value="0.976" />
          <Stat label="F1 Score" value="91.5%" />
          <Stat label="Dataset" value="20K CXR" />
        </div>

        <div className="topbar-badges">
          <span className="badge badge-blue">DenseNet-121</span>
          <span className="badge badge-gray">Grad-CAM</span>
          <span className="badge badge-gray">MC Dropout</span>
        </div>
      </header>

      {/* ── Main 3-Column Layout ── */}
      <main className="main-grid">
        {/* Column 1: Upload + Patient */}
        <motion.div
          className="col col-left"
          initial={{ opacity: 0, x: -30 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5 }}
        >
          <UploadPanel
            imageFile={imageFile}
            imagePreview={imagePreview}
            patientInfo={patientInfo}
            onImageDrop={handleImageDrop}
            onPatientInfoChange={setPatientInfo}
            onAnalyze={handleAnalyze}
            onReset={handleReset}
            isAnalyzing={isAnalyzing}
          />
        </motion.div>

        {/* Column 2: Analysis Results + Heatmap */}
        <motion.div
          className="col col-center"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.1 }}
        >
          <AnimatePresence mode="wait">
            {isAnalyzing ? (
              <motion.div
                key="loader"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="center-loader"
              >
                <Loader message="Analyzing X-ray with DenseNet-121..." />
              </motion.div>
            ) : (
              <motion.div
                key="analysis"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
              >
                <AnalysisPanel
                  results={results}
                  imagePreview={imagePreview}
                  showHeatmap={showHeatmap}
                  onToggleHeatmap={() => setShowHeatmap(h => !h)}
                />
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>

        {/* Column 3: Report */}
        <motion.div
          className="col col-right"
          initial={{ opacity: 0, x: 30 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
        >
          <ReportPanel
            results={results}
            patientInfo={patientInfo}
            reportUrl={reportUrl}
            isGenerating={isGeneratingReport}
            onGenerateReport={handleGenerateReport}
          />
        </motion.div>
      </main>

      {/* ── Footer ── */}
      <footer className="footer">
        <span>AstraMed Assist v1.0 · Decision-support tool only · Not a medical device</span>
        <span>Based on: Sakthi U. et al., ICAIHC 2026</span>
      </footer>
    </div>
  );
}

function Stat({ label, value }) {
  return (
    <div className="topbar-stat">
      <span className="stat-value">{value}</span>
      <span className="stat-label">{label}</span>
    </div>
  );
}
