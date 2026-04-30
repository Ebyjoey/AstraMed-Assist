import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { motion, AnimatePresence } from 'framer-motion';
import { Upload, X, Scan, RotateCcw, User, FileImage } from 'lucide-react';

export default function UploadPanel({
  imageFile, imagePreview, patientInfo,
  onImageDrop, onPatientInfoChange, onAnalyze, onReset, isAnalyzing
}) {
  const [dragActive, setDragActive] = useState(false);

  const onDrop = useCallback((accepted) => {
    if (accepted.length > 0) onImageDrop(accepted[0]);
    setDragActive(false);
  }, [onImageDrop]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/*': ['.jpg', '.jpeg', '.png', '.bmp', '.tiff'] },
    multiple: false,
    onDragEnter: () => setDragActive(true),
    onDragLeave: () => setDragActive(false),
  });

  const handleField = (key, val) =>
    onPatientInfoChange(prev => ({ ...prev, [key]: val }));

  return (
    <div>
      {/* Upload Section */}
      <div className="panel-title">
        <FileImage size={13} />
        X-Ray Upload
      </div>

      <AnimatePresence mode="wait">
        {!imageFile ? (
          <motion.div
            key="dropzone"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            {...getRootProps()}
            className={`dropzone-area ${isDragActive || dragActive ? 'active' : ''}`}
          >
            <input {...getInputProps()} />
            <Upload size={36} className="dropzone-icon" />
            <p className="dropzone-text">
              {isDragActive ? 'Drop X-ray here...' : 'Drag & drop chest X-ray'}
            </p>
            <p className="dropzone-sub">JPEG · PNG · BMP · TIFF</p>
            <p className="dropzone-sub">or click to browse files</p>
          </motion.div>
        ) : (
          <motion.div
            key="preview"
            initial={{ opacity: 0, scale: 0.97 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0 }}
          >
            <div className="dropzone-area has-file" style={{ padding: '12px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, width: '100%' }}>
                <FileImage size={20} color="var(--green-600)" />
                <div style={{ flex: 1, minWidth: 0 }}>
                  <p style={{ fontSize: 12, fontWeight: 600, color: 'var(--green-600)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {imageFile.name}
                  </p>
                  <p style={{ fontSize: 11, color: 'var(--gray-500)' }}>
                    {(imageFile.size / 1024).toFixed(1)} KB · Ready for analysis
                  </p>
                </div>
                <button
                  onClick={(e) => { e.stopPropagation(); onReset(); }}
                  style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--gray-400)', padding: '2px' }}
                >
                  <X size={16} />
                </button>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <div className="divider" />

      {/* Patient Information */}
      <div className="panel-title">
        <User size={13} />
        Patient Information
      </div>

      <div className="form-group">
        <label className="form-label">Full Name</label>
        <input
          className="form-input"
          placeholder="e.g. Priya Sharma"
          value={patientInfo.name}
          onChange={e => handleField('name', e.target.value)}
        />
      </div>

      <div className="form-row">
        <div className="form-group">
          <label className="form-label">Age</label>
          <input
            className="form-input"
            placeholder="e.g. 42"
            value={patientInfo.age}
            onChange={e => handleField('age', e.target.value)}
          />
        </div>
        <div className="form-group">
          <label className="form-label">Gender</label>
          <select
            className="form-select"
            value={patientInfo.gender}
            onChange={e => handleField('gender', e.target.value)}
          >
            <option value="">Select</option>
            <option value="Male">Male</option>
            <option value="Female">Female</option>
            <option value="Other">Other</option>
          </select>
        </div>
      </div>

      <div className="form-group">
        <label className="form-label">Patient ID</label>
        <input
          className="form-input"
          placeholder="e.g. PT-2026-0042"
          value={patientInfo.patient_id}
          onChange={e => handleField('patient_id', e.target.value)}
        />
      </div>

      <div className="form-group">
        <label className="form-label">Referring Physician</label>
        <input
          className="form-input"
          placeholder="e.g. Dr. Ramesh Kumar"
          value={patientInfo.referring_physician}
          onChange={e => handleField('referring_physician', e.target.value)}
        />
      </div>

      <div className="form-group">
        <label className="form-label">Clinical Indication</label>
        <input
          className="form-input"
          placeholder="e.g. Persistent cough, fever"
          value={patientInfo.clinical_indication}
          onChange={e => handleField('clinical_indication', e.target.value)}
        />
      </div>

      <div className="divider" />

      {/* Action Buttons */}
      <motion.button
        className="btn btn-primary btn-lg"
        onClick={onAnalyze}
        disabled={!imageFile || isAnalyzing}
        whileTap={{ scale: 0.98 }}
        style={{ marginBottom: 8 }}
      >
        {isAnalyzing ? (
          <>
            <span className="loader-ring" style={{ width: 16, height: 16, borderWidth: 2 }} />
            Analyzing...
          </>
        ) : (
          <>
            <Scan size={16} />
            Analyze X-Ray
          </>
        )}
      </motion.button>

      {imageFile && (
        <motion.button
          className="btn btn-outline"
          onClick={onReset}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
        >
          <RotateCcw size={14} />
          Reset
        </motion.button>
      )}

      {/* Info Note */}
      <div className="alert alert-info" style={{ marginTop: 14 }}>
        <strong>Model:</strong> DenseNet-121 · 20K chest X-rays ·
        Accuracy 90.8% · AUC 0.976
      </div>
    </div>
  );
}
