import { useState, useCallback, useRef, useEffect } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell,
} from 'recharts';
import { predictImage, fetchClasses, fetchSampleResults } from '../api';
import type { PredictResponse, SinglePredictionResult, ClassInfo } from '../types';
import { SectionHeader, ErrorBanner } from '../components/UI';
import {
  Upload, ImageIcon, Loader2, AlertTriangle, CheckCircle, X,
  Layers, Sparkles, ShieldAlert,
  Info, Eye, Sliders, ExternalLink
} from 'lucide-react';

type ModelSelection = 'both' | 'efficientnetv2s' | 'convnextbase';

const MODEL_OPTIONS: { id: ModelSelection; label: string; badge: string; desc: string }[] = [
  {
    id: 'both',
    label: '⚡ Dual Mode (Side-by-Side Comparison)',
    badge: 'Recommended',
    desc: 'Runs EfficientNetV2-S and ConvNeXtBase simultaneously to compare predictions & Grad-CAM feature focus side-by-side',
  },
  {
    id: 'efficientnetv2s',
    label: 'EfficientNetV2-S (Lightweight Champion)',
    badge: '92.8% Acc · 20.7M params',
    desc: 'Fast, compact model specialized in fine-grained lesion spot attribution',
  },
  {
    id: 'convnextbase',
    label: 'ConvNeXtBase (Modern ConvNet)',
    badge: '90.2% Acc · 87.8M params',
    desc: 'Deep architecture with 7x7 kernels capturing broader contextual plant leaf patterns',
  },
];

const CLASS_COLORS_MAP: Record<string, string> = {
  'Bacterial Leaf Blight': '#eab308', // yellow-500
  'Brown Spot': '#f59e0b',            // amber-500
  'Healthy Rice Leaf': '#16a34a',     // green-600
  'Leaf Blast': '#ef4444',            // red-500
  'Leaf Scald': '#f97316',            // orange-500
  'Sheath Blight': '#8b5cf6',         // purple-500
};

export default function Predict() {
  const [model, setModel] = useState<ModelSelection>('both');
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [classes, setClasses] = useState<ClassInfo[]>([]);

  // Grad-CAM visualization controls
  const [viewMode, setViewMode] = useState<'overlay' | 'raw' | 'original'>('overlay');
  const [opacity, setOpacity] = useState<number>(0.65);

  // Pre-computed sample results (both models) — loaded once at mount
  const [sampleResults, setSampleResults] = useState<Record<string, PredictResponse>>({});
  const [showColabNotice, setShowColabNotice] = useState(false);

  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    fetchClasses().then(setClasses).catch(() => {});
    fetchSampleResults().then(setSampleResults).catch(() => {});
  }, []);

  const handleFile = useCallback((f: File) => {
    if (!f.type.startsWith('image/')) {
      setError('Please select a valid JPEG or PNG image file.');
      return;
    }
    if (f.size > 12 * 1024 * 1024) {
      setError('Image must be smaller than 12 MB.');
      return;
    }
    setFile(f);
    setError(null);
    setResult(null);
    const url = URL.createObjectURL(f);
    setPreview(url);
  }, []);

  const onDrop = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f);
  }, [handleFile]);

  const onInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f) handleFile(f);
  };

  const clearImage = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
    setShowColabNotice(false);
    if (inputRef.current) inputRef.current.value = '';
  };

  const handlePredict = async () => {
    if (!file) return;

    // On local machine (localhost / 127.0.0.1), run inference normally using local models
    const isLocalhost = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';
    if (!isLocalhost) {
      setShowColabNotice(true);
      setResult(null);
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);
    setShowColabNotice(false);
    try {
      const res = await predictImage(file, model);
      setResult(res);
    } catch (e: any) {
      setError(e.message || 'Prediction failed');
    } finally {
      setLoading(false);
    }
  };

  const loadSampleImage = async (cls: ClassInfo) => {
    if (!cls.image) return;
    try {
      setLoading(true);
      setError(null);
      setResult(null);
      const res = await fetch(cls.image);
      const blob = await res.blob();
      const sampleFile = new File([blob], `${cls.name.toLowerCase().replace(/\s+/g, '_')}_sample.jpg`, { type: 'image/jpeg' });
      handleFile(sampleFile);
      setShowColabNotice(false);

      // Show pre-computed result instantly (both models) — no model run needed
      if (sampleResults[cls.name]) {
        setResult(sampleResults[cls.name]);
      }
    } catch (err) {
      setError('Could not load sample image');
    } finally {
      setLoading(false);
    }
  };

  // Helper to build probability chart data
  const getChartData = (scores?: Record<string, number>) => {
    if (!scores) return [];
    return Object.entries(scores)
      .map(([cls, score]) => ({ class: cls, score: +(score * 100).toFixed(1) }))
      .sort((a, b) => b.score - a.score);
  };

  // Find info about a disease class
  const getClassDetails = (className?: string) => {
    if (!className) return null;
    return classes.find((c) => c.name === className);
  };

  return (
    <div className="space-y-8 max-w-7xl mx-auto pb-12">
      <SectionHeader
        title="Rice Leaf Disease Prediction & Grad-CAM Visualizer"
        subtitle="Compare predictions and explainable AI attention heatmaps across EfficientNetV2-S and ConvNeXtBase"
      />

      {/* Model Selection Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {MODEL_OPTIONS.map((opt) => {
          const isSelected = model === opt.id;
          return (
            <button
              key={opt.id}
              onClick={() => { setModel(opt.id); setResult(null); }}
              className={`text-left p-4 rounded-2xl border-2 transition-all relative flex flex-col justify-between ${
                isSelected
                  ? 'border-green-600 bg-green-50/60 shadow-md ring-2 ring-green-500/20'
                  : 'border-slate-200 bg-white hover:border-slate-300 hover:bg-slate-50'
              }`}
            >
              <div>
                <div className="flex items-center justify-between gap-2 mb-2">
                  <span className={`text-sm font-bold ${isSelected ? 'text-green-900' : 'text-slate-800'}`}>
                    {opt.label}
                  </span>
                  <span className={`text-xs px-2 py-0.5 rounded-full font-medium shrink-0 ${
                    isSelected ? 'bg-green-700 text-white' : 'bg-slate-100 text-slate-600'
                  }`}>
                    {opt.badge}
                  </span>
                </div>
                <p className="text-xs text-slate-500 leading-relaxed">{opt.desc}</p>
              </div>
            </button>
          );
        })}
      </div>

      {/* Main Grid: Upload & Inspection */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Left Column: Input Image & Sample Gallery (5 cols) */}
        <div className="lg:col-span-5 space-y-4">
          <div className="bg-white border border-slate-200 rounded-2xl p-5 shadow-sm space-y-4">
            <h3 className="font-semibold text-slate-800 text-sm flex items-center gap-2">
              <ImageIcon className="w-4 h-4 text-green-700" />
              1. Input Leaf Image
            </h3>

            {/* Dropzone / Preview */}
            {!preview ? (
              <div
                onDrop={onDrop}
                onDragOver={(e) => e.preventDefault()}
                onClick={() => inputRef.current?.click()}
                className="border-2 border-dashed border-slate-300 hover:border-green-500 rounded-2xl aspect-square flex flex-col items-center justify-center gap-4 cursor-pointer bg-slate-50/70 hover:bg-green-50/40 transition-all group p-6"
              >
                <div className="w-16 h-16 rounded-full bg-white border border-slate-200 flex items-center justify-center group-hover:border-green-400 group-hover:scale-105 transition-all shadow-sm">
                  <Upload className="w-8 h-8 text-slate-400 group-hover:text-green-600" />
                </div>
                <div className="text-center">
                  <p className="text-slate-700 font-semibold text-sm">Drag & drop rice leaf image here</p>
                  <p className="text-slate-400 text-xs mt-1">or click to browse local files · JPEG / PNG up to 12MB</p>
                </div>
              </div>
            ) : (
              <div className="relative rounded-2xl overflow-hidden border border-slate-200 aspect-square bg-slate-900 group">
                <img src={preview} alt="Input leaf" className="w-full h-full object-contain" />
                <button
                  onClick={clearImage}
                  title="Remove image"
                  className="absolute top-3 right-3 w-8 h-8 bg-black/60 hover:bg-red-600 text-white rounded-full flex items-center justify-center shadow-md transition-colors"
                >
                  <X className="w-4 h-4" />
                </button>
                {file && (
                  <div className="absolute bottom-3 left-3 bg-black/75 backdrop-blur-sm text-white text-xs px-3 py-1.5 rounded-lg">
                    {file.name} · {(file.size / 1024).toFixed(0)} KB
                  </div>
                )}
              </div>
            )}

            <input
              ref={inputRef}
              type="file"
              accept="image/jpeg,image/jpg,image/png"
              className="hidden"
              onChange={onInputChange}
            />

            {error && <ErrorBanner message={error} />}

            {showColabNotice && (
              <div className="bg-amber-50 border border-amber-200 rounded-2xl p-4 text-left space-y-3">
                <div className="flex items-start gap-3">
                  <AlertTriangle className="w-5 h-5 text-amber-600 shrink-0 mt-0.5" />
                  <div>
                    <h5 className="font-bold text-sm text-amber-900">
                      การประมวลผลโมเดลสดถูกจำกัดบนเว็บสาธิต (Free Cloud Limit)
                    </h5>
                    <p className="text-xs text-amber-800 mt-1 leading-relaxed">
                      เนื่องจากทั้ง 2 โมเดล (รวมกันกว่า 421 MB) และการคำนวณ Grad-CAM ใช้หน่วยความจำ (RAM) สูงเกินขีดจำกัดของ Free Cloud Hosting ระบบจึงแสดงผลการสาธิตผ่าน <strong>1-Click Sample Images</strong> ด้านล่าง
                    </p>
                    <p className="text-xs text-amber-800 mt-1.5 leading-relaxed">
                      หากท่านต้องการอัปโหลดรูปภาพใหม่เพื่อทดสอบการทำนายสดด้วยตนเอง สามารถรันโค้ดและโมเดลฉบับเต็มบน Google Colab (GPU ฟรี) ได้ที่ลิงก์ด้านล่างนี้:
                    </p>
                  </div>
                </div>
                <a
                  href="https://colab.research.google.com/drive/197JoUaecxXV8inLMMK1PQWrZhoENJEOn?usp=sharing"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center justify-center gap-2 w-full py-2.5 px-4 bg-amber-600 hover:bg-amber-700 text-white font-semibold text-xs rounded-xl shadow-sm transition-colors"
                >
                  <ExternalLink className="w-4 h-4" />
                  เปิดทดสอบการทำนายบน Google Colab
                </a>
              </div>
            )}

            {/* Predict Button */}
            <button
              onClick={handlePredict}
              disabled={!file || loading}
              className={`w-full py-3.5 px-4 rounded-xl font-bold text-sm transition-all flex items-center justify-center gap-2 shadow-sm ${
                !file || loading
                  ? 'bg-slate-200 text-slate-400 cursor-not-allowed'
                  : 'bg-green-700 hover:bg-green-800 text-white hover:shadow-md active:scale-[0.99]'
              }`}
            >
              {loading ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Generating Grad-CAM Heatmaps & Inference…
                </>
              ) : (
                <>
                  <Sparkles className="w-4 h-4" />
                  {model === 'both' ? 'Run Dual Model Comparison & Grad-CAM' : `Classify with ${model === 'efficientnetv2s' ? 'EfficientNetV2-S' : 'ConvNeXtBase'}`}
                </>
              )}
            </button>
          </div>

          {/* Sample Images Quick Picker */}
          <div className="bg-white border border-slate-200 rounded-2xl p-4 shadow-sm">
            <p className="text-xs font-semibold text-slate-700 uppercase tracking-wider mb-2">
              Or Try A Sample Leaf Image (1-Click Test)
            </p>
            <div className="grid grid-cols-3 gap-2">
              {classes.map((c) => (
                <button
                  key={c.name}
                  onClick={() => loadSampleImage(c)}
                  className="p-2 border border-slate-200 hover:border-green-400 rounded-xl bg-slate-50 hover:bg-green-50 text-left transition-all flex flex-col items-center gap-1.5 group"
                >
                  <div className="w-full aspect-square rounded-lg overflow-hidden bg-slate-200 relative">
                    {c.image ? (
                      <img src={c.image} alt={c.name} className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-300" />
                    ) : (
                      <div className="w-full h-full flex items-center justify-center text-slate-400 text-xs">No img</div>
                    )}
                    {sampleResults[c.name] && (
                      <span className="absolute top-1 right-1 bg-green-600 text-white text-[9px] font-bold px-1.5 py-0.5 rounded-full leading-none">
                        DEMO
                      </span>
                    )}
                  </div>
                  <span className="text-[11px] font-medium text-slate-700 text-center line-clamp-1 leading-tight">
                    {c.name}
                  </span>
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Right Column: Prediction Results & Grad-CAM Visualizer (7 cols) */}
        <div className="lg:col-span-7 space-y-4">
          {!result && !loading && (
            <div className="bg-white border border-dashed border-slate-300 rounded-2xl p-12 text-center flex flex-col items-center justify-center gap-4">
              <div className="w-16 h-16 rounded-2xl bg-green-50 text-green-700 flex items-center justify-center">
                <Layers className="w-8 h-8" />
              </div>
              <div className="max-w-md">
                <h4 className="font-bold text-slate-800 text-base mb-1">Awaiting Leaf Image</h4>
                <p className="text-xs text-slate-500 leading-relaxed">
                  Upload a photo of a rice leaf lesion or select a sample image from the left. Click
                  <strong> Run Comparison</strong> to view live classification probabilities and
                  Explainable AI (Grad-CAM) visual overlays.
                </p>
              </div>
            </div>
          )}

          {loading && (
            <div className="bg-white border border-slate-200 rounded-2xl p-12 text-center flex flex-col items-center justify-center gap-4">
              <div className="relative">
                <div className="w-16 h-16 rounded-full border-4 border-green-200 border-t-green-700 animate-spin" />
                <Sparkles className="w-6 h-6 text-green-700 absolute inset-0 m-auto" />
              </div>
              <div>
                <p className="font-bold text-slate-800 text-base">Running Deep Learning Inference…</p>
                <p className="text-xs text-slate-500 mt-1">
                  Computing activations and calculating backpropagated Grad-CAM gradients for target disease lesions.
                </p>
              </div>
            </div>
          )}

          {result && (
            <div className="space-y-4">
              {/* Dual Mode View */}
              {result.mode === 'both' && result.efficientnetv2s && result.convnextbase && (
                <DualModeDisplay
                  eff={result.efficientnetv2s}
                  conv={result.convnextbase}
                  preview={preview}
                  viewMode={viewMode}
                  setViewMode={setViewMode}
                  opacity={opacity}
                  setOpacity={setOpacity}
                  chartDataEff={getChartData(result.efficientnetv2s.scores)}
                  chartDataConv={getChartData(result.convnextbase.scores)}
                  detailsEff={getClassDetails(result.efficientnetv2s.predicted_class)}
                  detailsConv={getClassDetails(result.convnextbase.predicted_class)}
                />
              )}

              {/* Single Model View */}
              {result.mode === 'single' && (
                <SingleModeDisplay
                  result={result}
                  preview={preview}
                  viewMode={viewMode}
                  setViewMode={setViewMode}
                  opacity={opacity}
                  setOpacity={setOpacity}
                  chartData={getChartData(result.scores)}
                  details={getClassDetails(result.predicted_class)}
                />
              )}

              {/* Disclaimer */}
              <div className="flex gap-2.5 rounded-xl bg-amber-50/80 border border-amber-200 px-4 py-3 text-xs text-amber-900 leading-relaxed">
                <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5 text-amber-700" />
                <p>{result.disclaimer}</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Subcomponents for Dual Mode & Single Mode
// ─────────────────────────────────────────────────────────────────────────────

interface DualModeProps {
  eff: SinglePredictionResult;
  conv: SinglePredictionResult;
  preview: string | null;
  viewMode: 'overlay' | 'raw' | 'original';
  setViewMode: (v: 'overlay' | 'raw' | 'original') => void;
  opacity: number;
  setOpacity: (v: number) => void;
  chartDataEff: Array<{ class: string; score: number }>;
  chartDataConv: Array<{ class: string; score: number }>;
  detailsEff: ClassInfo | null | undefined;
  detailsConv: ClassInfo | null | undefined;
}

function DualModeDisplay({
  eff,
  conv,
  preview,
  viewMode,
  setViewMode,
  opacity,
  setOpacity,
  chartDataEff,
  chartDataConv,
  detailsEff,
  detailsConv,
}: DualModeProps) {
  const isAgreed = eff.predicted_class === conv.predicted_class;

  return (
    <div className="space-y-5">
      {/* Agreement Banner */}
      <div className={`p-4 rounded-2xl border flex items-center justify-between gap-4 ${
        isAgreed
          ? 'bg-green-50 border-green-200 text-green-900'
          : 'bg-amber-50 border-amber-200 text-amber-900'
      }`}>
        <div className="flex items-center gap-3">
          {isAgreed ? (
            <CheckCircle className="w-6 h-6 text-green-600 shrink-0" />
          ) : (
            <ShieldAlert className="w-6 h-6 text-amber-600 shrink-0" />
          )}
          <div>
            <h4 className="font-bold text-sm">
              {isAgreed ? 'Consensus: Both Models Agree' : 'Divergent Classifications'}
            </h4>
            <p className="text-xs opacity-90 mt-0.5">
              {isAgreed
                ? `Both networks classified this specimen as ${eff.predicted_class}`
                : `EfficientNetV2-S predicts "${eff.predicted_class}", whereas ConvNeXtBase predicts "${conv.predicted_class}"`}
            </p>
          </div>
        </div>
        <div className="text-right shrink-0">
          <span className="text-xs font-semibold px-2.5 py-1 rounded-full bg-white shadow-sm border border-slate-200">
            {isAgreed ? 'High Confidence' : 'Review Needed'}
          </span>
        </div>
      </div>

      {/* Grad-CAM Viewer Toolbar */}
      <div className="bg-white border border-slate-200 rounded-2xl p-4 shadow-sm space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <Eye className="w-4 h-4 text-slate-700" />
            <span className="text-xs font-bold text-slate-800 uppercase tracking-wide">
              Grad-CAM Heatmap Visualizer
            </span>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            {viewMode === 'overlay' && (
              <div className="flex items-center gap-2 text-xs text-slate-600 bg-slate-50 px-2.5 py-1 rounded-lg border border-slate-200">
                <Sliders className="w-3.5 h-3.5 text-slate-400" />
                <span>Opacity: {Math.round(opacity * 100)}%</span>
                <input
                  type="range"
                  min="0.2"
                  max="1.0"
                  step="0.05"
                  value={opacity}
                  onChange={(e) => setOpacity(parseFloat(e.target.value))}
                  className="w-20 accent-green-700 h-1 cursor-pointer"
                />
              </div>
            )}

            {/* View Mode Toggle */}
            <div className="flex items-center rounded-xl bg-slate-100 p-1 text-xs font-medium text-slate-600">
              <button
                onClick={() => setViewMode('overlay')}
                className={`px-3 py-1 rounded-lg transition-all ${
                  viewMode === 'overlay' ? 'bg-white text-slate-900 font-semibold shadow-sm' : 'hover:text-slate-900'
                }`}
              >
                Overlay Heatmap
              </button>
              <button
                onClick={() => setViewMode('raw')}
                className={`px-3 py-1 rounded-lg transition-all ${
                  viewMode === 'raw' ? 'bg-white text-slate-900 font-semibold shadow-sm' : 'hover:text-slate-900'
                }`}
              >
                Raw Heatmap
              </button>
              <button
                onClick={() => setViewMode('original')}
                className={`px-3 py-1 rounded-lg transition-all ${
                  viewMode === 'original' ? 'bg-white text-slate-900 font-semibold shadow-sm' : 'hover:text-slate-900'
                }`}
              >
                Original Leaf
              </button>
            </div>
          </div>
        </div>

        {/* Side-by-Side Grad-CAM Images */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 pt-1">
          {/* EfficientNetV2-S Heatmap */}
          <div className="space-y-1.5">
            <div className="flex items-center justify-between text-xs font-semibold">
              <span className="text-green-800 flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-full bg-green-600 inline-block" />
                EfficientNetV2-S
              </span>
              <span className="text-slate-500 font-normal">{eff.inference_ms} ms</span>
            </div>
            <div className="relative aspect-square rounded-xl overflow-hidden border border-slate-200 bg-black">
              <img
                src={viewMode === 'raw' ? eff.gradcam_raw : viewMode === 'original' ? (preview || eff.gradcam_overlay) : eff.gradcam_overlay}
                alt="EfficientNetV2-S Grad-CAM"
                className="w-full h-full object-contain"
                style={{ opacity: viewMode === 'overlay' ? opacity : 1 }}
              />
              <div className="absolute top-2 left-2 bg-black/60 backdrop-blur-sm text-white text-[10px] px-2 py-0.5 rounded">
                Grad-CAM Focus
              </div>
            </div>
            <p className="text-[11px] text-slate-500 leading-tight">
              Sharp attention focused directly on specific necrotic lesion boundaries.
            </p>
          </div>

          {/* ConvNeXtBase Heatmap */}
          <div className="space-y-1.5">
            <div className="flex items-center justify-between text-xs font-semibold">
              <span className="text-blue-800 flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-full bg-blue-600 inline-block" />
                ConvNeXtBase
              </span>
              <span className="text-slate-500 font-normal">{conv.inference_ms} ms</span>
            </div>
            <div className="relative aspect-square rounded-xl overflow-hidden border border-slate-200 bg-black">
              <img
                src={viewMode === 'raw' ? conv.gradcam_raw : viewMode === 'original' ? (preview || conv.gradcam_overlay) : conv.gradcam_overlay}
                alt="ConvNeXtBase Grad-CAM"
                className="w-full h-full object-contain"
                style={{ opacity: viewMode === 'overlay' ? opacity : 1 }}
              />
              <div className="absolute top-2 left-2 bg-black/60 backdrop-blur-sm text-white text-[10px] px-2 py-0.5 rounded">
                Grad-CAM Focus
              </div>
            </div>
            <p className="text-[11px] text-slate-500 leading-tight">
              Large $7\times 7$ depthwise kernels pooling broader contextual leaf chlorosis.
            </p>
          </div>
        </div>
      </div>

      {/* Side-by-Side Classification Comparison */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {/* EfficientNet Card */}
        <div className="bg-white border border-slate-200 rounded-2xl p-4 shadow-sm space-y-3">
          <div className="border-b border-slate-100 pb-3">
            <span className="text-xs font-bold text-green-700 uppercase tracking-wide">EfficientNetV2-S</span>
            <h4 className="text-lg font-bold text-slate-900 mt-0.5">{eff.predicted_class}</h4>
            <div className="flex items-center gap-2 mt-1">
              <div className="flex-1 h-2 bg-slate-100 rounded-full overflow-hidden">
                <div
                  className="h-full bg-green-600 rounded-full"
                  style={{ width: `${eff.confidence * 100}%` }}
                />
              </div>
              <span className="text-xs font-bold text-green-700">{(eff.confidence * 100).toFixed(1)}%</span>
            </div>
          </div>

          {/* Probabilities chart */}
          <div className="pt-1">
            <span className="text-xs font-semibold text-slate-600 mb-2 block">Probability Breakdown</span>
            <div className="space-y-1.5">
              {chartDataEff.slice(0, 4).map((entry) => (
                <div key={entry.class} className="flex items-center gap-2 text-xs">
                  <span className="w-28 truncate text-slate-600 font-medium">{entry.class}</span>
                  <div className="flex-1 h-1.5 bg-slate-100 rounded-full overflow-hidden">
                    <div
                      className="h-full rounded-full"
                      style={{
                        width: `${entry.score}%`,
                        backgroundColor: entry.class === eff.predicted_class ? CLASS_COLORS_MAP[entry.class] || '#16a34a' : '#cbd5e1',
                      }}
                    />
                  </div>
                  <span className="w-10 text-right font-mono text-[11px] text-slate-500">{entry.score}%</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* ConvNeXt Card */}
        <div className="bg-white border border-slate-200 rounded-2xl p-4 shadow-sm space-y-3">
          <div className="border-b border-slate-100 pb-3">
            <span className="text-xs font-bold text-blue-700 uppercase tracking-wide">ConvNeXtBase</span>
            <h4 className="text-lg font-bold text-slate-900 mt-0.5">{conv.predicted_class}</h4>
            <div className="flex items-center gap-2 mt-1">
              <div className="flex-1 h-2 bg-slate-100 rounded-full overflow-hidden">
                <div
                  className="h-full bg-blue-600 rounded-full"
                  style={{ width: `${conv.confidence * 100}%` }}
                />
              </div>
              <span className="text-xs font-bold text-blue-700">{(conv.confidence * 100).toFixed(1)}%</span>
            </div>
          </div>

          {/* Probabilities chart */}
          <div className="pt-1">
            <span className="text-xs font-semibold text-slate-600 mb-2 block">Probability Breakdown</span>
            <div className="space-y-1.5">
              {chartDataConv.slice(0, 4).map((entry) => (
                <div key={entry.class} className="flex items-center gap-2 text-xs">
                  <span className="w-28 truncate text-slate-600 font-medium">{entry.class}</span>
                  <div className="flex-1 h-1.5 bg-slate-100 rounded-full overflow-hidden">
                    <div
                      className="h-full rounded-full"
                      style={{
                        width: `${entry.score}%`,
                        backgroundColor: entry.class === conv.predicted_class ? CLASS_COLORS_MAP[entry.class] || '#3b82f6' : '#cbd5e1',
                      }}
                    />
                  </div>
                  <span className="w-10 text-right font-mono text-[11px] text-slate-500">{entry.score}%</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Disease Management Guide */}
      {detailsEff && (
        <div className="bg-white border border-slate-200 rounded-2xl p-5 shadow-sm space-y-3">
          <div className="flex items-center gap-2 text-sm font-bold text-slate-800">
            <Info className="w-4 h-4 text-green-700" />
            Agronomic Field Guide: {detailsEff.name}
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs text-slate-600 leading-relaxed pt-1">
            <div className="bg-slate-50 p-3 rounded-xl border border-slate-100">
              <strong className="text-slate-800 block mb-1">Key Diagnostic Symptoms:</strong>
              {detailsEff.symptoms}
            </div>
            <div className="bg-green-50/50 p-3 rounded-xl border border-green-100">
              <strong className="text-green-900 block mb-1">Management Recommendations:</strong>
              {detailsEff.management}
            </div>
          </div>
        </div>
      )}

      {/* ConvNeXt Disease Guide (if predictions differ) */}
      {!isAgreed && detailsConv && (
        <div className="bg-white border border-blue-200 rounded-2xl p-5 shadow-sm space-y-3">
          <div className="flex items-center gap-2 text-sm font-bold text-blue-900">
            <Info className="w-4 h-4 text-blue-700" />
            ConvNeXtBase Differential Diagnosis: {detailsConv.name}
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs text-slate-600 leading-relaxed pt-1">
            <div className="bg-slate-50 p-3 rounded-xl border border-slate-100">
              <strong className="text-slate-800 block mb-1">Diagnostic Symptoms:</strong>
              {detailsConv.symptoms}
            </div>
            <div className="bg-blue-50/50 p-3 rounded-xl border border-blue-100">
              <strong className="text-blue-900 block mb-1">Management Strategy:</strong>
              {detailsConv.management}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Single Mode Subcomponent
// ─────────────────────────────────────────────────────────────────────────────

interface SingleModeProps {
  result: PredictResponse;
  preview: string | null;
  viewMode: 'overlay' | 'raw' | 'original';
  setViewMode: (v: 'overlay' | 'raw' | 'original') => void;
  opacity: number;
  setOpacity: (v: number) => void;
  chartData: Array<{ class: string; score: number }>;
  details: ClassInfo | null | undefined;
}

function SingleModeDisplay({
  result,
  preview,
  viewMode,
  setViewMode,
  opacity,
  setOpacity,
  chartData,
  details,
}: SingleModeProps) {
  const modelLabel = result.model === 'convnextbase' ? 'ConvNeXtBase' : 'EfficientNetV2-S';

  return (
    <div className="space-y-4">
      {/* Top Prediction Card */}
      <div className="bg-white border border-green-200 rounded-2xl p-5 shadow-sm">
        <div className="flex items-start justify-between gap-4">
          <div>
            <span className="text-xs font-semibold text-slate-500 uppercase tracking-wide">
              {modelLabel} Output
            </span>
            <h3 className="text-2xl font-extrabold text-green-900 mt-0.5 leading-tight">
              {result.predicted_class}
            </h3>
            <div className="flex items-center gap-3 mt-2 text-xs text-slate-600">
              <span>Confidence: <strong className="text-green-700">{((result.confidence || 0) * 100).toFixed(1)}%</strong></span>
              <span>·</span>
              <span>Latency: <strong>{result.inference_ms} ms</strong></span>
            </div>
          </div>
          <span className="px-3 py-1 bg-green-100 text-green-800 font-bold text-xs rounded-full">
            Top Result
          </span>
        </div>

        <div className="mt-4">
          <div className="h-2.5 bg-slate-100 rounded-full overflow-hidden">
            <div
              className="h-full bg-green-600 rounded-full transition-all duration-500"
              style={{ width: `${(result.confidence || 0) * 100}%` }}
            />
          </div>
        </div>
      </div>

      {/* Grad-CAM Viewer */}
      {result.gradcam_overlay && (
        <div className="bg-white border border-slate-200 rounded-2xl p-4 shadow-sm space-y-3">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <span className="text-xs font-bold text-slate-800 uppercase tracking-wide flex items-center gap-2">
              <Eye className="w-4 h-4 text-slate-700" />
              Grad-CAM Visual Attention Map
            </span>
            <div className="flex flex-wrap items-center gap-3">
              {viewMode === 'overlay' && (
                <div className="flex items-center gap-2 text-xs text-slate-600 bg-slate-50 px-2.5 py-1 rounded-lg border border-slate-200">
                  <Sliders className="w-3.5 h-3.5 text-slate-400" />
                  <span>Opacity: {Math.round(opacity * 100)}%</span>
                  <input
                    type="range"
                    min="0.2"
                    max="1.0"
                    step="0.05"
                    value={opacity}
                    onChange={(e) => setOpacity(parseFloat(e.target.value))}
                    className="w-20 accent-green-700 h-1 cursor-pointer"
                  />
                </div>
              )}
              <div className="flex items-center rounded-xl bg-slate-100 p-1 text-xs">
                <button
                  onClick={() => setViewMode('overlay')}
                  className={`px-3 py-1 rounded-lg transition-all ${
                    viewMode === 'overlay' ? 'bg-white text-slate-900 font-semibold shadow-sm' : 'text-slate-600'
                  }`}
                >
                  Overlay
                </button>
                <button
                  onClick={() => setViewMode('raw')}
                  className={`px-3 py-1 rounded-lg transition-all ${
                    viewMode === 'raw' ? 'bg-white text-slate-900 font-semibold shadow-sm' : 'text-slate-600'
                  }`}
                >
                  Raw Heatmap
                </button>
                <button
                  onClick={() => setViewMode('original')}
                  className={`px-3 py-1 rounded-lg transition-all ${
                    viewMode === 'original' ? 'bg-white text-slate-900 font-semibold shadow-sm' : 'text-slate-600'
                  }`}
                >
                  Original
                </button>
              </div>
            </div>
          </div>

          <div className="aspect-video max-h-80 rounded-xl overflow-hidden border border-slate-200 bg-black flex items-center justify-center">
            <img
              src={viewMode === 'raw' ? result.gradcam_raw : viewMode === 'original' ? (preview || result.gradcam_overlay) : result.gradcam_overlay}
              alt="Grad-CAM"
              className="max-h-full max-w-full object-contain"
              style={{ opacity: viewMode === 'overlay' ? opacity : 1 }}
            />
          </div>
        </div>
      )}

      {/* Probabilities Distribution */}
      <div className="bg-white border border-slate-200 rounded-2xl p-4 shadow-sm">
        <h4 className="font-semibold text-slate-800 text-xs uppercase tracking-wide mb-3">
          Class Probability Distribution
        </h4>
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={chartData} layout="vertical" margin={{ left: 8, right: 16, top: 4, bottom: 4 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" horizontal={false} />
            <XAxis type="number" domain={[0, 100]} tick={{ fontSize: 11, fill: '#94a3b8' }} tickFormatter={(v) => `${v}%`} />
            <YAxis type="category" dataKey="class" tick={{ fontSize: 11, fill: '#475569' }} width={130} />
            <Tooltip formatter={(v) => `${Number(v).toFixed(1)}%`} />
            <Bar dataKey="score" radius={[0, 4, 4, 0]}>
              {chartData.map((entry) => (
                <Cell
                  key={entry.class}
                  fill={entry.class === result.predicted_class ? CLASS_COLORS_MAP[entry.class] || '#16a34a' : '#cbd5e1'}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Disease Information Card */}
      {details && (
        <div className="bg-white border border-slate-200 rounded-2xl p-5 shadow-sm space-y-2 text-xs text-slate-600">
          <h4 className="font-bold text-slate-800 text-sm">{details.name}</h4>
          <p>{details.description}</p>
          <div className="pt-2 border-t border-slate-100">
            <strong className="text-slate-700 block mb-0.5">Management:</strong>
            {details.management}
          </div>
        </div>
      )}
    </div>
  );
}
