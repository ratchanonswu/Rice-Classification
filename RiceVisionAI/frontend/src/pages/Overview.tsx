import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import {
  BarChart2, Upload, ArrowRight,
  CheckCircle, AlertCircle, Sparkles, Layers, Compass, Zap, Shield
} from 'lucide-react';
import { fetchHealth, fetchModels } from '../api';
import type { HealthResponse, ModelsResponse } from '../types';
import { LoadingSpinner, ErrorBanner } from '../components/UI';

export default function Overview() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [models, setModels] = useState<ModelsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([fetchHealth(), fetchModels()])
      .then(([h, m]) => {
        setHealth(h);
        setModels(m);
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <LoadingSpinner label="Loading research dashboard…" />;
  if (error) return <ErrorBanner message={`Could not reach backend: ${error}`} />;

  const eff = models?.efficientnetv2s;
  const conv = models?.convnextbase;

  return (
    <div className="space-y-10 max-w-7xl mx-auto pb-12">
      {/* Hero Banner */}
      <div className="rounded-3xl bg-gradient-to-br from-green-950 via-green-900 to-emerald-900 p-8 sm:p-10 text-white shadow-xl relative overflow-hidden">
        <div className="absolute right-0 top-0 bottom-0 w-1/3 opacity-10 pointer-events-none flex items-center justify-center">
          <Layers className="w-96 h-96 text-white" />
        </div>
        <div className="max-w-3xl relative z-10">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-white/10 backdrop-blur-md text-green-300 text-xs font-semibold uppercase tracking-wider mb-4 border border-white/10">
            <Sparkles className="w-3.5 h-3.5" />
            Dual-Model Deep Learning Benchmark
          </div>
          <h1 className="text-3xl sm:text-4xl font-extrabold mb-4 tracking-tight leading-tight">
            Rice Leaf Disease Classification & Explainable AI
          </h1>
          <p className="text-green-100 text-base leading-relaxed mb-8 max-w-2xl">
            A controlled, data-leakage-free study comparing <strong>EfficientNetV2-S</strong> (Lightweight Champion)
            and <strong>ConvNeXtBase</strong> (Modern ConvNet) on 3,798 images across 6 rice diseases, featuring
            on-the-fly <strong>Grad-CAM visual attribution maps</strong>.
          </p>

          <div className="flex flex-wrap gap-4">
            <Link
              to="/predict"
              className="flex items-center gap-2 px-6 py-3.5 bg-white text-green-900 rounded-xl text-sm font-bold hover:bg-green-50 shadow-md hover:shadow-lg transition-all"
            >
              <Upload className="w-4 h-4 text-green-700" />
              Try Dual Predictor & Grad-CAM
            </Link>
            <Link
              to="/comparison"
              className="flex items-center gap-2 px-6 py-3.5 bg-green-800/80 hover:bg-green-800 border border-white/20 text-white rounded-xl text-sm font-semibold transition-all"
            >
              <BarChart2 className="w-4 h-4" />
              Head-to-Head Comparison
            </Link>
          </div>
        </div>
      </div>

      {/* Backend Status Bar */}
      <div className="flex items-center justify-between gap-4 bg-white border border-slate-200 rounded-2xl px-5 py-4 shadow-sm">
        <div className="flex items-center gap-3">
          {health?.status === 'ok' ? (
            <CheckCircle className="w-5 h-5 text-green-600 shrink-0" />
          ) : (
            <AlertCircle className="w-5 h-5 text-amber-500 shrink-0" />
          )}
          <div className="text-xs sm:text-sm text-slate-600 flex flex-wrap gap-x-4 gap-y-1">
            <span>
              Backend API: <strong className={health?.status === 'ok' ? 'text-green-700' : 'text-amber-600'}>
                {health?.status === 'ok' ? 'Online & Ready' : 'Unavailable'}
              </strong>
            </span>
            <span>·</span>
            <span>
              EfficientNetV2-S:{' '}
              <strong className={health?.efficientnetv2s_available ? 'text-green-700' : 'text-red-600'}>
                {health?.efficientnetv2s_available ? 'Loaded (83 MB)' : 'Missing'}
              </strong>
            </span>
            <span>·</span>
            <span>
              ConvNeXtBase:{' '}
              <strong className={health?.convnextbase_available ? 'text-green-700' : 'text-red-600'}>
                {health?.convnextbase_available ? 'Loaded (338 MB)' : 'Missing'}
              </strong>
            </span>
          </div>
        </div>
        <span className="hidden md:inline-flex text-xs px-2.5 py-1 rounded-full bg-slate-100 text-slate-600 font-medium">
          Unified Server :8000
        </span>
      </div>

      {/* Head-to-Head Model Showdown Cards */}
      <div>
        <h2 className="text-lg font-bold text-slate-800 mb-4 flex items-center gap-2">
          <Compass className="w-5 h-5 text-green-700" />
          Final Models Benchmark Overview
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* EfficientNetV2-S Card */}
          <div className="bg-white border-2 border-green-500/30 rounded-3xl p-6 shadow-sm hover:shadow-md transition-shadow relative overflow-hidden">
            <div className="flex items-start justify-between gap-3 mb-4">
              <div>
                <span className="text-xs font-bold text-green-700 uppercase tracking-wider bg-green-50 px-2.5 py-1 rounded-full border border-green-200">
                  Top Performance
                </span>
                <h3 className="text-2xl font-black text-slate-900 mt-2">EfficientNetV2-S</h3>
                <p className="text-xs text-slate-500 mt-1">Lightweight Fused-MBConv CNN Architecture</p>
              </div>
              <div className="text-right">
                <span className="text-3xl font-extrabold text-green-700">
                  {eff ? `${(eff.metrics.accuracy * 100).toFixed(1)}%` : '92.8%'}
                </span>
                <p className="text-[11px] font-semibold text-slate-400 uppercase">Test Accuracy</p>
              </div>
            </div>

            <div className="grid grid-cols-3 gap-3 my-5 py-4 border-y border-slate-100 text-center">
              <div>
                <p className="text-xs text-slate-400 font-medium">Parameters</p>
                <p className="text-base font-bold text-slate-800 mt-0.5">20.7 M</p>
              </div>
              <div>
                <p className="text-xs text-slate-400 font-medium">Macro F1</p>
                <p className="text-base font-bold text-green-700 mt-0.5">
                  {eff ? `${(eff.metrics.f1_score_macro * 100).toFixed(1)}%` : '92.7%'}
                </p>
              </div>
              <div>
                <p className="text-xs text-slate-400 font-medium">Model Size</p>
                <p className="text-base font-bold text-slate-800 mt-0.5">83 MB</p>
              </div>
            </div>

            <p className="text-xs text-slate-600 leading-relaxed mb-4">
              EfficientNetV2-S achieved the highest overall classification accuracy and macro F1 score.
              Its Grad-CAM heatmaps demonstrate tight, pinpoint localization around distinct disease lesion boundaries.
            </p>

            <Link
              to="/predict"
              className="inline-flex items-center gap-1.5 text-xs font-bold text-green-700 hover:text-green-800"
            >
              Test with EfficientNetV2-S <ArrowRight className="w-3.5 h-3.5" />
            </Link>
          </div>

          {/* ConvNeXtBase Card */}
          <div className="bg-white border border-slate-200 rounded-3xl p-6 shadow-sm hover:shadow-md transition-shadow relative overflow-hidden">
            <div className="flex items-start justify-between gap-3 mb-4">
              <div>
                <span className="text-xs font-bold text-blue-700 uppercase tracking-wider bg-blue-50 px-2.5 py-1 rounded-full border border-blue-200">
                  Modern ConvNet
                </span>
                <h3 className="text-2xl font-black text-slate-900 mt-2">ConvNeXtBase</h3>
                <p className="text-xs text-slate-500 mt-1">7x7 Depthwise Large-Kernel Architecture</p>
              </div>
              <div className="text-right">
                <span className="text-3xl font-extrabold text-blue-700">
                  {conv ? `${(conv.metrics.accuracy * 100).toFixed(1)}%` : '90.2%'}
                </span>
                <p className="text-[11px] font-semibold text-slate-400 uppercase">Test Accuracy</p>
              </div>
            </div>

            <div className="grid grid-cols-3 gap-3 my-5 py-4 border-y border-slate-100 text-center">
              <div>
                <p className="text-xs text-slate-400 font-medium">Parameters</p>
                <p className="text-base font-bold text-slate-800 mt-0.5">87.8 M</p>
              </div>
              <div>
                <p className="text-xs text-slate-400 font-medium">Macro F1</p>
                <p className="text-base font-bold text-blue-700 mt-0.5">
                  {conv ? `${(conv.metrics.f1_score_macro * 100).toFixed(1)}%` : '90.2%'}
                </p>
              </div>
              <div>
                <p className="text-xs text-slate-400 font-medium">Model Size</p>
                <p className="text-base font-bold text-slate-800 mt-0.5">338 MB</p>
              </div>
            </div>

            <p className="text-xs text-slate-600 leading-relaxed mb-4">
              ConvNeXt modernizes traditional ConvNets inspired by Vision Transformers. Its large receptive field
              pools context across broader sections of leaf tissue, providing complementary diagnostic signals.
            </p>

            <Link
              to="/predict"
              className="inline-flex items-center gap-1.5 text-xs font-bold text-blue-700 hover:text-blue-800"
            >
              Test with ConvNeXtBase <ArrowRight className="w-3.5 h-3.5" />
            </Link>
          </div>
        </div>
      </div>

      {/* Highlights / Features Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-white border border-slate-200 rounded-2xl p-5 space-y-2">
          <div className="w-10 h-10 rounded-xl bg-green-100 text-green-800 flex items-center justify-center font-bold">
            <Zap className="w-5 h-5" />
          </div>
          <h4 className="font-bold text-slate-800 text-sm">Dual Mode via 1-Click</h4>
          <p className="text-xs text-slate-500 leading-relaxed">
            Run both architectures on a single leaf image concurrently to cross-validate diagnostic hypotheses
            and highlight agreement or divergence.
          </p>
        </div>

        <div className="bg-white border border-slate-200 rounded-2xl p-5 space-y-2">
          <div className="w-10 h-10 rounded-xl bg-emerald-100 text-emerald-800 flex items-center justify-center font-bold">
            <Sparkles className="w-5 h-5" />
          </div>
          <h4 className="font-bold text-slate-800 text-sm">Explainable Grad-CAM</h4>
          <p className="text-xs text-slate-500 leading-relaxed">
            Head-propagation Gradient-weighted Class Activation Mapping displays exact activation heatmaps
            directly superimposed onto the specimen leaves.
          </p>
        </div>

        <div className="bg-white border border-slate-200 rounded-2xl p-5 space-y-2">
          <div className="w-10 h-10 rounded-xl bg-blue-100 text-blue-800 flex items-center justify-center font-bold">
            <Shield className="w-5 h-5" />
          </div>
          <h4 className="font-bold text-slate-800 text-sm">Leakage-Free Validation</h4>
          <p className="text-xs text-slate-500 leading-relaxed">
            Evaluated on a rigorously audited, deduplicated test split of 570 unseen field images across
            all 6 canonical rice disease classes.
          </p>
        </div>
      </div>
    </div>
  );
}
