import { useEffect, useState } from 'react';
import {
  RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, Legend,
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
} from 'recharts';
import { fetchModels, fetchMetrics } from '../api';
import type { ModelsResponse, MetricsResponse } from '../types';
import { StatCard, LoadingSpinner, ErrorBanner, SectionHeader } from '../components/UI';
import { BarChart2, Cpu } from 'lucide-react';

const CLASS_ABBR: Record<string, string> = {
  'Bacterial Leaf Blight': 'BLB',
  'Brown Spot': 'BS',
  'Healthy Rice Leaf': 'HRL',
  'Leaf Blast': 'LB',
  'Leaf Scald': 'LS',
  'Sheath Blight': 'SB',
};

const PCT = (v: number) => `${(v * 100).toFixed(2)}%`;
const PCT1 = (v: number) => `${(v * 100).toFixed(1)}%`;

export default function Comparison() {
  const [models, setModels] = useState<ModelsResponse | null>(null);
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'overview' | 'per-class' | 'confusion'>('overview');

  useEffect(() => {
    Promise.all([fetchModels(), fetchMetrics()])
      .then(([m, me]) => { setModels(m); setMetrics(me); })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <LoadingSpinner label="Loading model comparison data…" />;
  if (error) return <ErrorBanner message={error} />;

  const eff = models!.efficientnetv2s;
  const vit = models!.vit_base;

  // Summary comparison rows
  const compRows = [
    { label: 'Test Accuracy', eff: eff.metrics.accuracy, vit: vit.metrics.accuracy },
    { label: 'Macro Precision', eff: eff.metrics.precision_macro, vit: vit.metrics.precision_macro },
    { label: 'Macro Recall', eff: eff.metrics.recall_macro, vit: vit.metrics.recall_macro },
    { label: 'Macro F1', eff: eff.metrics.f1_score_macro, vit: vit.metrics.f1_score_macro },
    { label: 'ROC-AUC (macro)', eff: eff.metrics.roc_auc_macro, vit: vit.metrics.roc_auc_macro },
  ];

  const effRows = [
    { label: 'Model Size (MB)', eff: eff.efficiency.model_file_size_mb, vit: vit.efficiency.model_file_size_mb, fmt: (v: number) => `${v} MB`, lowerBetter: true },
    { label: 'Total Parameters', eff: eff.efficiency.total_parameters, vit: vit.efficiency.total_parameters, fmt: (v: number) => v.toLocaleString(), lowerBetter: true },
    { label: 'Training Time (min)', eff: eff.efficiency.training_time_minutes, vit: vit.efficiency.training_time_minutes, fmt: (v: number) => `${v.toFixed(2)} min`, lowerBetter: true },
    { label: 'Avg Inference (ms)', eff: eff.efficiency.inference_latency.average_inference_time_ms, vit: vit.efficiency.inference_latency.average_inference_time_ms, fmt: (v: number) => `${v.toFixed(2)} ms`, lowerBetter: true },
    { label: 'Throughput (FPS)', eff: eff.efficiency.inference_latency.throughput_fps, vit: vit.efficiency.inference_latency.throughput_fps, fmt: (v: number) => `${v.toFixed(1)} fps`, lowerBetter: false },
  ];

  // Per-class bar chart data
  const perClassData = metrics!.class_names.map((cls) => {
    const effCls = metrics!.efficientnetv2s.per_class.find((r) => r.class === cls);
    const vitCls = metrics!.vit_base.per_class.find((r) => r.class === cls);
    return {
      class: CLASS_ABBR[cls] || cls,
      fullName: cls,
      'EfficientNetV2-S': effCls ? +(effCls.f1_score * 100).toFixed(1) : 0,
      'ViT-B/16': vitCls ? +(vitCls.f1_score * 100).toFixed(1) : 0,
    };
  });

  // Radar chart data
  const radarData = compRows.map(({ label, eff: e, vit: v }) => ({
    metric: label.replace('Macro ', '').replace(' (macro)', ''),
    'EfficientNetV2-S': +(e * 100).toFixed(2),
    'ViT-B/16': +(v * 100).toFixed(2),
  }));

  // Confusion matrix renderer
  const ConfusionMatrix = ({ modelKey, label }: { modelKey: 'efficientnetv2s' | 'vit_base'; label: string }) => {
    const cm = metrics![modelKey].confusion_matrix;
    const classNames = metrics!.class_names;
    if (!cm || Object.keys(cm).length === 0) return <p className="text-slate-400 text-sm">No confusion matrix data.</p>;

    const maxVal = Math.max(...classNames.flatMap((r) => classNames.map((c) => cm[r]?.[c] ?? 0)));

    return (
      <div className="bg-white border border-slate-200 rounded-2xl p-5 shadow-sm space-y-3">
        <div className="flex items-center justify-between">
          <h4 className="font-bold text-slate-800 text-sm">{label} Confusion Matrix</h4>
          <span className="text-xs text-slate-400">570 Test Images</span>
        </div>
        <div className="overflow-x-auto">
          <table className="text-xs border-collapse w-full">
            <thead>
              <tr>
                <th className="p-1 text-slate-400 font-normal text-right pr-2">Actual ↓ / Pred →</th>
                {classNames.map((c) => (
                  <th key={c} className="p-1 text-slate-600 font-semibold text-center" style={{ minWidth: 50 }}>
                    {CLASS_ABBR[c]}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {classNames.map((actual) => (
                <tr key={actual}>
                  <td className="pr-2 py-1 text-slate-700 font-medium text-right whitespace-nowrap">{CLASS_ABBR[actual]}</td>
                  {classNames.map((predicted) => {
                    const val = cm[actual]?.[predicted] ?? 0;
                    const isDiag = actual === predicted;
                    const opacity = maxVal > 0 ? val / maxVal : 0;
                    const bg = isDiag
                      ? `rgba(22, 163, 74, ${0.15 + opacity * 0.7})`
                      : val > 0 ? `rgba(239, 68, 68, ${0.05 + opacity * 0.5})` : 'transparent';
                    return (
                      <td
                        key={predicted}
                        className="p-1.5 text-center font-mono border border-slate-100 rounded"
                        style={{ backgroundColor: bg, minWidth: 50 }}
                        title={`${actual} → ${predicted}: ${val}`}
                      >
                        <span className={isDiag ? 'font-bold text-green-800' : val > 0 ? 'text-red-700 font-medium' : 'text-slate-300'}>
                          {val}
                        </span>
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="flex items-center justify-between text-[11px] text-slate-400 pt-1">
          <span>Diagonal = Correct predictions</span>
          <span>Off-diagonal = Misclassifications</span>
        </div>
      </div>
    );
  };

  return (
    <div className="space-y-8 max-w-7xl mx-auto pb-12">
      <SectionHeader
        title="Head-to-Head Model Benchmark"
        subtitle="Empirical comparison between EfficientNetV2-S (Lightweight CNN) and ViT-B/16 (Vision Transformer)"
      />

      {/* Tabs */}
      <div className="flex border-b border-slate-200 gap-2">
        {(['overview', 'per-class', 'confusion'] as const).map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`pb-3 px-4 text-sm font-semibold capitalize border-b-2 transition-all ${
              activeTab === tab
                ? 'border-green-700 text-green-900'
                : 'border-transparent text-slate-500 hover:text-slate-800'
            }`}
          >
            {tab === 'overview' && 'Benchmark Overview'}
            {tab === 'per-class' && 'Per-Class Metrics'}
            {tab === 'confusion' && 'Confusion Matrices'}
          </button>
        ))}
      </div>

      {/* Tab 1: Overview */}
      {activeTab === 'overview' && (
        <div className="space-y-6">
          {/* Key Metric Highlights */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <StatCard
              label="ViT-B/16 Test Acc"
              value={PCT(vit.metrics.accuracy)}
              sub="Transformer Leader (+2.46%)"
              accent
            />
            <StatCard
              label="EfficientNetV2-S Acc"
              value={PCT(eff.metrics.accuracy)}
              sub="Lightweight CNN Baseline"
            />
            <StatCard
              label="Model Size Gap"
              value="11.8x Lighter"
              sub="83 MB vs 982 MB"
              accent
            />
            <StatCard
              label="ViT Throughput"
              value="130.8 FPS"
              sub="Sub-8ms Latency on GPU"
            />
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Classification Performance Table */}
            <div className="bg-white border border-slate-200 rounded-2xl p-5 shadow-sm space-y-4">
              <h3 className="font-bold text-slate-800 text-sm flex items-center gap-2">
                <BarChart2 className="w-4 h-4 text-green-700" />
                Classification Metrics (570 Test Samples)
              </h3>
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="border-b border-slate-200 text-slate-500 text-left">
                      <th className="py-2.5 font-semibold">Metric</th>
                      <th className="py-2.5 font-bold text-green-800 text-right">EfficientNetV2-S</th>
                      <th className="py-2.5 font-bold text-purple-800 text-right">ViT-B/16</th>
                      <th className="py-2.5 font-semibold text-slate-500 text-right">Difference</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    {compRows.map((r) => {
                      const diff = r.vit - r.eff;
                      return (
                        <tr key={r.label} className="hover:bg-slate-50/60">
                          <td className="py-2.5 font-medium text-slate-700">{r.label}</td>
                          <td className="py-2.5 font-bold text-green-700 text-right">{PCT(r.eff)}</td>
                          <td className="py-2.5 font-bold text-purple-700 text-right">{PCT(r.vit)}</td>
                          <td className="py-2.5 font-mono text-right font-semibold text-purple-600">
                            +{PCT(diff)} (ViT)
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Radar Comparison Chart */}
            <div className="bg-white border border-slate-200 rounded-2xl p-5 shadow-sm space-y-2">
              <h3 className="font-bold text-slate-800 text-sm">Performance Profile Radar</h3>
              <ResponsiveContainer width="100%" height={260}>
                <RadarChart data={radarData}>
                  <PolarGrid stroke="#e2e8f0" />
                  <PolarAngleAxis dataKey="metric" tick={{ fontSize: 11, fill: '#64748b' }} />
                  <PolarRadiusAxis domain={[90, 100]} tick={{ fontSize: 9 }} />
                  <Radar name="EfficientNetV2-S" dataKey="EfficientNetV2-S" stroke="#16a34a" fill="#16a34a" fillOpacity={0.3} />
                  <Radar name="ViT-B/16" dataKey="ViT-B/16" stroke="#9333ea" fill="#9333ea" fillOpacity={0.25} />
                  <Legend />
                  <Tooltip />
                </RadarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Efficiency Table */}
          <div className="bg-white border border-slate-200 rounded-2xl p-5 shadow-sm space-y-4">
            <h3 className="font-bold text-slate-800 text-sm flex items-center gap-2">
              <Cpu className="w-4 h-4 text-green-700" />
              Computational & Deployment Efficiency
            </h3>
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-slate-200 text-slate-500 text-left">
                    <th className="py-2.5 font-semibold">Parameter</th>
                    <th className="py-2.5 font-bold text-green-800 text-right">EfficientNetV2-S</th>
                    <th className="py-2.5 font-bold text-purple-800 text-right">ViT-B/16</th>
                    <th className="py-2.5 font-semibold text-slate-500 text-right">Efficiency Winner</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {effRows.map((r) => {
                    const effWins = r.lowerBetter ? r.eff < r.vit : r.eff > r.vit;
                    return (
                      <tr key={r.label} className="hover:bg-slate-50/60">
                        <td className="py-2.5 font-medium text-slate-700">{r.label}</td>
                        <td className="py-2.5 font-semibold text-slate-900 text-right">{r.fmt(r.eff)}</td>
                        <td className="py-2.5 font-medium text-slate-600 text-right">{r.fmt(r.vit)}</td>
                        <td className="py-2.5 text-right">
                          <span className={`px-2 py-0.5 rounded text-[11px] font-bold ${
                            effWins ? 'bg-green-100 text-green-800' : 'bg-purple-100 text-purple-800'
                          }`}>
                            {effWins ? 'EfficientNetV2-S' : 'ViT-B/16'}
                          </span>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* Tab 2: Per-Class */}
      {activeTab === 'per-class' && (
        <div className="space-y-6">
          <div className="bg-white border border-slate-200 rounded-2xl p-5 shadow-sm space-y-4">
            <h3 className="font-bold text-slate-800 text-sm">F1-Score Comparison Across All 6 Disease Classes</h3>
            <ResponsiveContainer width="100%" height={320}>
              <BarChart data={perClassData} margin={{ top: 10, right: 20, left: 0, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                <XAxis dataKey="class" tick={{ fontSize: 11 }} />
                <YAxis domain={[85, 100]} tick={{ fontSize: 11 }} unit="%" />
                <Tooltip formatter={(v) => `${Number(v).toFixed(1)}%`} />
                <Legend />
                <Bar dataKey="EfficientNetV2-S" fill="#16a34a" radius={[4, 4, 0, 0]} />
                <Bar dataKey="ViT-B/16" fill="#9333ea" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {metrics!.class_names.map((cls) => {
              const effCls = metrics!.efficientnetv2s.per_class.find((r) => r.class === cls);
              const vitCls = metrics!.vit_base.per_class.find((r) => r.class === cls);
              return (
                <div key={cls} className="bg-white border border-slate-200 rounded-2xl p-4 shadow-sm space-y-2">
                  <div className="flex items-center justify-between">
                    <h4 className="font-bold text-slate-800 text-sm">{cls}</h4>
                    <span className="text-xs text-slate-400 font-mono">Support: {effCls?.support}</span>
                  </div>
                  <div className="grid grid-cols-3 gap-2 text-center text-xs pt-1 border-t border-slate-100">
                    <div>
                      <span className="text-slate-400 block text-[10px]">Precision</span>
                      <strong className="text-green-700">{PCT1(effCls?.precision ?? 0)}</strong>
                      <span className="text-slate-400 text-[10px] block">vs <strong className="text-purple-700">{PCT1(vitCls?.precision ?? 0)}</strong></span>
                    </div>
                    <div>
                      <span className="text-slate-400 block text-[10px]">Recall</span>
                      <strong className="text-green-700">{PCT1(effCls?.recall ?? 0)}</strong>
                      <span className="text-slate-400 text-[10px] block">vs <strong className="text-purple-700">{PCT1(vitCls?.recall ?? 0)}</strong></span>
                    </div>
                    <div>
                      <span className="text-slate-400 block text-[10px]">F1 Score</span>
                      <strong className="text-green-700">{PCT1(effCls?.f1_score ?? 0)}</strong>
                      <span className="text-slate-400 text-[10px] block">vs <strong className="text-purple-700">{PCT1(vitCls?.f1_score ?? 0)}</strong></span>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Tab 3: Confusion Matrices */}
      {activeTab === 'confusion' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <ConfusionMatrix modelKey="efficientnetv2s" label="EfficientNetV2-S" />
          <ConfusionMatrix modelKey="vit_base" label="ViT-B/16 (Vision Transformer)" />
        </div>
      )}
    </div>
  );
}
