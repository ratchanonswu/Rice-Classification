import { type ElementType, type ReactNode } from 'react';
import { SectionHeader } from '../components/UI';
import { Database, Filter, Layers, FlaskConical, Sparkles } from 'lucide-react';

interface Step {
  icon: ElementType;
  title: string;
  badge?: string;
  badgeColor?: 'green' | 'blue' | 'orange' | 'slate';
  content: ReactNode;
}

const steps: Step[] = [
  {
    icon: Database,
    title: 'Dataset Collection & Standardization',
    badge: '3,829 images',
    badgeColor: 'slate',
    content: (
      <div className="space-y-2 text-sm text-slate-600 leading-relaxed">
        <p>
          The dataset comprises 3,829 high-resolution images of rice leaves across six pathological categories:
          Bacterial Leaf Blight, Brown Spot, Healthy Rice Leaf, Leaf Blast, Leaf Scald, and Sheath Blight.
        </p>
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 mt-3">
          {[
            ['Bacterial Leaf Blight', '630 images'],
            ['Brown Spot', '570 images'],
            ['Healthy Rice Leaf', '700 images'],
            ['Leaf Blast', '630 images'],
            ['Leaf Scald', '640 images'],
            ['Sheath Blight', '659 images'],
          ].map(([cls, count]) => (
            <div key={cls} className="bg-slate-50 border border-slate-200 rounded-xl px-3 py-2">
              <p className="text-xs font-semibold text-slate-700">{cls}</p>
              <p className="text-[11px] text-slate-400">{count}</p>
            </div>
          ))}
        </div>
      </div>
    ),
  },
  {
    icon: Filter,
    title: 'Audit & Leakage-Free Splitting',
    badge: 'Zero Overlap',
    badgeColor: 'orange',
    content: (
      <div className="space-y-3 text-sm text-slate-600 leading-relaxed">
        <p>
          An automated two-stage deduplication pipeline used SHA-256 cryptographic hashing and perceptual hashing (pHash).
          Exact duplicates were removed and near-duplicate clusters were grouped into the same split so no identical
          or visually similar leaf specimen could leak across train, validation, and test sets.
        </p>
        <ul className="space-y-1.5 list-disc list-inside text-xs text-slate-500">
          <li>31 exact duplicate copies excluded via SHA-256; 3,798 unique canonical images preserved</li>
          <li>67 near-duplicate clusters grouped via pHash and locked to the same split</li>
          <li>Leakage-free stratified splitting: 75% train (2,847), 10% validation (381), 15% test (570)</li>
        </ul>
      </div>
    ),
  },
  {
    icon: Layers,
    title: 'Model Architectures: CNN vs. Vision Transformer',
    badge: 'Comparative Study',
    badgeColor: 'blue',
    content: (
      <div className="space-y-4 text-sm text-slate-600 leading-relaxed">
        <p>
          We benchmarked two complementary modern deep learning paradigms: a lightweight convolutional mobile backbone
          (<strong>EfficientNetV2-S</strong>) versus a full-capacity pretrained self-attention transformer
          (<strong>ViT-B/16</strong>).
        </p>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div className="bg-green-50/50 border border-green-200 rounded-2xl p-4 space-y-2">
            <h4 className="font-bold text-green-900 text-sm">EfficientNetV2-S (Lightweight CNN)</h4>
            <ul className="space-y-1 text-xs text-slate-600">
              <li>• <strong>20.66M</strong> total parameters</li>
              <li>• <strong>83.04 MB</strong> model checkpoint</li>
              <li>• Fused-MBConv layers with progressive learning</li>
              <li>• Native Keras internal normalization layers</li>
              <li>• <strong>GlobalAveragePooling2D</strong> (7×7×1280 → 1280-d) → Dense(256, ReLU) → Dropout(0.3) → Dense(6, Softmax)</li>
            </ul>
          </div>
          <div className="bg-purple-50/50 border border-purple-200 rounded-2xl p-4 space-y-2">
            <h4 className="font-bold text-purple-900 text-sm">ViT-B/16 (Vision Transformer)</h4>
            <ul className="space-y-1 text-xs text-slate-600">
              <li>• <strong>85.80M</strong> total parameters</li>
              <li>• <strong>982.5 MB</strong> model checkpoint</li>
              <li>• 12 Transformer layers, 12 attention heads, 768-dim</li>
              <li>• 16×16 non-overlapping patch embeddings (196 tokens)</li>
              <li>• <strong>CLS Token Pooling</strong> (197×768 → 768-d) → Dropout(0.1) → Dense(6, Softmax)</li>
            </ul>
          </div>
        </div>
        <div className="bg-slate-50 border border-slate-200 rounded-2xl p-4 text-xs space-y-2 text-slate-600">
          <p className="font-semibold text-slate-800">Pooling Layer Flow Across Architectures &amp; XAI</p>
          <p>
            <strong>1. CNN Spatial Pooling (GlobalAveragePooling2D):</strong> Collapses the 7×7×1280 backbone feature maps
            into a 1280-d vector by averaging each channel across space instead of <code>Flatten()</code>, reducing
            overfitting and enabling direct channel-weight attribution for Grad-CAM++.
          </p>
          <p>
            <strong>2. Transformer Token Pooling (CLS Token):</strong> Summarizes global self-attention across all 196 leaf
            patches into the learned <code>[CLS]</code> token (768-d) prior to classification.
          </p>
          <p>
            <strong>3. XAI Patch Pooling (avg_pool2d 16×16):</strong> Aggregates 224×224 pixel input gradients via{' '}
            <code>tf.nn.avg_pool2d(ksize=16, strides=16)</code> into a 14×14 patch saliency grid.
          </p>
        </div>
      </div>
    ),
  },
  {
    icon: Sparkles,
    title: 'Explainable AI: Grad-CAM++ & ViT Attention Saliency',
    badge: 'Visual Attribution',
    badgeColor: 'green',
    content: (
      <div className="space-y-3 text-sm text-slate-600 leading-relaxed">
        <p>
          Dual Explainable AI (XAI) pipelines visually verify the internal reasoning and decisive leaf regions
          each network utilized to predict a given rice disease:
        </p>
        <div className="bg-slate-50 border border-slate-200 rounded-2xl p-4 text-xs space-y-2 text-slate-600">
          <p>
            <strong>EfficientNet Grad-CAM++:</strong> Computes positive gradient activations weighted by higher-order
            derivatives from the final convolutional stage, isolating sharp, pinpoint lesion borders.
          </p>
          <p>
            <strong>ViT Attention Saliency:</strong> Computes backpropagated input-gradient saliency grouped into
            14×14 patch tokens, mapping the transformer's global self-attention across the whole blade of the leaf.
          </p>
          <p>
            <strong>Colormap Superimposition:</strong> Heatmaps are min-max normalized, resized to leaf resolution,
            and blended via OpenCV <code>COLORMAP_JET</code> at $\alpha=0.45$.
          </p>
        </div>
      </div>
    ),
  },
  {
    icon: FlaskConical,
    title: 'Empirical Results & Test Evaluation',
    badge: '570 Test Images',
    badgeColor: 'green',
    content: (
      <div className="space-y-3 text-sm text-slate-600 leading-relaxed">
        <p>
          Final model checkpoints were evaluated on the 570 unseen test samples. ViT-B/16 achieved state-of-the-art
          accuracy (97.72%) with robust lesion localization, while EfficientNetV2-S provided 11.8× lighter storage footprint.
        </p>
        <div className="overflow-x-auto">
          <table className="w-full text-xs border border-slate-200 rounded-xl overflow-hidden">
            <thead className="bg-slate-50">
              <tr>
                <th className="text-left px-3 py-2 text-slate-600 font-semibold">Metric</th>
                <th className="text-right px-3 py-2 text-green-700 font-bold">EfficientNetV2-S</th>
                <th className="text-right px-3 py-2 text-purple-700 font-bold">ViT-B/16 (Transformer)</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {[
                ['Test Accuracy', '95.26%', '97.72%'],
                ['Macro Precision', '95.23%', '97.80%'],
                ['Macro Recall', '95.22%', '97.70%'],
                ['Macro F1 Score', '95.21%', '97.71%'],
                ['ROC-AUC (OvR)', '0.9961', '0.9974'],
                ['Total Parameters', '20.66 M', '85.80 M'],
                ['Model File Size', '83.04 MB', '982.5 MB'],
                ['Inference Latency', '14.88 ms', '7.64 ms'],
                ['Throughput', '67.2 FPS', '130.8 FPS'],
              ].map(([k, eff, vit], i) => (
                <tr key={k} className={i % 2 === 0 ? 'bg-white' : 'bg-slate-50/50'}>
                  <td className="px-3 py-2 text-slate-700 font-medium">{k}</td>
                  <td className="px-3 py-2 text-right text-green-700 font-bold">{eff}</td>
                  <td className="px-3 py-2 text-right text-purple-700 font-bold">{vit}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    ),
  },
];

export default function Methodology() {
  return (
    <div className="space-y-8 max-w-7xl mx-auto pb-12">
      <SectionHeader
        title="Research Methodology & Explainable AI Pipeline"
        subtitle="End-to-end experimental protocol from data audit to dual-model CNN vs Vision Transformer XAI feature visualization"
      />

      <div className="space-y-6">
        {steps.map((step, idx) => {
          const Icon = step.icon;
          return (
            <div key={step.title} className="bg-white border border-slate-200 rounded-3xl p-6 shadow-sm space-y-4">
              <div className="flex items-center justify-between gap-4">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-2xl bg-green-50 text-green-800 flex items-center justify-center font-bold">
                    <Icon className="w-5 h-5" />
                  </div>
                  <div>
                    <h3 className="font-extrabold text-slate-800 text-base">
                      {idx + 1}. {step.title}
                    </h3>
                  </div>
                </div>
                {step.badge && (
                  <span className="text-xs font-semibold px-2.5 py-1 rounded-full bg-slate-100 text-slate-700">
                    {step.badge}
                  </span>
                )}
              </div>
              {step.content}
            </div>
          );
        })}
      </div>
    </div>
  );
}
