import { useEffect, useState } from 'react';
import { fetchClasses } from '../api';
import type { ClassInfo } from '../types';
import { LoadingSpinner, ErrorBanner, SectionHeader } from '../components/UI';
import { Leaf, ImageOff, ShieldAlert, Activity } from 'lucide-react';

const CLASS_COLORS: Record<string, string> = {
  'Bacterial Leaf Blight': 'border-yellow-300 bg-yellow-50',
  'Brown Spot': 'border-amber-300 bg-amber-50',
  'Healthy Rice Leaf': 'border-green-300 bg-green-50',
  'Leaf Blast': 'border-red-300 bg-red-50',
  'Leaf Scald': 'border-orange-300 bg-orange-50',
  'Sheath Blight': 'border-purple-300 bg-purple-50',
};

const CLASS_BADGE: Record<string, string> = {
  'Bacterial Leaf Blight': 'bg-yellow-100 text-yellow-800',
  'Brown Spot': 'bg-amber-100 text-amber-800',
  'Healthy Rice Leaf': 'bg-green-100 text-green-800',
  'Leaf Blast': 'bg-red-100 text-red-800',
  'Leaf Scald': 'bg-orange-100 text-orange-800',
  'Sheath Blight': 'bg-purple-100 text-purple-800',
};

export default function Classes() {
  const [classes, setClasses] = useState<ClassInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<ClassInfo | null>(null);

  useEffect(() => {
    fetchClasses()
      .then((data) => { setClasses(data); setSelected(data[0] ?? null); })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <LoadingSpinner label="Loading disease classes…" />;
  if (error) return <ErrorBanner message={error} />;

  return (
    <div className="space-y-8 max-w-7xl mx-auto pb-12">
      <SectionHeader
        title="Rice Disease Class Reference"
        subtitle="Six authoritative pathology classes classified by EfficientNetV2-S and ConvNeXtBase"
      />

      <div className="flex flex-col lg:flex-row gap-6">
        {/* Sidebar list */}
        <div className="lg:w-72 shrink-0 space-y-2">
          {classes.map((cls) => (
            <button
              key={cls.name}
              onClick={() => setSelected(cls)}
              className={`w-full text-left px-4 py-3.5 rounded-2xl border transition-all flex items-center gap-3 ${
                selected?.name === cls.name
                  ? `${CLASS_COLORS[cls.name] || 'border-green-300 bg-green-50'} shadow-sm font-semibold`
                  : 'bg-white border-slate-200 hover:border-slate-300'
              }`}
            >
              <span className={`w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold shrink-0 ${CLASS_BADGE[cls.name] || 'bg-slate-100 text-slate-600'}`}>
                {cls.index}
              </span>
              <span className="text-sm text-slate-800 leading-tight">{cls.name}</span>
            </button>
          ))}
        </div>

        {/* Detail panel */}
        {selected && (
          <div className="flex-1 bg-white border border-slate-200 rounded-3xl overflow-hidden shadow-sm">
            {/* Image banner */}
            <div className="aspect-video max-h-80 bg-slate-100 relative flex items-center justify-center overflow-hidden">
              {selected.image ? (
                <img
                  src={selected.image}
                  alt={selected.name}
                  className="w-full h-full object-cover"
                />
              ) : (
                <div className="flex flex-col items-center gap-2 text-slate-400">
                  <ImageOff className="w-10 h-10" />
                  <span className="text-sm">No representative image available</span>
                </div>
              )}
              <div className="absolute bottom-4 left-4">
                <span className={`px-3 py-1.5 rounded-full text-xs font-bold shadow-md ${CLASS_BADGE[selected.name] || 'bg-slate-100 text-slate-700'}`}>
                  Class {selected.index} · {selected.name}
                </span>
              </div>
            </div>

            {/* Content */}
            <div className="p-6 sm:p-8 space-y-6">
              <div className="flex items-center gap-3">
                <div className={`w-12 h-12 rounded-2xl flex items-center justify-center ${CLASS_COLORS[selected.name] || 'bg-green-50'}`}>
                  <Leaf className="w-6 h-6 text-green-700" />
                </div>
                <div>
                  <h2 className="font-extrabold text-slate-900 text-2xl">{selected.name}</h2>
                  <p className="text-xs text-slate-400">Diagnostic Category Index: {selected.index}</p>
                </div>
              </div>

              <p className="text-slate-600 text-sm leading-relaxed">{selected.description}</p>

              {/* Clinical symptoms and management */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-2">
                <div className="bg-amber-50/60 border border-amber-200/80 rounded-2xl p-4 space-y-1.5">
                  <h4 className="text-xs font-bold uppercase tracking-wider text-amber-900 flex items-center gap-1.5">
                    <Activity className="w-4 h-4 text-amber-700" />
                    Key Foliar Symptoms
                  </h4>
                  <p className="text-xs text-slate-700 leading-relaxed">
                    {selected.symptoms || 'Water-soaked lesion margins, chlorotic halos, or necrotic patches depending on fungal or bacterial sporulation stage.'}
                  </p>
                </div>

                <div className="bg-green-50/60 border border-green-200/80 rounded-2xl p-4 space-y-1.5">
                  <h4 className="text-xs font-bold uppercase tracking-wider text-green-900 flex items-center gap-1.5">
                    <ShieldAlert className="w-4 h-4 text-green-700" />
                    Field Management & Intervention
                  </h4>
                  <p className="text-xs text-slate-700 leading-relaxed">
                    {selected.management || 'Adopt balanced nutrient applications, control canopy moisture through proper spacing, and apply preventive biological or chemical fungicide regimens.'}
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* All classes grid view */}
      <div>
        <h3 className="font-bold text-slate-800 text-base mb-4">All Classes at a Glance</h3>
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
          {classes.map((cls) => (
            <button
              key={cls.name}
              onClick={() => setSelected(cls)}
              className="bg-white border border-slate-200 rounded-2xl overflow-hidden hover:border-green-400 hover:shadow-md transition-all text-left group"
            >
              <div className="aspect-video bg-slate-100 overflow-hidden">
                {cls.image ? (
                  <img src={cls.image} alt={cls.name} className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300" />
                ) : (
                  <div className="w-full h-full flex items-center justify-center text-slate-300">
                    <ImageOff className="w-8 h-8" />
                  </div>
                )}
              </div>
              <div className="p-3">
                <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full ${CLASS_BADGE[cls.name] || 'bg-slate-100 text-slate-600'}`}>
                  Class {cls.index}
                </span>
                <p className="mt-1 text-xs font-bold text-slate-800 line-clamp-1">{cls.name}</p>
              </div>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
