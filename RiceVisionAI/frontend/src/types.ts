// API types shared across the app

export interface HealthResponse {
  status: string;
  efficientnetv2s_available: boolean;
  convnextbase_available: boolean;
  efficientnetv2s_loaded: boolean;
  convnextbase_loaded: boolean;
  active_models: string[];
}

export interface ClassInfo {
  name: string;
  index: number;
  description: string;
  symptoms: string;
  management: string;
  image: string | null;
}

export interface ModelMetrics {
  model: string;
  test_sample_count: number;
  accuracy: number;
  precision_macro: number;
  precision_weighted: number;
  recall_macro: number;
  recall_weighted: number;
  f1_score_macro: number;
  f1_score_weighted: number;
  roc_auc_macro: number;
  roc_auc_weighted: number;
}

export interface InferenceLatency {
  profiled_samples: number;
  average_inference_time_ms: number;
  median_inference_time_ms: number;
  std_inference_time_ms: number;
  throughput_fps: number;
}

export interface ModelEfficiency {
  model_architecture: string;
  total_parameters: number;
  trainable_parameters: number;
  model_file_path: string;
  model_file_size_bytes: number;
  model_file_size_mb: number;
  training_time_seconds: number;
  training_time_minutes: number;
  inference_latency: InferenceLatency;
}

export interface ModelData {
  available: boolean;
  metrics: ModelMetrics;
  efficiency: ModelEfficiency;
}

export interface ModelsResponse {
  efficientnetv2s: ModelData;
  convnextbase: ModelData;
}

export interface PerClassMetric {
  class: string;
  precision: number;
  recall: number;
  f1_score: number;
  support: number;
}

export interface MetricsResponse {
  efficientnetv2s: {
    per_class: PerClassMetric[];
    confusion_matrix: Record<string, Record<string, number>>;
  };
  convnextbase: {
    per_class: PerClassMetric[];
    confusion_matrix: Record<string, Record<string, number>>;
  };
  class_names: string[];
}

export interface SinglePredictionResult {
  model: string;
  model_label: string;
  predicted_class: string;
  predicted_index: number;
  confidence: number;
  scores: Record<string, number>;
  inference_ms: number;
  gradcam_overlay: string;
  gradcam_raw: string;
}

export interface PredictResponse {
  mode: 'single' | 'both';
  model?: string;
  // Fields present when mode === 'single'
  predicted_class?: string;
  predicted_index?: number;
  confidence?: number;
  scores?: Record<string, number>;
  inference_ms?: number;
  gradcam_overlay?: string;
  gradcam_raw?: string;
  // Fields present when mode === 'both'
  efficientnetv2s?: SinglePredictionResult;
  convnextbase?: SinglePredictionResult;
  disclaimer: string;
}
