import React, { useState, useCallback } from 'react';
import { Cpu, HardDrive, Server, Zap, AlertCircle, CheckCircle2, Loader2, ExternalLink, ChevronDown, ChevronUp } from 'lucide-react';
import acceleratorData from './gpu-specs.json';

// Accelerator specifications loaded from config file
const ACCELERATORS = acceleratorData.accelerators.map(acc => ({
  name: acc.name,
  memory: acc.memory_gb,
  interconnect: acc.interconnect,
  tier: acc.tier,
  type: acc.type || 'gpu',
  perNode: acc.per_node,
  maxUnits: acc.max_units,
  notes: acc.notes,
}));

// Precision configurations
const PRECISIONS = [
  { name: 'FP32', bytesPerParam: 4, description: 'Full precision', color: '#ef4444' },
  { name: 'BF16/FP16', bytesPerParam: 2, description: 'Half precision', color: '#f59e0b' },
  { name: 'INT8', bytesPerParam: 1, description: '8-bit quantized', color: '#22c55e' },
  { name: 'INT4/NF4', bytesPerParam: 0.5, description: '4-bit quantized', color: '#06b6d4' },
  { name: 'INT2', bytesPerParam: 0.25, description: '2-bit quantized', color: '#8b5cf6' },
];

// Helper to format bytes
const formatBytes = (bytes) => {
  if (bytes >= 1e12) return `${(bytes / 1e12).toFixed(2)} TB`;
  if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(2)} GB`;
  if (bytes >= 1e6) return `${(bytes / 1e6).toFixed(2)} MB`;
  return `${(bytes / 1e3).toFixed(2)} KB`;
};

// Helper to format large numbers
const formatNumber = (num) => {
  if (num >= 1e12) return `${(num / 1e12).toFixed(1)}T`;
  if (num >= 1e9) return `${(num / 1e9).toFixed(1)}B`;
  if (num >= 1e6) return `${(num / 1e6).toFixed(1)}M`;
  if (num >= 1e3) return `${(num / 1e3).toFixed(1)}K`;
  return num.toString();
};

// Parse model config to extract parameters
const parseModelConfig = (config, repoId) => {
  // Try to get parameter count from various sources
  let numParams = null;
  let numLayers = null;
  let hiddenSize = null;
  let numHeads = null;
  let numKvHeads = null;
  let vocabSize = null;
  let intermediateSize = null;
  let contextLength = null;
  let architecture = null;
  let modelType = config.model_type || 'unknown';

  // Handle nested text_config for multimodal models (Mistral3, Pixtral, etc.)
  const textConfig = config.text_config || config.language_config || config.llm_config || config;
  const visionConfig = config.vision_config || null;

  // Detect if this is Mistral params.json format
  const isMistralFormat = config.dim !== undefined && config.n_layers !== undefined;
  
  if (isMistralFormat) {
    // Mistral params.json format
    hiddenSize = config.dim;
    numLayers = config.n_layers;
    numHeads = config.n_heads;
    numKvHeads = config.n_kv_heads || numHeads;
    vocabSize = config.vocab_size;
    intermediateSize = config.hidden_dim;
    contextLength = config.max_seq_len || config.sliding_window || 32768;
    architecture = 'Mistral';
    modelType = 'mistral';
    
    // Check for MoE in Mistral format
    if (config.moe) {
      const numExperts = config.moe.num_experts;
      const numActiveExperts = config.moe.num_experts_per_tok || 2;
      
      // Calculate MoE parameters
      if (numLayers && hiddenSize && vocabSize && intermediateSize) {
        const embeddingParams = vocabSize * hiddenSize;
        const headDim = hiddenSize / numHeads;
        
        // Attention per layer
        const attnParams = (hiddenSize * hiddenSize) + // Q
                          (hiddenSize * numKvHeads * headDim) + // K
                          (hiddenSize * numKvHeads * headDim) + // V
                          (hiddenSize * hiddenSize); // O
        
        // MoE FFN per layer (each expert has gate, up, down projections)
        const expertFFNParams = 3 * hiddenSize * intermediateSize;
        const moeFFNParams = numExperts * expertFFNParams;
        const routerParams = numExperts * hiddenSize;
        
        // Layer norms
        const lnParams = 4 * hiddenSize;
        
        const paramsPerLayer = attnParams + moeFFNParams + routerParams + lnParams;
        numParams = embeddingParams + (paramsPerLayer * numLayers) + hiddenSize + embeddingParams;
      }
      
      return {
        numParams,
        numLayers,
        hiddenSize,
        numHeads,
        numKvHeads,
        vocabSize,
        intermediateSize,
        contextLength,
        architecture,
        modelType,
        isMoE: true,
        numExperts: config.moe.num_experts,
        numActiveExperts: config.moe.num_experts_per_tok || 2,
        isMultimodal: false,
        rawConfig: config,
      };
    }
  }

  // Architecture detection (standard HF format)
  architecture = config.architectures?.[0] || config.model_type || architecture || 'Unknown';

  // Common parameter extraction - prefer textConfig for multimodal, fallback to root config
  numLayers = textConfig.num_hidden_layers || textConfig.n_layer || textConfig.num_layers || textConfig.n_layers ||
              config.num_hidden_layers || config.n_layer || config.num_layers || config.n_layers;
  hiddenSize = textConfig.hidden_size || textConfig.d_model || textConfig.n_embd || textConfig.dim ||
               config.hidden_size || config.d_model || config.n_embd || config.dim;
  numHeads = textConfig.num_attention_heads || textConfig.n_head || textConfig.num_heads ||
             config.num_attention_heads || config.n_head || config.num_heads;
  numKvHeads = textConfig.num_key_value_heads || textConfig.n_head_kv ||
               config.num_key_value_heads || config.n_head_kv || numHeads;
  vocabSize = textConfig.vocab_size || config.vocab_size;
  intermediateSize = textConfig.intermediate_size || textConfig.n_inner || textConfig.ffn_dim ||
                     config.intermediate_size || config.n_inner || config.ffn_dim;
  contextLength = textConfig.max_position_embeddings || textConfig.max_seq_len || textConfig.n_positions || textConfig.seq_length ||
                  config.max_position_embeddings || config.max_seq_len || config.n_positions || config.seq_length || 4096;

  // Detect multimodal
  const isMultimodal = !!(visionConfig || config.vision_tower || config.image_token_index !== undefined);

  // Calculate parameters if not directly available
  let textParams = null;
  let visionParams = null;
  
  if (numLayers && hiddenSize && vocabSize) {
    // Embedding parameters
    const embeddingParams = vocabSize * hiddenSize;
    
    // Attention parameters per layer (Q, K, V, O projections)
    const headDim = textConfig.head_dim || config.head_dim || (hiddenSize / numHeads);
    const qParams = hiddenSize * numHeads * headDim;
    const kParams = hiddenSize * (numKvHeads || numHeads) * headDim;
    const vParams = hiddenSize * (numKvHeads || numHeads) * headDim;
    const oParams = numHeads * headDim * hiddenSize;
    const attnParams = qParams + kParams + vParams + oParams;
    
    // FFN parameters per layer
    const ffnHidden = intermediateSize || hiddenSize * 4;
    // Check for gated FFN (like in Llama, Gemma)
    const hiddenAct = textConfig.hidden_act || config.hidden_act || '';
    const isGatedFFN = hiddenAct === 'silu' || modelType.includes('llama') || modelType.includes('gemma') || modelType.includes('mistral');
    const ffnParams = isGatedFFN ? 3 * hiddenSize * ffnHidden : 2 * hiddenSize * ffnHidden;
    
    // Layer norm parameters per layer (2 layer norms)
    const lnParams = 4 * hiddenSize;
    
    // Total per layer
    const paramsPerLayer = attnParams + ffnParams + lnParams;
    
    // Final layer norm and output projection
    const tiedEmbeddings = textConfig.tie_word_embeddings || config.tie_word_embeddings;
    const finalParams = hiddenSize + (tiedEmbeddings ? 0 : embeddingParams);
    
    textParams = embeddingParams + (paramsPerLayer * numLayers) + finalParams;
  }

  // Calculate vision parameters if multimodal
  if (visionConfig && visionConfig.hidden_size && visionConfig.num_hidden_layers) {
    const vHidden = visionConfig.hidden_size;
    const vLayers = visionConfig.num_hidden_layers;
    const vHeads = visionConfig.num_attention_heads;
    const vIntermediate = visionConfig.intermediate_size || vHidden * 4;
    const vHeadDim = visionConfig.head_dim || (vHidden / vHeads);
    
    // Vision transformer params
    const patchSize = visionConfig.patch_size || 14;
    const channels = visionConfig.num_channels || 3;
    const patchEmbed = channels * patchSize * patchSize * vHidden;
    
    const vAttnParams = 4 * vHidden * vHidden; // Q, K, V, O
    const vFFNParams = 2 * vHidden * vIntermediate;
    const vLNParams = 4 * vHidden;
    
    visionParams = patchEmbed + (vAttnParams + vFFNParams + vLNParams) * vLayers;
  }

  // Add multimodal projector params if present
  let projectorParams = 0;
  if (isMultimodal && hiddenSize && visionConfig?.hidden_size) {
    // Simple linear projector estimate
    projectorParams = visionConfig.hidden_size * hiddenSize * 2;
  }

  numParams = (textParams || 0) + (visionParams || 0) + projectorParams;
  if (numParams === 0) numParams = null;

  // MoE handling for models like DeepSeek, Mixtral (standard HF format)
  const moeExperts = textConfig.num_experts || textConfig.num_local_experts || config.num_experts || config.num_local_experts;
  if (moeExperts) {
    const numExperts = moeExperts;
    const numActiveExperts = textConfig.num_experts_per_tok || textConfig.num_selected_experts || 
                             config.num_experts_per_tok || config.num_selected_experts || 2;
    const expertSize = intermediateSize || hiddenSize * 4;
    
    // MoE FFN calculation
    if (numLayers && hiddenSize) {
      const moeFFNParams = numExperts * 3 * hiddenSize * expertSize; // per layer
      const routerParams = numExperts * hiddenSize; // per layer
      
      // Recalculate with MoE
      const embeddingParams = vocabSize * hiddenSize;
      const headDim = textConfig.head_dim || config.head_dim || (hiddenSize / numHeads);
      const attnParams = hiddenSize * numHeads * headDim + 
                        hiddenSize * numKvHeads * headDim * 2 +
                        numHeads * headDim * hiddenSize;
      const lnParams = 4 * hiddenSize;
      
      const paramsPerMoeLayer = attnParams + moeFFNParams + routerParams + lnParams;
      
      numParams = embeddingParams + (paramsPerMoeLayer * numLayers) + hiddenSize + embeddingParams + (visionParams || 0) + projectorParams;
    }
    
    return {
      numParams,
      numLayers,
      hiddenSize,
      numHeads,
      numKvHeads,
      vocabSize,
      intermediateSize,
      contextLength,
      architecture,
      modelType,
      isMoE: true,
      numExperts,
      numActiveExperts,
      isMultimodal,
      visionParams,
      rawConfig: config,
    };
  }

  // Override with explicit parameter counts if available
  if (config.num_parameters) {
    numParams = config.num_parameters;
  }

  // Try to extract from repo name (e.g., "gemma-2b" -> 2B)
  if (!numParams && repoId) {
    const match = repoId.match(/(\d+\.?\d*)\s*[bB]/);
    if (match) {
      numParams = parseFloat(match[1]) * 1e9;
    }
    const matchT = repoId.match(/(\d+\.?\d*)\s*[tT]/);
    if (matchT) {
      numParams = parseFloat(matchT[1]) * 1e12;
    }
  }

  return {
    numParams,
    numLayers,
    hiddenSize,
    numHeads,
    numKvHeads,
    vocabSize,
    intermediateSize,
    contextLength,
    architecture,
    modelType,
    isMoE: false,
    numExperts: null,
    numActiveExperts: null,
    isMultimodal,
    visionParams,
    rawConfig: config,
  };
};

// Calculate memory requirements
const calculateMemory = (modelInfo, precision, contextLength = 4096, batchSize = 1) => {
  const { numParams, numLayers, hiddenSize, numHeads, numKvHeads } = modelInfo;
  
  if (!numParams) return null;

  // Model weights
  const weightsMemory = numParams * precision.bytesPerParam;
  
  // KV cache calculation
  let kvCacheMemory = 0;
  if (numLayers && hiddenSize && numHeads) {
    const headDim = hiddenSize / numHeads;
    const kvHeads = numKvHeads || numHeads;
    // KV cache per token: 2 (K+V) * layers * kv_heads * head_dim * bytes
    const kvPerToken = 2 * numLayers * kvHeads * headDim * precision.bytesPerParam;
    kvCacheMemory = kvPerToken * contextLength * batchSize;
  }
  
  // Activation memory (rough estimate: ~10-20% of model size for inference)
  const activationMemory = weightsMemory * 0.1;
  
  // Total with overhead
  const totalMemory = weightsMemory + kvCacheMemory + activationMemory;
  const withOverhead = totalMemory * 1.1; // 10% overhead for framework

  return {
    weights: weightsMemory,
    kvCache: kvCacheMemory,
    activations: activationMemory,
    total: totalMemory,
    withOverhead,
  };
};

// Calculate accelerator requirements
const calculateAcceleratorRequirements = (memoryRequired, acc) => {
  const totalMemory = acc.memory * 1e9; // Convert GB to bytes
  const usableMemory = totalMemory * 0.9; // 90% usable
  
  const unitsNeeded = Math.ceil(memoryRequired / usableMemory);
  const nodesNeeded = Math.ceil(unitsNeeded / acc.perNode);
  const efficiency = (memoryRequired / (unitsNeeded * usableMemory)) * 100;
  
  // Check if this exceeds the max allowed units for this type
  const isValid = acc.maxUnits === null || unitsNeeded <= acc.maxUnits;
  
  return {
    unitsNeeded: Math.max(1, unitsNeeded),
    nodesNeeded: Math.max(1, nodesNeeded),
    efficiency: Math.min(100, efficiency),
    fits: unitsNeeded <= 1,
    isValid,
    maxUnits: acc.maxUnits,
  };
};

// Main component
export default function LLMCalculator() {
  const [url, setUrl] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [contextLength, setContextLength] = useState(4096);
  const [batchSize, setBatchSize] = useState(1);
  const [showConfig, setShowConfig] = useState(false);
  const [manualParams, setManualParams] = useState('');

  const fetchModelConfig = useCallback(async () => {
    if (!url.trim()) return;
    
    setLoading(true);
    setError(null);
    setModelInfo(null);

    try {
      // Parse the HuggingFace URL
      let repoId = '';
      const hfMatch = url.match(/huggingface\.co\/([^\/]+\/[^\/]+)/);
      if (hfMatch) {
        repoId = hfMatch[1].replace(/\/blob\/.*/, '').replace(/\/tree\/.*/, '').replace(/\/$/, '');
      } else if (url.includes('/') && !url.includes('://')) {
        // Assume it's a repo ID like "google/gemma-2b"
        repoId = url.trim();
      } else {
        setError('Please enter a valid Hugging Face URL (e.g., https://huggingface.co/meta-llama/Llama-3-8B) or repo ID (e.g., google/gemma-2b)');
        setLoading(false);
        return;
      }

      let config = null;
      let apiInfo = null;
      let configUrl = null;
      let isGated = false;

      // First, try the API endpoint - it often works even for gated models
      try {
        const infoUrl = `https://huggingface.co/api/models/${repoId}`;
        const infoResponse = await fetch(infoUrl);
        if (infoResponse.ok) {
          apiInfo = await infoResponse.json();
        } else if (infoResponse.status === 404) {
          throw new Error(`Model not found: ${repoId}`);
        }
      } catch (e) {
        if (e.message.includes('not found')) throw e;
        // Continue - we'll try config.json directly
      }

      // Try to fetch config.json or params.json
      const configUrls = [
        `https://huggingface.co/${repoId}/resolve/main/config.json`,
        `https://huggingface.co/${repoId}/raw/main/config.json`,
        `https://huggingface.co/${repoId}/resolve/main/params.json`,  // Mistral format
        `https://huggingface.co/${repoId}/raw/main/params.json`,
      ];

      for (const tryUrl of configUrls) {
        try {
          const response = await fetch(tryUrl);
          if (response.ok) {
            config = await response.json();
            configUrl = tryUrl;
            break;
          } else if (response.status === 401 || response.status === 403) {
            isGated = true;
          }
        } catch (e) {
          // Try next URL
        }
      }

      // Build model info from whatever we got
      let parsed = {
        repoId,
        configUrl,
        numParams: null,
        numLayers: null,
        hiddenSize: null,
        numHeads: null,
        numKvHeads: null,
        vocabSize: null,
        intermediateSize: null,
        contextLength: 4096,
        architecture: 'Unknown',
        modelType: 'unknown',
        isMoE: false,
        isGated,
      };

      // Parse config if we got it
      if (config) {
        parsed = { ...parsed, ...parseModelConfig(config, repoId) };
        parsed.rawConfig = config;
      }

      // Override/supplement with API info
      if (apiInfo) {
        // Get parameter count from safetensors info (most accurate)
        if (apiInfo.safetensors?.total) {
          parsed.numParams = apiInfo.safetensors.total;
        } else if (apiInfo.safetensors?.parameters) {
          // Sum all parameter counts
          const params = Object.values(apiInfo.safetensors.parameters);
          if (params.length > 0) {
            parsed.numParams = params.reduce((a, b) => a + b, 0);
          }
        }
        
        // Get architecture from tags
        if (apiInfo.tags) {
          const archTag = apiInfo.tags.find(t => 
            t.includes('llama') || t.includes('mistral') || t.includes('gemma') || 
            t.includes('gpt') || t.includes('bert') || t.includes('moe')
          );
          if (archTag && parsed.architecture === 'Unknown') {
            parsed.architecture = archTag;
          }
        }

        // Check if gated
        if (apiInfo.gated) {
          parsed.isGated = true;
        }

        // Get model card data if available
        if (apiInfo.cardData) {
          if (apiInfo.cardData.model_type) {
            parsed.modelType = apiInfo.cardData.model_type;
          }
        }
        
        parsed.apiInfo = apiInfo;
      }

      // If still no params, try to extract from model name
      if (!parsed.numParams) {
        const paramMatch = repoId.match(/(\d+\.?\d*)\s*[xX]?\s*([bBmMtT])/);
        if (paramMatch) {
          const num = parseFloat(paramMatch[1]);
          const unit = paramMatch[2].toLowerCase();
          if (unit === 't') parsed.numParams = num * 1e12;
          else if (unit === 'b') parsed.numParams = num * 1e9;
          else if (unit === 'm') parsed.numParams = num * 1e6;
        }
      }

      // Validate we have something useful
      if (!parsed.numParams && !config) {
        if (isGated) {
          setError(`This model is gated and requires authentication. The API didn't return parameter info. Try entering the parameter count manually (e.g., "675B" for Mistral Large).`);
        } else {
          setError(`Could not determine model parameters. Try entering the parameter count manually.`);
        }
        setLoading(false);
        return;
      }

      parsed.repoId = repoId;
      parsed.configUrl = configUrl;
      setModelInfo(parsed);
      
      if (parsed.contextLength) {
        setContextLength(parsed.contextLength);
      }
      
    } catch (err) {
      setError(err.message || 'Failed to fetch model configuration');
    } finally {
      setLoading(false);
    }
  }, [url]);

  const handleManualParams = () => {
    const params = parseFloat(manualParams);
    if (isNaN(params) || params <= 0) {
      setError('Please enter a valid number of parameters');
      return;
    }
    
    // Convert to actual number (handle B/T suffix)
    let numParams = params;
    if (manualParams.toLowerCase().includes('t')) {
      numParams = params * 1e12;
    } else if (manualParams.toLowerCase().includes('b')) {
      numParams = params * 1e9;
    } else if (manualParams.toLowerCase().includes('m')) {
      numParams = params * 1e6;
    } else if (params < 1000) {
      numParams = params * 1e9; // Assume billions if small number
    }
    
    setModelInfo({
      numParams,
      architecture: 'Manual Entry',
      modelType: 'custom',
      repoId: `Custom ${formatNumber(numParams)} model`,
    });
    setError(null);
  };

  return (
    <div className="min-h-screen bg-[#0a0a0f] text-gray-100 font-mono">
      {/* Background pattern */}
      <div className="fixed inset-0 opacity-[0.03]" style={{
        backgroundImage: `url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='1'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E")`,
      }} />
      
      <div className="relative z-10 max-w-7xl mx-auto px-4 py-8 sm:px-6 lg:px-8">
        {/* Header */}
        <header className="mb-12 text-center">
          <div className="inline-flex items-center gap-3 mb-4">
            <div className="p-3 bg-gradient-to-br from-emerald-500/20 to-cyan-500/20 rounded-xl border border-emerald-500/30">
              <Cpu className="w-8 h-8 text-emerald-400" />
            </div>
            <h1 className="text-3xl sm:text-4xl font-bold bg-gradient-to-r from-emerald-400 via-cyan-400 to-blue-400 bg-clip-text text-transparent">
              LLM Hardware Calculator
            </h1>
          </div>
          <p className="text-gray-400 max-w-2xl mx-auto">
            Calculate memory and hardware requirements for running large language models. 
            Supports NVIDIA GPUs, Google TPUs, and Apple Silicon.
          </p>
        </header>

        {/* Input Section */}
        <div className="bg-gray-900/50 backdrop-blur-sm rounded-2xl border border-gray-800 p-6 mb-8">
          <div className="space-y-4">
            <div>
              <label className="block text-sm text-gray-400 mb-2">Hugging Face Model URL</label>
              <div className="flex gap-3">
                <input
                  type="text"
                  value={url}
                  onChange={(e) => setUrl(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && fetchModelConfig()}
                  placeholder="https://huggingface.co/meta-llama/Llama-3-8B"
                  className="flex-1 bg-gray-800/50 border border-gray-700 rounded-lg px-4 py-3 text-gray-100 placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-emerald-500/50 focus:border-emerald-500/50 transition-all"
                />
                <button
                  onClick={fetchModelConfig}
                  disabled={loading || !url.trim()}
                  className="px-6 py-3 bg-gradient-to-r from-emerald-600 to-cyan-600 hover:from-emerald-500 hover:to-cyan-500 disabled:from-gray-700 disabled:to-gray-700 rounded-lg font-medium transition-all flex items-center gap-2 disabled:cursor-not-allowed"
                >
                  {loading ? (
                    <Loader2 className="w-5 h-5 animate-spin" />
                  ) : (
                    <Zap className="w-5 h-5" />
                  )}
                  Analyze
                </button>
              </div>
            </div>

            <div className="flex items-center gap-4 text-sm text-gray-500">
              <span>or</span>
              <div className="flex items-center gap-2">
                <input
                  type="text"
                  value={manualParams}
                  onChange={(e) => setManualParams(e.target.value)}
                  placeholder="e.g., 70B, 405B, 671B"
                  className="bg-gray-800/50 border border-gray-700 rounded-lg px-3 py-2 text-gray-100 placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-emerald-500/50 w-40"
                />
                <button
                  onClick={handleManualParams}
                  className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg text-sm transition-all"
                >
                  Use Manual
                </button>
              </div>
            </div>
          </div>

          {error && (
            <div className="mt-4 p-4 bg-red-500/10 border border-red-500/30 rounded-lg flex items-start gap-3">
              <AlertCircle className="w-5 h-5 text-red-400 mt-0.5 flex-shrink-0" />
              <p className="text-red-400 text-sm">{error}</p>
            </div>
          )}
        </div>

        {modelInfo && (
          <>
            {/* Model Info Card */}
            <div className="bg-gray-900/50 backdrop-blur-sm rounded-2xl border border-gray-800 p-6 mb-8">
              <div className="flex items-start justify-between mb-6">
                <div>
                  <div className="flex items-center gap-2 mb-2">
                    <CheckCircle2 className="w-5 h-5 text-emerald-400" />
                    <h2 className="text-xl font-semibold text-gray-100">Model Loaded</h2>
                  </div>
                  <p className="text-gray-400 text-sm">{modelInfo.repoId}</p>
                </div>
                {modelInfo.configUrl && (
                  <a
                    href={modelInfo.configUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-gray-400 hover:text-emerald-400 transition-colors"
                  >
                    <ExternalLink className="w-5 h-5" />
                  </a>
                )}
              </div>

              <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-6">
                <div className="bg-gray-800/50 rounded-xl p-4">
                  <p className="text-gray-500 text-xs uppercase tracking-wider mb-1">Parameters</p>
                  <p className="text-2xl font-bold text-emerald-400">
                    {modelInfo.numParams ? formatNumber(modelInfo.numParams) : 'Unknown'}
                  </p>
                </div>
                <div className="bg-gray-800/50 rounded-xl p-4">
                  <p className="text-gray-500 text-xs uppercase tracking-wider mb-1">Architecture</p>
                  <p className="text-lg font-medium text-gray-200 truncate">{modelInfo.architecture}</p>
                </div>
                <div className="bg-gray-800/50 rounded-xl p-4">
                  <p className="text-gray-500 text-xs uppercase tracking-wider mb-1">Layers</p>
                  <p className="text-2xl font-bold text-cyan-400">{modelInfo.numLayers || '—'}</p>
                </div>
                <div className="bg-gray-800/50 rounded-xl p-4">
                  <p className="text-gray-500 text-xs uppercase tracking-wider mb-1">Hidden Size</p>
                  <p className="text-2xl font-bold text-blue-400">{modelInfo.hiddenSize?.toLocaleString() || '—'}</p>
                </div>
              </div>

              {modelInfo.isMoE && (
                <div className="bg-purple-500/10 border border-purple-500/30 rounded-xl p-4 mb-6">
                  <p className="text-purple-400 font-medium">
                    🔀 Mixture of Experts Model: {modelInfo.numExperts} experts, {modelInfo.numActiveExperts} active per token
                  </p>
                </div>
              )}

              {modelInfo.isGated && (
                <div className="bg-amber-500/10 border border-amber-500/30 rounded-xl p-4 mb-6">
                  <p className="text-amber-400 font-medium">
                    🔒 Gated Model — Config details limited. Parameter count from API metadata.
                  </p>
                </div>
              )}

              {modelInfo.isMultimodal && (
                <div className="bg-blue-500/10 border border-blue-500/30 rounded-xl p-4 mb-6">
                  <p className="text-blue-400 font-medium">
                    🖼️ Multimodal Model — Includes vision encoder
                    {modelInfo.visionParams && ` (~${formatNumber(modelInfo.visionParams)} vision params)`}
                  </p>
                </div>
              )}

              {/* Inference Settings */}
              <div className="flex flex-wrap gap-6">
                <div>
                  <label className="block text-sm text-gray-400 mb-2">Context Length</label>
                  <select
                    value={contextLength}
                    onChange={(e) => setContextLength(Number(e.target.value))}
                    className="bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-gray-100 focus:outline-none focus:ring-2 focus:ring-emerald-500/50"
                  >
                    <option value={2048}>2K tokens</option>
                    <option value={4096}>4K tokens</option>
                    <option value={8192}>8K tokens</option>
                    <option value={16384}>16K tokens</option>
                    <option value={32768}>32K tokens</option>
                    <option value={65536}>64K tokens</option>
                    <option value={131072}>128K tokens</option>
                    <option value={262144}>256K tokens</option>
                  </select>
                </div>
                <div>
                  <label className="block text-sm text-gray-400 mb-2">Batch Size</label>
                  <select
                    value={batchSize}
                    onChange={(e) => setBatchSize(Number(e.target.value))}
                    className="bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-gray-100 focus:outline-none focus:ring-2 focus:ring-emerald-500/50"
                  >
                    <option value={1}>1</option>
                    <option value={2}>2</option>
                    <option value={4}>4</option>
                    <option value={8}>8</option>
                    <option value={16}>16</option>
                    <option value={32}>32</option>
                  </select>
                </div>
              </div>

              {/* Raw Config Toggle */}
              {modelInfo.rawConfig && (
                <div className="mt-6">
                  <button
                    onClick={() => setShowConfig(!showConfig)}
                    className="flex items-center gap-2 text-gray-400 hover:text-gray-200 transition-colors text-sm"
                  >
                    {showConfig ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                    {showConfig ? 'Hide' : 'Show'} raw config.json
                  </button>
                  {showConfig && (
                    <pre className="mt-4 bg-gray-950 rounded-xl p-4 overflow-x-auto text-xs text-gray-400 max-h-96 overflow-y-auto">
                      {JSON.stringify(modelInfo.rawConfig, null, 2)}
                    </pre>
                  )}
                </div>
              )}
            </div>

            {/* Memory Requirements */}
            <div className="bg-gray-900/50 backdrop-blur-sm rounded-2xl border border-gray-800 p-6 mb-8">
              <h2 className="text-xl font-semibold text-gray-100 mb-6 flex items-center gap-2">
                <HardDrive className="w-5 h-5 text-cyan-400" />
                Memory Requirements by Precision
              </h2>

              <div className="space-y-4">
                {PRECISIONS.map((precision) => {
                  const memory = calculateMemory(modelInfo, precision, contextLength, batchSize);
                  if (!memory) return null;

                  return (
                    <div
                      key={precision.name}
                      className="bg-gray-800/50 rounded-xl p-4 border border-gray-700/50"
                    >
                      <div className="flex items-center justify-between mb-3">
                        <div className="flex items-center gap-3">
                          <div
                            className="w-3 h-3 rounded-full"
                            style={{ backgroundColor: precision.color }}
                          />
                          <div>
                            <span className="font-medium text-gray-100">{precision.name}</span>
                            <span className="text-gray-500 text-sm ml-2">({precision.description})</span>
                          </div>
                        </div>
                        <span className="text-xl font-bold" style={{ color: precision.color }}>
                          {formatBytes(memory.withOverhead)}
                        </span>
                      </div>
                      
                      <div className="grid grid-cols-3 gap-4 text-sm">
                        <div>
                          <span className="text-gray-500">Weights:</span>
                          <span className="text-gray-300 ml-2">{formatBytes(memory.weights)}</span>
                        </div>
                        <div>
                          <span className="text-gray-500">KV Cache:</span>
                          <span className="text-gray-300 ml-2">{formatBytes(memory.kvCache)}</span>
                        </div>
                        <div>
                          <span className="text-gray-500">Activations:</span>
                          <span className="text-gray-300 ml-2">{formatBytes(memory.activations)}</span>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Hardware Requirements */}
            <div className="bg-gray-900/50 backdrop-blur-sm rounded-2xl border border-gray-800 p-6">
              <h2 className="text-xl font-semibold text-gray-100 mb-6 flex items-center gap-2">
                <Server className="w-5 h-5 text-emerald-400" />
                Hardware Requirements
              </h2>

              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-gray-700">
                      <th className="text-left py-3 px-4 text-gray-400 font-medium text-sm">Accelerator</th>
                      <th className="text-center py-3 px-4 text-gray-400 font-medium text-sm">Memory</th>
                      {PRECISIONS.slice(1, 4).map((p) => (
                        <th key={p.name} className="text-center py-3 px-4 font-medium text-sm" style={{ color: p.color }}>
                          {p.name}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {ACCELERATORS.map((acc) => (
                      <tr key={acc.name} className="border-b border-gray-800/50 hover:bg-gray-800/30 transition-colors">
                        <td className="py-3 px-4">
                          <div className="font-medium text-gray-200">
                            {acc.name}
                            {acc.type === 'tpu' && <span className="ml-2 text-xs px-1.5 py-0.5 bg-blue-500/20 text-blue-400 rounded">TPU</span>}
                          </div>
                          <div className="text-xs text-gray-500">
                            {acc.interconnect}
                            {acc.maxUnits !== null 
                              ? ` • max ${acc.maxUnits}` 
                              : ` • ${acc.perNode}/node`}
                          </div>
                        </td>
                        <td className="text-center py-3 px-4 text-gray-400">{acc.memory}GB</td>
                        {PRECISIONS.slice(1, 4).map((precision) => {
                          const memory = calculateMemory(modelInfo, precision, contextLength, batchSize);
                          if (!memory) return <td key={precision.name} className="text-center py-3 px-4 text-gray-500">—</td>;
                          
                          const req = calculateAcceleratorRequirements(memory.withOverhead, acc);
                          
                          // Show red X for invalid/impossible configurations
                          if (!req.isValid) {
                            return (
                              <td key={precision.name} className="text-center py-3 px-4">
                                <div className="inline-flex flex-col items-center text-red-500">
                                  <span className="text-lg">✗</span>
                                  <span className="text-xs text-red-400/70">
                                    needs {req.unitsNeeded} (max {req.maxUnits})
                                  </span>
                                </div>
                              </td>
                            );
                          }
                          
                          // Determine unit label based on type
                          const unitLabel = acc.type === 'tpu' ? 'chip' : (acc.type === 'soc' ? 'device' : 'GPU');
                          const unitsLabel = acc.type === 'tpu' ? 'chips' : (acc.type === 'soc' ? 'devices' : 'GPUs');
                          
                          return (
                            <td key={precision.name} className="text-center py-3 px-4">
                              <div className={`inline-flex flex-col items-center ${req.fits ? 'text-emerald-400' : 'text-gray-300'}`}>
                                <span className="font-bold text-lg">{req.unitsNeeded}</span>
                                <span className="text-xs text-gray-500">
                                  {req.unitsNeeded === 1 
                                    ? `single ${unitLabel}` 
                                    : req.nodesNeeded > 1 
                                      ? `${req.nodesNeeded} nodes` 
                                      : `${req.unitsNeeded} ${unitsLabel}`}
                                </span>
                              </div>
                            </td>
                          );
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <p className="mt-6 text-gray-500 text-sm">
                * Calculations include model weights, KV cache for {contextLength.toLocaleString()} tokens with batch size {batchSize}, 
                activation memory, and 10% overhead. Actual requirements may vary based on framework and optimizations.
                <br />
                <span className="text-red-400/70">✗</span> = Exceeds practical limit for that hardware type.
              </p>
            </div>
          </>
        )}

        {/* Example Models */}
        {!modelInfo && (
          <div className="bg-gray-900/50 backdrop-blur-sm rounded-2xl border border-gray-800 p-6">
            <h2 className="text-lg font-semibold text-gray-100 mb-4">Try these models</h2>
            <div className="flex flex-wrap gap-3">
              {[
                { name: 'Gemma 2 2B', url: 'https://huggingface.co/google/gemma-2-2b' },
                { name: 'Qwen2.5 7B', url: 'https://huggingface.co/Qwen/Qwen2.5-7B' },
                { name: 'Mistral 7B v0.3', url: 'https://huggingface.co/mistralai/Mistral-7B-v0.3' },
                { name: 'Llama 3.1 70B', url: 'https://huggingface.co/meta-llama/Llama-3.1-70B' },
                { name: 'Qwen2.5 72B', url: 'https://huggingface.co/Qwen/Qwen2.5-72B' },
                { name: 'DeepSeek V3', url: 'https://huggingface.co/deepseek-ai/DeepSeek-V3' },
                { name: 'Llama 3.1 405B', url: 'https://huggingface.co/meta-llama/Llama-3.1-405B' },
                { name: 'Mistral Large 3 675B', url: 'https://huggingface.co/mistralai/Mistral-Large-3-675B-Instruct-2512' },
              ].map((model) => (
                <button
                  key={model.name}
                  onClick={() => setUrl(model.url)}
                  className="px-4 py-2 bg-gray-800 hover:bg-gray-700 border border-gray-700 hover:border-emerald-500/50 rounded-lg text-sm text-gray-300 hover:text-emerald-400 transition-all"
                >
                  {model.name}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Footer */}
        <footer className="mt-12 text-center text-gray-500 text-sm">
          <p>Built for ML engineers. Memory calculations are estimates for inference workloads.</p>
        </footer>
      </div>
    </div>
  );
}
