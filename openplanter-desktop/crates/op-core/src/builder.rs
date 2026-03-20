/// Engine construction and provider inference.
///
/// Mirrors `agent/builder.py` — provider detection, model validation,
/// and engine factory. All provider metadata comes from the registry.
use std::collections::HashMap;

use crate::config::{AgentConfig, PROVIDER_DEFAULT_MODELS};
use crate::model::BaseModel;
use crate::model::openai::OpenAIModel;
use crate::model::anthropic::AnthropicModel;
use crate::providers::{self, infer_provider_for_model, get_provider, all_model_providers};

/// Error type for model/builder operations.
#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("{0}")]
    Message(String),
}

/// Validate that a model name is compatible with the given provider.
pub fn validate_model_provider(model_name: &str, provider: &str) -> Result<(), ModelError> {
    if provider == "openrouter" {
        return Ok(());
    }
    let inferred = infer_provider_for_model(model_name);
    match inferred {
        None | Some("openrouter") => Ok(()),
        Some(p) if p == provider => Ok(()),
        Some(p) => Err(ModelError::Message(format!(
            "Model '{}' belongs to provider '{}', not '{}'. \
             Use --provider {} or pick a model that matches the current provider.",
            model_name, p, provider, p
        ))),
    }
}

/// Resolve the model name from config, handling the "newest" keyword.
pub fn resolve_model_name(cfg: &AgentConfig) -> Result<String, ModelError> {
    let selected = cfg.model.trim();
    if !selected.is_empty() && selected.to_lowercase() != "newest" {
        return Ok(selected.to_string());
    }
    Ok(PROVIDER_DEFAULT_MODELS
        .get(cfg.provider.as_str())
        .unwrap_or(&"claude-opus-4-6")
        .to_string())
}

/// Resolve the provider, handling "auto" by inferring from model name
/// or falling back to the first provider with an available API key.
pub fn resolve_provider(cfg: &AgentConfig) -> Result<String, ModelError> {
    let provider = cfg.provider.trim().to_lowercase();
    if !provider.is_empty() && provider != "auto" {
        return Ok(provider);
    }

    // Try to infer from model name
    let model = cfg.model.trim();
    if !model.is_empty() {
        if let Some(inferred) = infer_provider_for_model(model) {
            return Ok(inferred.to_string());
        }
    }

    // Fallback: first provider with an available key, in inference-priority order
    let ordered = providers::provider_names();
    for name in ordered {
        if name == "ollama" {
            continue; // ollama is last (no key needed)
        }
        if let Some(provider) = get_provider(name) {
            for env_var in provider.api_key_env {
                if let Ok(val) = std::env::var(env_var) {
                    if !val.trim().is_empty() {
                        return Ok(name.to_string());
                    }
                }
            }
            // Also check config-level API key fields
            let has_key = match name {
                "openai" => cfg.openai_api_key.is_some(),
                "anthropic" => cfg.anthropic_api_key.is_some(),
                "openrouter" => cfg.openrouter_api_key.is_some(),
                "cerebras" => cfg.cerebras_api_key.is_some(),
                "kilo" => cfg.kilo_api_key.is_some(),
                "zai" => cfg.zai_api_key.is_some(),
                "opencode-go" => cfg.opencodego_api_key.is_some(),
                _ => false,
            };
            if has_key {
                return Ok(name.to_string());
            }
        }
    }

    // Default to ollama (no key needed)
    Ok("ollama".to_string())
}

/// Resolve the base URL and API key for the given provider.
///
/// Uses the registry for default URL and env var names, with config overrides.
pub fn resolve_endpoint(
    cfg: &AgentConfig,
    provider: &str,
) -> Result<(String, String), ModelError> {
    let p = get_provider(provider)
        .ok_or_else(|| ModelError::Message(format!("Unknown provider: {provider}")))?;

    // Resolve API key: config field first, then env, then registry env vars
    let key = match provider {
        "openai" => cfg.openai_api_key.clone().or(cfg.api_key.clone()),
        "anthropic" => cfg.anthropic_api_key.clone().or(cfg.api_key.clone()),
        "openrouter" => cfg.openrouter_api_key.clone().or(cfg.api_key.clone()),
        "cerebras" => cfg.cerebras_api_key.clone().or(cfg.api_key.clone()),
        "kilo" => cfg.kilo_api_key.clone().or(cfg.api_key.clone()),
        "zai" => cfg.zai_api_key.clone().or(cfg.api_key.clone()),
        "opencode-go" => cfg.opencodego_api_key.clone().or(cfg.api_key.clone()),
        "ollama" => Some("ollama".to_string()), // no key needed
        _ => cfg.api_key.clone(),
    };

    let key = key.filter(|k| !k.is_empty()).or_else(|| {
        providers::resolve_api_key_from_env(provider)
    });

    let key = key.ok_or_else(|| {
        let env_names = p.api_key_env.join(" or ");
        if env_names.is_empty() {
            ModelError::Message(format!("No API key for provider '{provider}'."))
        } else {
            ModelError::Message(format!(
                "No {provider} API key. Set {env_names}."
            ))
        }
    })?;

    // Resolve base URL: config field first, then registry default
    let base_url = match provider {
        "openai" => cfg.openai_base_url.clone(),
        "anthropic" => cfg.anthropic_base_url.clone(),
        "openrouter" => cfg.openrouter_base_url.clone(),
        "cerebras" => cfg.cerebras_base_url.clone(),
        "kilo" => cfg.kilo_base_url.clone(),
        "zai" => cfg.zai_base_url.clone(),
        "opencode-go" => cfg.opencodego_base_url.clone(),
        "ollama" => cfg.ollama_base_url.clone(),
        _ => p.default_base_url.to_string(),
    };

    Ok((base_url, key))
}

/// Build a model instance from the agent configuration.
pub fn build_model(cfg: &AgentConfig) -> Result<Box<dyn BaseModel>, ModelError> {
    let provider = resolve_provider(cfg)?;
    let model_name = resolve_model_name(cfg)?;
    validate_model_provider(&model_name, &provider)?;
    let (base_url, api_key) = resolve_endpoint(cfg, &provider)?;

    match provider.as_str() {
        "anthropic" => Ok(Box::new(AnthropicModel::new(
            model_name,
            base_url,
            api_key,
            cfg.reasoning_effort.clone(),
        ))),
        _ => {
            // OpenAI-compatible: openai, openrouter, cerebras, ollama, kilo, zai, opencode-go
            let mut extra_headers = HashMap::new();
            if provider == "openrouter" {
                extra_headers.insert(
                    "HTTP-Referer".to_string(),
                    "https://github.com/openplanter".to_string(),
                );
                extra_headers.insert("X-Title".to_string(), "OpenPlanter".to_string());
            }
            Ok(Box::new(OpenAIModel::new(
                model_name,
                provider,
                base_url,
                api_key,
                cfg.reasoning_effort.clone(),
                extra_headers,
            )))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_infer_anthropic() {
        assert_eq!(infer_provider_for_model("claude-opus-4-6"), Some("anthropic"));
        assert_eq!(infer_provider_for_model("claude-sonnet-4-5"), Some("anthropic"));
        assert_eq!(infer_provider_for_model("claude-haiku-4-5"), Some("anthropic"));
    }

    #[test]
    fn test_infer_openai() {
        assert_eq!(infer_provider_for_model("gpt-5.2"), Some("openai"));
        assert_eq!(infer_provider_for_model("o1-preview"), Some("openai"));
        assert_eq!(infer_provider_for_model("o3"), Some("openai"));
        assert_eq!(infer_provider_for_model("chatgpt-4o"), Some("openai"));
    }

    #[test]
    fn test_infer_openrouter() {
        assert_eq!(infer_provider_for_model("anthropic/claude-sonnet-4-5"), Some("openrouter"));
        assert_eq!(infer_provider_for_model("openai/gpt-5.2"), Some("openrouter"));
    }

    #[test]
    fn test_infer_cerebras() {
        assert_eq!(infer_provider_for_model("qwen-3-235b"), Some("cerebras"));
    }

    #[test]
    fn test_infer_ollama() {
        assert_eq!(infer_provider_for_model("llama3.2"), Some("ollama"));
        assert_eq!(infer_provider_for_model("mistral"), Some("ollama"));
        assert_eq!(infer_provider_for_model("phi"), Some("ollama"));
        assert_eq!(infer_provider_for_model("deepseek"), Some("ollama"));
        assert_eq!(infer_provider_for_model("qwen2"), Some("ollama"));
    }

    #[test]
    fn test_cerebras_before_ollama_qwen() {
        assert_eq!(infer_provider_for_model("qwen-3"), Some("cerebras"));
        assert_eq!(infer_provider_for_model("qwen2"), Some("ollama"));
    }

    #[test]
    fn test_infer_kilo() {
        assert_eq!(infer_provider_for_model("kilo-auto/frontier"), Some("kilo"));
        assert_eq!(infer_provider_for_model("kilo-auto/balanced"), Some("kilo"));
    }

    #[test]
    fn test_infer_zai() {
        assert_eq!(infer_provider_for_model("glm-5"), Some("zai"));
        assert_eq!(infer_provider_for_model("glm-4.7"), Some("zai"));
        assert_eq!(infer_provider_for_model("glm-4.5-air"), Some("zai"));
    }

    #[test]
    fn test_infer_opencode_go() {
        assert_eq!(infer_provider_for_model("opencode-go/glm-5"), Some("opencode-go"));
        assert_eq!(infer_provider_for_model("opencode-go/kimi-k2.5"), Some("opencode-go"));
    }

    #[test]
    fn test_infer_unknown() {
        assert_eq!(infer_provider_for_model("some-random-model"), None);
    }

    #[test]
    fn test_validate_model_provider_ok() {
        assert!(validate_model_provider("gpt-5.2", "openai").is_ok());
        assert!(validate_model_provider("claude-opus-4-6", "anthropic").is_ok());
        assert!(validate_model_provider("anthropic/claude-sonnet-4-5", "openrouter").is_ok());
        assert!(validate_model_provider("kilo-auto/balanced", "kilo").is_ok());
        assert!(validate_model_provider("glm-5", "zai").is_ok());
        assert!(validate_model_provider("opencode-go/glm-5", "opencode-go").is_ok());
    }

    #[test]
    fn test_validate_model_provider_mismatch() {
        let result = validate_model_provider("gpt-5.2", "anthropic");
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("openai"));
        assert!(err.contains("anthropic"));
    }

    #[test]
    fn test_resolve_model_name_explicit() {
        let cfg = AgentConfig {
            model: "gpt-5.2".into(),
            provider: "openai".into(),
            ..Default::default()
        };
        assert_eq!(resolve_model_name(&cfg).unwrap(), "gpt-5.2");
    }

    #[test]
    fn test_resolve_model_name_default() {
        let cfg = AgentConfig {
            model: "".into(),
            provider: "openai".into(),
            ..Default::default()
        };
        assert_eq!(resolve_model_name(&cfg).unwrap(), "gpt-5.2");
    }

    #[test]
    fn test_resolve_provider_explicit() {
        let cfg = AgentConfig {
            provider: "openai".into(),
            ..Default::default()
        };
        assert_eq!(resolve_provider(&cfg).unwrap(), "openai");
    }

    #[test]
    fn test_resolve_provider_auto_infers_from_model() {
        let cfg = AgentConfig {
            provider: "auto".into(),
            model: "claude-opus-4-6".into(),
            ..Default::default()
        };
        assert_eq!(resolve_provider(&cfg).unwrap(), "anthropic");
    }

    #[test]
    fn test_resolve_provider_auto_falls_back_to_key() {
        let cfg = AgentConfig {
            provider: "auto".into(),
            model: "some-unknown-model".into(),
            openai_api_key: Some("sk-test".into()),
            ..Default::default()
        };
        assert_eq!(resolve_provider(&cfg).unwrap(), "openai");
    }

    #[test]
    fn test_resolve_provider_auto_no_keys_defaults_ollama() {
        let cfg = AgentConfig {
            provider: "auto".into(),
            model: "some-unknown-model".into(),
            ..Default::default()
        };
        assert_eq!(resolve_provider(&cfg).unwrap(), "ollama");
    }

    #[test]
    fn test_resolve_provider_anthropic_key_preferred_first() {
        let cfg = AgentConfig {
            provider: "auto".into(),
            model: "some-unknown-model".into(),
            anthropic_api_key: Some("sk-ant-test".into()),
            openai_api_key: Some("sk-test".into()),
            ..Default::default()
        };
        assert_eq!(resolve_provider(&cfg).unwrap(), "anthropic");
    }

    #[test]
    fn test_resolve_endpoint_anthropic() {
        let cfg = AgentConfig {
            anthropic_api_key: Some("sk-ant-key".into()),
            ..Default::default()
        };
        let (url, key) = resolve_endpoint(&cfg, "anthropic").unwrap();
        assert_eq!(url, "https://api.anthropic.com/v1");
        assert_eq!(key, "sk-ant-key");
    }

    #[test]
    fn test_resolve_endpoint_anthropic_fallback_to_api_key() {
        let cfg = AgentConfig {
            api_key: Some("fallback-key".into()),
            ..Default::default()
        };
        let (_, key) = resolve_endpoint(&cfg, "anthropic").unwrap();
        assert_eq!(key, "fallback-key");
    }

    #[test]
    fn test_resolve_endpoint_anthropic_missing_key() {
        let cfg = AgentConfig::default();
        let result = resolve_endpoint(&cfg, "anthropic");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("ANTHROPIC_API_KEY"));
    }

    #[test]
    fn test_resolve_endpoint_openai() {
        let cfg = AgentConfig {
            openai_api_key: Some("sk-openai".into()),
            ..Default::default()
        };
        let (url, key) = resolve_endpoint(&cfg, "openai").unwrap();
        assert_eq!(url, "https://api.openai.com/v1");
        assert_eq!(key, "sk-openai");
    }

    #[test]
    fn test_resolve_endpoint_kilo() {
        let cfg = AgentConfig {
            kilo_api_key: Some("kilo-key".into()),
            ..Default::default()
        };
        let (url, key) = resolve_endpoint(&cfg, "kilo").unwrap();
        assert_eq!(url, "https://api.kilo.ai/api/gateway");
        assert_eq!(key, "kilo-key");
    }

    #[test]
    fn test_resolve_endpoint_zai() {
        let cfg = AgentConfig {
            zai_api_key: Some("zai-key".into()),
            ..Default::default()
        };
        let (url, key) = resolve_endpoint(&cfg, "zai").unwrap();
        assert_eq!(url, "https://api.z.ai/api/coding/paas/v4");
        assert_eq!(key, "zai-key");
    }

    #[test]
    fn test_resolve_endpoint_opencode_go() {
        let cfg = AgentConfig {
            opencodego_api_key: Some("ocgo-key".into()),
            ..Default::default()
        };
        let (url, key) = resolve_endpoint(&cfg, "opencode-go").unwrap();
        assert_eq!(url, "https://opencode.ai/zen/go/v1");
        assert_eq!(key, "ocgo-key");
    }

    #[test]
    fn test_resolve_endpoint_ollama_dummy_key() {
        let cfg = AgentConfig::default();
        let (url, key) = resolve_endpoint(&cfg, "ollama").unwrap();
        assert!(url.contains("11434"));
        assert_eq!(key, "ollama");
    }

    #[test]
    fn test_resolve_endpoint_unknown_provider() {
        let cfg = AgentConfig::default();
        let result = resolve_endpoint(&cfg, "unknown");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Unknown provider"));
    }

    #[test]
    fn test_build_model_anthropic() {
        let cfg = AgentConfig {
            provider: "anthropic".into(),
            model: "claude-opus-4-6".into(),
            anthropic_api_key: Some("sk-ant-key".into()),
            ..Default::default()
        };
        let model = build_model(&cfg).unwrap();
        assert_eq!(model.model_name(), "claude-opus-4-6");
        assert_eq!(model.provider_name(), "anthropic");
    }

    #[test]
    fn test_build_model_openai() {
        let cfg = AgentConfig {
            provider: "openai".into(),
            model: "gpt-4o".into(),
            openai_api_key: Some("sk-key".into()),
            ..Default::default()
        };
        let model = build_model(&cfg).unwrap();
        assert_eq!(model.model_name(), "gpt-4o");
        assert_eq!(model.provider_name(), "openai");
    }

    #[test]
    fn test_build_model_ollama_no_key_needed() {
        let cfg = AgentConfig {
            provider: "ollama".into(),
            model: "llama3.2".into(),
            ..Default::default()
        };
        let model = build_model(&cfg).unwrap();
        assert_eq!(model.model_name(), "llama3.2");
        assert_eq!(model.provider_name(), "ollama");
    }

    #[test]
    fn test_build_model_auto_anthropic() {
        let cfg = AgentConfig {
            provider: "auto".into(),
            model: "claude-sonnet-4-5".into(),
            anthropic_api_key: Some("sk-ant-key".into()),
            ..Default::default()
        };
        let model = build_model(&cfg).unwrap();
        assert_eq!(model.provider_name(), "anthropic");
        assert_eq!(model.model_name(), "claude-sonnet-4-5");
    }

    #[test]
    fn test_build_model_missing_key_errors() {
        let cfg = AgentConfig {
            provider: "openai".into(),
            model: "gpt-4o".into(),
            ..Default::default()
        };
        let result = build_model(&cfg);
        assert!(result.is_err());
    }

    #[test]
    fn test_build_model_mismatch_errors() {
        let cfg = AgentConfig {
            provider: "anthropic".into(),
            model: "gpt-4o".into(),
            anthropic_api_key: Some("key".into()),
            ..Default::default()
        };
        let result = build_model(&cfg);
        assert!(result.is_err());
        let err_msg = match result {
            Err(e) => e.to_string(),
            Ok(_) => panic!("expected error"),
        };
        assert!(err_msg.contains("openai"), "error should mention openai: {err_msg}");
    }

    #[test]
    fn test_build_model_openrouter_has_extra_headers() {
        let cfg = AgentConfig {
            provider: "openrouter".into(),
            model: "anthropic/claude-sonnet-4-5".into(),
            openrouter_api_key: Some("sk-or-key".into()),
            ..Default::default()
        };
        let model = build_model(&cfg).unwrap();
        assert_eq!(model.provider_name(), "openrouter");
    }
}
