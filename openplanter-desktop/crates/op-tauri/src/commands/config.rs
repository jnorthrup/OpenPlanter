use std::collections::HashMap;
use tauri::State;
use crate::state::AppState;
use op_core::events::{ConfigView, ModelInfo, PartialConfig};
use op_core::settings::{PersistentSettings, SettingsStore};
use op_core::credentials::credentials_from_env;
use op_core::providers::{self, get_provider, all_model_providers};

/// Get the current configuration.
#[tauri::command]
pub async fn get_config(
    state: State<'_, AppState>,
) -> Result<ConfigView, String> {
    let cfg = state.config.lock().await;
    let session_id = state.session_id.lock().await;
    Ok(ConfigView {
        provider: cfg.provider.clone(),
        model: cfg.model.clone(),
        reasoning_effort: cfg.reasoning_effort.clone(),
        workspace: cfg.workspace.display().to_string(),
        session_id: session_id.clone(),
        recursive: cfg.recursive,
        max_depth: cfg.max_depth,
        max_steps_per_call: cfg.max_steps_per_call,
        demo: cfg.demo,
    })
}

/// Update configuration fields.
#[tauri::command]
pub async fn update_config(
    partial: PartialConfig,
    state: State<'_, AppState>,
) -> Result<ConfigView, String> {
    let mut cfg = state.config.lock().await;
    if let Some(provider) = partial.provider {
        cfg.provider = provider;
    }
    if let Some(model) = partial.model {
        cfg.model = model;
    }
    if let Some(effort) = partial.reasoning_effort {
        cfg.reasoning_effort = if effort.is_empty() {
            None
        } else {
            Some(effort)
        };
    }
    let session_id = state.session_id.lock().await;
    Ok(ConfigView {
        provider: cfg.provider.clone(),
        model: cfg.model.clone(),
        reasoning_effort: cfg.reasoning_effort.clone(),
        workspace: cfg.workspace.display().to_string(),
        session_id: session_id.clone(),
        recursive: cfg.recursive,
        max_depth: cfg.max_depth,
        max_steps_per_call: cfg.max_steps_per_call,
        demo: cfg.demo,
    })
}

/// Look up known models for a provider from the registry.
fn known_models_for_provider(provider: &str) -> Vec<ModelInfo> {
    match get_provider(provider) {
        Some(p) => p.known_models
            .iter()
            .map(|m| ModelInfo {
                id: m.id.to_string(),
                name: Some(m.display_name.to_string()),
                provider: provider.to_string(),
            })
            .collect(),
        None => vec![],
    }
}

/// List available models for a provider.
#[tauri::command]
pub async fn list_models(
    provider: String,
    _state: State<'_, AppState>,
) -> Result<Vec<ModelInfo>, String> {
    if provider == "all" {
        let mut all = Vec::new();
        for p in all_model_providers() {
            all.extend(known_models_for_provider(p));
        }
        Ok(all)
    } else {
        Ok(known_models_for_provider(&provider))
    }
}

/// Save persistent settings to disk.
#[tauri::command]
pub async fn save_settings(
    settings: PersistentSettings,
    state: State<'_, AppState>,
) -> Result<(), String> {
    let cfg = state.config.lock().await;
    let store = SettingsStore::new(&cfg.workspace, &cfg.session_root_dir);
    store.save(&settings).map_err(|e| e.to_string())
}

/// API key field accessor for AgentConfig by provider name.
fn api_key_for_provider(cfg: &op_core::config::AgentConfig, provider: &str) -> bool {
    match provider {
        "openai" => cfg.openai_api_key.is_some(),
        "anthropic" => cfg.anthropic_api_key.is_some(),
        "openrouter" => cfg.openrouter_api_key.is_some(),
        "cerebras" => cfg.cerebras_api_key.is_some(),
        "ollama" => true, // never needs a key
        "kilo" => cfg.kilo_api_key.is_some(),
        "zai" => cfg.zai_api_key.is_some(),
        "opencode-go" => cfg.opencodego_api_key.is_some(),
        "exa" => cfg.exa_api_key.is_some(),
        _ => false,
    }
}

/// Build credential status from config: which providers/services have API keys configured.
pub fn build_credential_status(cfg: &op_core::config::AgentConfig) -> HashMap<String, bool> {
    let mut status = HashMap::new();
    for name in all_model_providers() {
        status.insert(name.to_string(), api_key_for_provider(cfg, name));
    }
    status.insert("exa".to_string(), cfg.exa_api_key.is_some());
    status
}

/// Get credential status: which providers/services have API keys configured.
#[tauri::command]
pub async fn get_credentials_status(
    state: State<'_, AppState>,
) -> Result<HashMap<String, bool>, String> {
    let cfg = state.config.lock().await;
    let env_creds = credentials_from_env();

    let mut status = HashMap::new();
    for name in all_model_providers() {
        let config_has = api_key_for_provider(&cfg, name);
        let env_has = match name {
            "openai" => env_creds.openai_api_key.is_some(),
            "anthropic" => env_creds.anthropic_api_key.is_some(),
            "openrouter" => env_creds.openrouter_api_key.is_some(),
            "cerebras" => env_creds.cerebras_api_key.is_some(),
            "ollama" => true,
            "kilo" => env_creds.kilo_api_key.is_some(),
            "zai" => env_creds.zai_api_key.is_some(),
            "opencode-go" => env_creds.opencodego_api_key.is_some(),
            _ => false,
        };
        status.insert(name.to_string(), config_has || env_has);
    }
    status.insert(
        "exa".to_string(),
        cfg.exa_api_key.is_some() || env_creds.exa_api_key.is_some(),
    );
    Ok(status)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    // ── known_models_for_provider ──

    #[test]
    fn test_openai_models_nonempty() {
        let models = known_models_for_provider("openai");
        assert!(!models.is_empty(), "openai should have known models");
    }

    #[test]
    fn test_anthropic_models_nonempty() {
        let models = known_models_for_provider("anthropic");
        assert!(!models.is_empty(), "anthropic should have known models");
    }

    #[test]
    fn test_openrouter_models_nonempty() {
        let models = known_models_for_provider("openrouter");
        assert!(!models.is_empty(), "openrouter should have known models");
    }

    #[test]
    fn test_cerebras_models_nonempty() {
        let models = known_models_for_provider("cerebras");
        assert!(!models.is_empty(), "cerebras should have known models");
    }

    #[test]
    fn test_ollama_models_nonempty() {
        let models = known_models_for_provider("ollama");
        assert!(!models.is_empty(), "ollama should have known models");
    }

    #[test]
    fn test_kilo_models_nonempty() {
        let models = known_models_for_provider("kilo");
        assert!(!models.is_empty(), "kilo should have known models");
    }

    #[test]
    fn test_zai_models_nonempty() {
        let models = known_models_for_provider("zai");
        assert!(!models.is_empty(), "zai should have known models");
    }

    #[test]
    fn test_opencodego_models_nonempty() {
        let models = known_models_for_provider("opencode-go");
        assert!(!models.is_empty(), "opencode-go should have known models");
    }

    #[test]
    fn test_unknown_provider_empty() {
        let models = known_models_for_provider("foo");
        assert!(models.is_empty(), "unknown provider should return empty vec");
    }

    #[test]
    fn test_all_providers_model_ids_unique() {
        let mut all_ids = HashSet::new();
        for p in all_model_providers() {
            for m in known_models_for_provider(p) {
                assert!(
                    all_ids.insert(m.id.clone()),
                    "duplicate model ID: {}",
                    m.id
                );
            }
        }
    }

    #[test]
    fn test_model_info_fields() {
        for provider in all_model_providers() {
            for m in known_models_for_provider(provider) {
                assert!(!m.id.is_empty(), "model id should not be empty");
                assert!(m.name.is_some(), "model name should be Some for {}", m.id);
                assert_eq!(m.provider, *provider, "provider mismatch for {}", m.id);
            }
        }
    }

    // ── build_credential_status ──

    #[test]
    fn test_cred_status_all_none() {
        let cfg = op_core::config::AgentConfig::from_env("/nonexistent");
        let mut cfg = cfg;
        cfg.openai_api_key = None;
        cfg.anthropic_api_key = None;
        cfg.openrouter_api_key = None;
        cfg.cerebras_api_key = None;
        cfg.kilo_api_key = None;
        cfg.zai_api_key = None;
        cfg.opencodego_api_key = None;
        let status = build_credential_status(&cfg);
        assert_eq!(status["openai"], false);
        assert_eq!(status["anthropic"], false);
        assert_eq!(status["openrouter"], false);
        assert_eq!(status["cerebras"], false);
        assert_eq!(status["ollama"], true, "ollama always true");
    }

    #[test]
    fn test_cred_status_openai_set() {
        let mut cfg = op_core::config::AgentConfig::from_env("/nonexistent");
        cfg.openai_api_key = Some("sk-test".to_string());
        cfg.anthropic_api_key = None;
        cfg.openrouter_api_key = None;
        cfg.cerebras_api_key = None;
        let status = build_credential_status(&cfg);
        assert_eq!(status["openai"], true);
        assert_eq!(status["anthropic"], false);
    }

    #[test]
    fn test_cred_status_anthropic_set() {
        let mut cfg = op_core::config::AgentConfig::from_env("/nonexistent");
        cfg.openai_api_key = None;
        cfg.anthropic_api_key = Some("sk-ant-test".to_string());
        cfg.openrouter_api_key = None;
        cfg.cerebras_api_key = None;
        let status = build_credential_status(&cfg);
        assert_eq!(status["anthropic"], true);
        assert_eq!(status["openai"], false);
    }

    #[test]
    fn test_cred_status_ollama_always_true() {
        let mut cfg = op_core::config::AgentConfig::from_env("/nonexistent");
        cfg.openai_api_key = None;
        cfg.anthropic_api_key = None;
        cfg.openrouter_api_key = None;
        cfg.cerebras_api_key = None;
        let status = build_credential_status(&cfg);
        assert_eq!(status["ollama"], true);
    }

    #[test]
    fn test_cred_status_all_set() {
        let mut cfg = op_core::config::AgentConfig::from_env("/nonexistent");
        cfg.openai_api_key = Some("k1".to_string());
        cfg.anthropic_api_key = Some("k2".to_string());
        cfg.openrouter_api_key = Some("k3".to_string());
        cfg.cerebras_api_key = Some("k4".to_string());
        cfg.kilo_api_key = Some("k5".to_string());
        cfg.zai_api_key = Some("k6".to_string());
        cfg.opencodego_api_key = Some("k7".to_string());
        cfg.exa_api_key = Some("k8".to_string());
        let status = build_credential_status(&cfg);
        for (provider, has_key) in &status {
            assert!(has_key, "{} should be true when key is set", provider);
        }
    }

    #[test]
    fn test_cred_status_has_nine_entries() {
        let cfg = op_core::config::AgentConfig::from_env("/nonexistent");
        let status = build_credential_status(&cfg);
        assert_eq!(status.len(), 9, "should have 9 entries (8 model providers + exa)");
    }
}
