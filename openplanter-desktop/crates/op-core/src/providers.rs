use std::collections::HashMap;
use std::sync::LazyLock;

/// A model entry for a provider's known models list.
pub struct ModelEntry {
    pub id: &'static str,
    pub display_name: &'static str,
}

/// Central provider definition — single source of truth for provider metadata.
///
/// Every module (builder, config, credentials, settings, tauri commands) consumes
/// this registry instead of maintaining its own hardcoded provider lists.
pub struct Provider {
    /// Internal key, e.g. "openai", "opencode-go".
    pub name: &'static str,
    /// Human-readable display name.
    pub display_name: &'static str,
    /// Default base URL for the API endpoint.
    pub default_base_url: &'static str,
    /// Environment variable names for the API key (checked in order).
    pub api_key_env: &'static [&'static str],
    /// Default model identifier.
    pub default_model: &'static str,
    /// Regex pattern to infer this provider from a model name (case-insensitive).
    pub infer_pattern: Option<&'static str>,
    /// Priority for inference ordering (lower = checked first).
    pub infer_priority: u8,
    /// Whether this provider accepts any model name (like openrouter).
    pub accepts_any_model: bool,
    /// Known models with display names.
    pub known_models: &'static [ModelEntry],
}

/// The canonical provider registry — ordered by inference priority.
pub static PROVIDERS: LazyLock<HashMap<&'static str, &'static Provider>> =
    LazyLock::new(|| {
        let mut m = HashMap::new();
        for p in ALL_PROVIDERS.iter() {
            m.insert(p.name, *p);
        }
        m
    });

/// All providers in inference-priority order.
pub static ALL_PROVIDERS: [&Provider; 9] = [
    &OPENCODE_GO,  // prefix with slash, must beat openrouter
    &KILO,         // prefix models, must beat openrouter
    &ZAI,          // prefix models
    &OPENROUTER,   // slash-containing models
    &ANTHROPIC,    // claude prefix
    &CEREBRAS,     // specific prefixes (qwen-3, etc.)
    &OPENAI,       // gpt/o-series prefix
    &OLLAMA,       // broad local model prefix (checked last)
    &EXA,          // exa search provider
];

pub static OPENAI: Provider = Provider {
    name: "openai",
    display_name: "OpenAI",
    default_base_url: "https://api.openai.com/v1",
    api_key_env: &["OPENPLANTER_OPENAI_API_KEY", "OPENAI_API_KEY"],
    default_model: "gpt-5.2",
    infer_pattern: Some(r"(?i)^(gpt|o[1-4]-|o[1-4]$|chatgpt|dall-e|tts-|whisper)"),
    infer_priority: 60,
    accepts_any_model: false,
    known_models: &[
        ModelEntry { id: "gpt-5.2", display_name: "GPT-5.2" },
        ModelEntry { id: "gpt-4o", display_name: "GPT-4o" },
        ModelEntry { id: "gpt-4o-mini", display_name: "GPT-4o Mini" },
        ModelEntry { id: "o1", display_name: "o1" },
        ModelEntry { id: "o3", display_name: "o3" },
        ModelEntry { id: "o4-mini", display_name: "o4-mini" },
    ],
};

pub static ANTHROPIC: Provider = Provider {
    name: "anthropic",
    display_name: "Anthropic",
    default_base_url: "https://api.anthropic.com/v1",
    api_key_env: &["OPENPLANTER_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY"],
    default_model: "claude-opus-4-6",
    infer_pattern: Some(r"(?i)^claude"),
    infer_priority: 40,
    accepts_any_model: false,
    known_models: &[
        ModelEntry { id: "claude-opus-4-6", display_name: "Claude Opus 4.6" },
        ModelEntry { id: "claude-sonnet-4-5", display_name: "Claude Sonnet 4.5" },
        ModelEntry { id: "claude-haiku-4-5", display_name: "Claude Haiku 4.5" },
    ],
};

pub static OPENROUTER: Provider = Provider {
    name: "openrouter",
    display_name: "OpenRouter",
    default_base_url: "https://openrouter.ai/api/v1",
    api_key_env: &["OPENPLANTER_OPENROUTER_API_KEY", "OPENROUTER_API_KEY"],
    default_model: "anthropic/claude-sonnet-4-5",
    infer_pattern: None, // handled by contains('/') check after prefix providers
    infer_priority: 30,
    accepts_any_model: true,
    known_models: &[
        ModelEntry { id: "anthropic/claude-sonnet-4-5", display_name: "Claude Sonnet 4.5 (OR)" },
        ModelEntry { id: "anthropic/claude-opus-4-6", display_name: "Claude Opus 4.6 (OR)" },
        ModelEntry { id: "openai/gpt-5.2", display_name: "GPT-5.2 (OR)" },
    ],
};

pub static CEREBRAS: Provider = Provider {
    name: "cerebras",
    display_name: "Cerebras",
    default_base_url: "https://api.cerebras.ai/v1",
    api_key_env: &["OPENPLANTER_CEREBRAS_API_KEY", "CEREBRAS_API_KEY"],
    default_model: "qwen-3-235b-a22b-instruct-2507",
    infer_pattern: Some(r"(?i)^(llama.*cerebras|qwen-3|gpt-oss|zai-glm)"),
    infer_priority: 50,
    accepts_any_model: false,
    known_models: &[
        ModelEntry { id: "qwen-3-235b-a22b-instruct-2507", display_name: "Qwen-3 235B" },
        ModelEntry { id: "llama-4-scout-17b-16e-instruct", display_name: "Llama-4 Scout" },
    ],
};

pub static OLLAMA: Provider = Provider {
    name: "ollama",
    display_name: "Ollama",
    default_base_url: "http://localhost:11434/v1",
    api_key_env: &[], // no key needed
    default_model: "llama3.2",
    infer_pattern: Some(r"(?i)^(llama|mistral|gemma|phi|codellama|deepseek|vicuna|tinyllama|neural-chat|dolphin|wizardlm|orca|nous-hermes|command-r|qwen)"),
    infer_priority: 70,
    accepts_any_model: false,
    known_models: &[
        ModelEntry { id: "llama3.2", display_name: "Llama 3.2" },
        ModelEntry { id: "mistral", display_name: "Mistral" },
        ModelEntry { id: "gemma", display_name: "Gemma" },
        ModelEntry { id: "phi", display_name: "Phi" },
        ModelEntry { id: "deepseek", display_name: "DeepSeek" },
        ModelEntry { id: "qwen2", display_name: "Qwen 2" },
    ],
};

pub static KILO: Provider = Provider {
    name: "kilo",
    display_name: "Kilo AI Gateway",
    default_base_url: "https://api.kilo.ai/api/gateway",
    api_key_env: &["OPENPLANTER_KILO_API_KEY", "KILO_API_KEY"],
    default_model: "anthropic/claude-sonnet-4-5",
    infer_pattern: Some(r"(?i)^kilo-"),
    infer_priority: 10,
    accepts_any_model: false,
    known_models: &[
        ModelEntry { id: "kilo-auto/frontier", display_name: "Kilo Auto (Frontier)" },
        ModelEntry { id: "kilo-auto/balanced", display_name: "Kilo Auto (Balanced)" },
        ModelEntry { id: "kilo-auto/fast", display_name: "Kilo Auto (Fast)" },
        ModelEntry { id: "anthropic/claude-sonnet-4-5", display_name: "Claude Sonnet 4.5 (Kilo)" },
        ModelEntry { id: "anthropic/claude-opus-4-6", display_name: "Claude Opus 4.6 (Kilo)" },
        ModelEntry { id: "openai/gpt-5.2", display_name: "GPT-5.2 (Kilo)" },
    ],
};

pub static ZAI: Provider = Provider {
    name: "zai",
    display_name: "Z.ai (GLM)",
    default_base_url: "https://api.z.ai/api/coding/paas/v4",
    api_key_env: &["OPENPLANTER_ZAI_API_KEY", "ZAI_API_KEY"],
    default_model: "glm-5",
    infer_pattern: Some(r"(?i)^glm-"),
    infer_priority: 20,
    accepts_any_model: false,
    known_models: &[
        ModelEntry { id: "glm-5", display_name: "GLM-5" },
        ModelEntry { id: "glm-4.7", display_name: "GLM-4.7" },
        ModelEntry { id: "glm-4.6", display_name: "GLM-4.6" },
        ModelEntry { id: "glm-4.5", display_name: "GLM-4.5" },
        ModelEntry { id: "glm-4.5-air", display_name: "GLM-4.5 Air" },
    ],
};

pub static OPENCODE_GO: Provider = Provider {
    name: "opencode-go",
    display_name: "OpenCode Go",
    default_base_url: "https://opencode.ai/zen/go/v1",
    api_key_env: &["OPENPLANTER_OPENCODEGO_API_KEY", "OPENCODEGO_API_KEY"],
    default_model: "opencode-go/glm-5",
    infer_pattern: Some(r"(?i)^opencode-go/"),
    infer_priority: 5,
    accepts_any_model: false,
    known_models: &[
        ModelEntry { id: "opencode-go/glm-5", display_name: "GLM-5 (OpenCode Go)" },
        ModelEntry { id: "opencode-go/kimi-k2.5", display_name: "Kimi K2.5 (OpenCode Go)" },
        ModelEntry { id: "opencode-go/minimax-m2.5", display_name: "MiniMax M2.5 (OpenCode Go)" },
    ],
};

pub static EXA: Provider = Provider {
    name: "exa",
    display_name: "Exa Search",
    default_base_url: "https://api.exa.ai",
    api_key_env: &["OPENPLANTER_EXA_API_KEY", "EXA_API_KEY"],
    default_model: "",
    infer_pattern: None,
    infer_priority: 99,
    accepts_any_model: false,
    known_models: &[],
};

/// Infer the provider for a model name using the registry patterns.
/// Returns the provider name, or None if ambiguous.
pub fn infer_provider_for_model(model: &str) -> Option<&'static str> {
    // Sort by priority (lowest first)
    let mut ordered: Vec<&&Provider> = ALL_PROVIDERS.iter().collect();
    ordered.sort_by_key(|p| p.infer_priority);

    for provider in ordered {
        // Check prefix patterns first
        if let Some(pattern) = provider.infer_pattern {
            if let Ok(re) = regex::Regex::new(pattern) {
                if re.is_match(model) {
                    return Some(provider.name);
                }
            }
        }
    }

    // OpenRouter catches anything with '/' that wasn't caught by prefix providers
    if model.contains('/') {
        return Some("openrouter");
    }

    None
}

/// Get provider by name from the registry.
pub fn get_provider(name: &str) -> Option<&'static Provider> {
    PROVIDERS.get(name).copied()
}

/// Get all provider names in inference-priority order.
pub fn provider_names() -> Vec<&'static str> {
    let mut ordered: Vec<&&Provider> = ALL_PROVIDERS.iter().collect();
    ordered.sort_by_key(|p| p.infer_priority);
    ordered.iter().map(|p| p.name).collect()
}

/// Get the API key from environment for a provider.
pub fn resolve_api_key_from_env(provider_name: &str) -> Option<String> {
    let provider = get_provider(provider_name)?;
    for env_var in provider.api_key_env {
        if let Ok(val) = std::env::var(env_var) {
            let trimmed = val.trim().to_string();
            if !trimmed.is_empty() {
                return Some(trimmed);
            }
        }
    }
    None
}

/// Get the default base URL from environment (with override support) or fallback.
pub fn resolve_base_url_from_env(provider_name: &str) -> String {
    let provider = get_provider(provider_name);
    let env_key = format!(
        "OPENPLANTER_{}_BASE_URL",
        provider_name.to_uppercase().replace('-', "_")
    );
    if let Ok(url) = std::env::var(&env_key) {
        let trimmed = url.trim().to_string();
        if !trimmed.is_empty() {
            return trimmed;
        }
    }
    provider.map(|p| p.default_base_url.to_string())
        .unwrap_or_default()
}

/// List of provider names to show in "all" model listings (excludes exa).
pub fn all_model_providers() -> &'static [&'static str] {
    &["openai", "anthropic", "openrouter", "cerebras", "ollama", "kilo", "zai", "opencode-go"]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_providers_registered() {
        for p in ALL_PROVIDERS.iter() {
            assert!(
                PROVIDERS.contains_key(p.name),
                "provider '{}' not in PROVIDERS map",
                p.name
            );
        }
    }

    #[test]
    fn test_provider_names_no_duplicates() {
        let mut seen = std::collections::HashSet::new();
        for name in provider_names() {
            assert!(seen.insert(name), "duplicate provider: {}", name);
        }
    }

    #[test]
    fn test_infer_anthropic() {
        assert_eq!(infer_provider_for_model("claude-opus-4-6"), Some("anthropic"));
        assert_eq!(infer_provider_for_model("claude-sonnet-4-5"), Some("anthropic"));
        assert_eq!(infer_provider_for_model("Claude-3-Haiku"), Some("anthropic"));
    }

    #[test]
    fn test_infer_openai() {
        assert_eq!(infer_provider_for_model("gpt-5.2"), Some("openai"));
        assert_eq!(infer_provider_for_model("o1-preview"), Some("openai"));
        assert_eq!(infer_provider_for_model("o3"), Some("openai"));
        assert_eq!(infer_provider_for_model("chatgpt-4o"), Some("openai"));
    }

    #[test]
    fn test_infer_cerebras() {
        assert_eq!(infer_provider_for_model("qwen-3-235b"), Some("cerebras"));
        assert_eq!(infer_provider_for_model("gpt-oss-120b"), Some("cerebras"));
        assert_eq!(infer_provider_for_model("llama-4-scout-cerebras"), Some("cerebras"));
    }

    #[test]
    fn test_infer_ollama() {
        assert_eq!(infer_provider_for_model("llama3.2"), Some("ollama"));
        assert_eq!(infer_provider_for_model("mistral"), Some("ollama"));
        assert_eq!(infer_provider_for_model("phi3"), Some("ollama"));
        assert_eq!(infer_provider_for_model("deepseek-v2"), Some("ollama"));
    }

    #[test]
    fn test_cerebras_before_ollama() {
        assert_eq!(infer_provider_for_model("qwen-3-235b"), Some("cerebras"));
        assert_eq!(infer_provider_for_model("qwen2"), Some("ollama"));
    }

    #[test]
    fn test_infer_openrouter() {
        assert_eq!(infer_provider_for_model("anthropic/claude-sonnet-4-5"), Some("openrouter"));
        assert_eq!(infer_provider_for_model("openai/gpt-5.2"), Some("openrouter"));
    }

    #[test]
    fn test_infer_kilo() {
        assert_eq!(infer_provider_for_model("kilo-auto/frontier"), Some("kilo"));
        assert_eq!(infer_provider_for_model("kilo-auto/balanced"), Some("kilo"));
        assert_eq!(infer_provider_for_model("kilo-auto/fast"), Some("kilo"));
    }

    #[test]
    fn test_infer_kilo_not_openrouter() {
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
        assert_eq!(infer_provider_for_model("opencode-go/minimax-m2.5"), Some("opencode-go"));
    }

    #[test]
    fn test_infer_opencode_go_not_openrouter() {
        assert_eq!(infer_provider_for_model("opencode-go/glm-5"), Some("opencode-go"));
    }

    #[test]
    fn test_infer_unknown() {
        assert_eq!(infer_provider_for_model("my-custom-model"), None);
        assert_eq!(infer_provider_for_model("some-random-model"), None);
    }

    #[test]
    fn test_get_provider() {
        let p = get_provider("openai").unwrap();
        assert_eq!(p.display_name, "OpenAI");
        assert_eq!(p.default_base_url, "https://api.openai.com/v1");

        let p = get_provider("zai").unwrap();
        assert_eq!(p.display_name, "Z.ai (GLM)");

        let p = get_provider("opencode-go").unwrap();
        assert_eq!(p.display_name, "OpenCode Go");

        assert!(get_provider("nonexistent").is_none());
    }

    #[test]
    fn test_provider_names_complete() {
        let names = provider_names();
        assert!(names.contains(&"openai"));
        assert!(names.contains(&"anthropic"));
        assert!(names.contains(&"kilo"));
        assert!(names.contains(&"zai"));
        assert!(names.contains(&"opencode-go"));
        assert!(names.contains(&"openrouter"));
        assert!(names.contains(&"cerebras"));
        assert!(names.contains(&"ollama"));
    }

    #[test]
    fn test_all_model_providers_includes_new() {
        let providers = all_model_providers();
        assert!(providers.contains(&"kilo"));
        assert!(providers.contains(&"zai"));
        assert!(providers.contains(&"opencode-go"));
        assert!(!providers.contains(&"exa"));
    }

    #[test]
    fn test_default_models_valid() {
        for p in ALL_PROVIDERS.iter() {
            if p.default_model.is_empty() {
                continue; // exa has no models
            }
            // Default model should either infer to the provider itself, or be a slash model (routed)
            let inferred = infer_provider_for_model(p.default_model);
            match inferred {
                Some(provider_name) if provider_name == p.name => {} // perfect match
                Some("openrouter") => {} // routed through openrouter (e.g., kilo's default)
                Some(other) => {
                    // zai/opencode-go models that route through their own provider
                    assert!(
                        other == p.name || p.accepts_any_model,
                        "{} default '{}' inferred as '{}'",
                        p.name, p.default_model, other
                    );
                }
                None => {} // unknown pattern, acceptable
            }
        }
    }
}
