/// Provider catalog loaded from the shared `data/providers.json`.
///
/// Single source of truth for provider metadata, model lists, aliases,
/// and inference rules.  Replaces the scattered hardcoded constants that
/// previously lived in `config.rs`, `builder.rs`, and the Tauri commands.
use std::collections::HashMap;
use std::sync::LazyLock;

use serde::Deserialize;

// ---------------------------------------------------------------------------
// Deserialisation types (mirror the JSON schema)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
struct CatalogJson {
    providers: Vec<ProviderJson>,
    aliases: HashMap<String, String>,
}

#[derive(Debug, Clone, Deserialize)]
struct ProviderJson {
    id: String,
    description: String,
    #[serde(rename = "defaultModel")]
    default_model: String,
    models: Vec<ModelJson>,
    #[serde(rename = "inferRules")]
    infer_rules: Vec<InferRuleJson>,
}

#[derive(Debug, Clone, Deserialize)]
struct ModelJson {
    id: String,
    name: String,
}

#[derive(Debug, Clone, Deserialize)]
struct InferRuleJson {
    #[serde(rename = "match")]
    match_spec: String,
}

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// A single provider entry from the catalog.
#[derive(Debug, Clone)]
pub struct Provider {
    pub id: String,
    pub description: String,
    pub default_model: String,
    pub models: Vec<ModelInfo>,
    pub infer_rules: Vec<InferRule>,
}

/// A model belonging to a provider.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub id: String,
    pub name: String,
}

/// How an inference rule matches a model name.
#[derive(Debug, Clone)]
pub enum MatchType {
    /// Model name contains the pattern anywhere.
    Contains,
    /// Model name starts with the pattern.
    Prefix,
    /// Model name equals the pattern exactly.
    Exact,
}

/// A single inference rule: match type + pattern.
#[derive(Debug, Clone)]
pub struct InferRule {
    pub match_type: MatchType,
    pub pattern: String,
}

/// The full provider catalog.
#[derive(Debug, Clone)]
pub struct ProviderCatalog {
    pub providers: Vec<Provider>,
    pub aliases: HashMap<String, String>,
}

// ---------------------------------------------------------------------------
// JSON → public type conversion
// ---------------------------------------------------------------------------

fn parse_infer_rule(spec: &str) -> InferRule {
    if let Some(rest) = spec.strip_prefix("contains:") {
        InferRule {
            match_type: MatchType::Contains,
            pattern: rest.to_string(),
        }
    } else if let Some(rest) = spec.strip_prefix("prefix:") {
        InferRule {
            match_type: MatchType::Prefix,
            pattern: rest.to_string(),
        }
    } else if let Some(rest) = spec.strip_prefix("exact:") {
        InferRule {
            match_type: MatchType::Exact,
            pattern: rest.to_string(),
        }
    } else {
        // Fallback: treat bare strings as prefix (backward compat)
        InferRule {
            match_type: MatchType::Prefix,
            pattern: spec.to_string(),
        }
    }
}

fn convert_provider(p: ProviderJson) -> Provider {
    Provider {
        id: p.id,
        description: p.description,
        default_model: p.default_model,
        models: p
            .models
            .into_iter()
            .map(|m| ModelInfo {
                id: m.id,
                name: m.name,
            })
            .collect(),
        infer_rules: p
            .infer_rules
            .into_iter()
            .map(|r| parse_infer_rule(&r.match_spec))
            .collect(),
    }
}

// ---------------------------------------------------------------------------
// Lazy-loaded catalog
// ---------------------------------------------------------------------------

static CATALOG: LazyLock<ProviderCatalog> = LazyLock::new(|| {
    let raw = include_str!("../data/providers.json");
    let json: CatalogJson =
        serde_json::from_str(raw).expect("providers.json must be valid JSON");
    ProviderCatalog {
        providers: json.providers.into_iter().map(convert_provider).collect(),
        aliases: json.aliases,
    }
});

/// Return a reference to the lazily-loaded provider catalog.
pub fn catalog() -> &'static ProviderCatalog {
    &CATALOG
}

// ---------------------------------------------------------------------------
// Convenience helpers
// ---------------------------------------------------------------------------

impl ProviderCatalog {
    /// Look up a provider by id.
    pub fn provider(&self, id: &str) -> Option<&Provider> {
        self.providers.iter().find(|p| p.id == id)
    }

    /// Resolve a model alias to its full name, or return the input unchanged.
    pub fn resolve_alias(&self, name: &str) -> String {
        self.aliases
            .get(&name.to_lowercase())
            .cloned()
            .unwrap_or_else(|| name.to_string())
    }

    /// Infer the provider for a model name by checking each provider's
    /// inference rules in order (providers are checked in list order;
    /// rules within a provider are also checked in list order).
    pub fn infer_provider(&self, model: &str) -> Option<&str> {
        for provider in &self.providers {
            for rule in &provider.infer_rules {
                let matched = match rule.match_type {
                    MatchType::Contains => model.contains(&rule.pattern),
                    MatchType::Prefix => model.starts_with(&rule.pattern),
                    MatchType::Exact => model == rule.pattern,
                };
                if matched {
                    return Some(&provider.id);
                }
            }
        }
        None
    }

    /// Get the default model for a provider.
    pub fn default_model(&self, provider_id: &str) -> Option<&str> {
        self.provider(provider_id)
            .map(|p| p.default_model.as_str())
    }

    /// Get all models for a provider.
    pub fn models_for_provider(&self, provider_id: &str) -> &[ModelInfo] {
        self.provider(provider_id)
            .map(|p| p.models.as_slice())
            .unwrap_or(&[])
    }

    /// Get the list of provider ids.
    pub fn provider_ids(&self) -> Vec<&str> {
        self.providers.iter().map(|p| p.id.as_str()).collect()
    }

    /// Get all models across all providers.
    pub fn all_models(&self) -> Vec<&ModelInfo> {
        self.providers.iter().flat_map(|p| &p.models).collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_catalog_loads() {
        let c = catalog();
        assert!(!c.providers.is_empty(), "catalog should have providers");
        assert!(!c.aliases.is_empty(), "catalog should have aliases");
    }

    #[test]
    fn test_provider_lookup() {
        let c = catalog();
        assert!(c.provider("anthropic").is_some());
        assert!(c.provider("openai").is_some());
        assert!(c.provider("ollama").is_some());
        assert!(c.provider("nonexistent").is_none());
    }

    #[test]
    fn test_default_models() {
        let c = catalog();
        assert_eq!(c.default_model("anthropic"), Some("claude-opus-4-6"));
        assert_eq!(c.default_model("openai"), Some("gpt-5.2"));
        assert_eq!(c.default_model("ollama"), Some("llama3.2"));
    }

    #[test]
    fn test_infer_anthropic() {
        let c = catalog();
        assert_eq!(c.infer_provider("claude-opus-4-6"), Some("anthropic"));
        assert_eq!(c.infer_provider("claude-sonnet-4-5"), Some("anthropic"));
        assert_eq!(c.infer_provider("claude-haiku-4-5"), Some("anthropic"));
    }

    #[test]
    fn test_infer_openai() {
        let c = catalog();
        assert_eq!(c.infer_provider("gpt-5.2"), Some("openai"));
        assert_eq!(c.infer_provider("o1-preview"), Some("openai"));
        assert_eq!(c.infer_provider("o3"), Some("openai"));
        assert_eq!(c.infer_provider("chatgpt-4o"), Some("openai"));
    }

    #[test]
    fn test_infer_openrouter() {
        let c = catalog();
        assert_eq!(
            c.infer_provider("anthropic/claude-sonnet-4-5"),
            Some("openrouter")
        );
        assert_eq!(c.infer_provider("openai/gpt-5.2"), Some("openrouter"));
    }

    #[test]
    fn test_infer_cerebras() {
        let c = catalog();
        assert_eq!(
            c.infer_provider("qwen-3-235b-a22b-instruct-2507"),
            Some("cerebras")
        );
        assert_eq!(c.infer_provider("qwen-3"), Some("cerebras"));
    }

    #[test]
    fn test_infer_ollama() {
        let c = catalog();
        assert_eq!(c.infer_provider("llama3.2"), Some("ollama"));
        assert_eq!(c.infer_provider("mistral"), Some("ollama"));
        assert_eq!(c.infer_provider("phi"), Some("ollama"));
        assert_eq!(c.infer_provider("deepseek"), Some("ollama"));
        assert_eq!(c.infer_provider("qwen2"), Some("ollama"));
    }

    #[test]
    fn test_cerebras_before_ollama_qwen() {
        let c = catalog();
        assert_eq!(c.infer_provider("qwen-3"), Some("cerebras"));
        assert_eq!(c.infer_provider("qwen2"), Some("ollama"));
    }

    #[test]
    fn test_infer_unknown() {
        let c = catalog();
        assert_eq!(c.infer_provider("some-random-model"), None);
    }

    #[test]
    fn test_resolve_alias() {
        let c = catalog();
        assert_eq!(c.resolve_alias("opus"), "claude-opus-4-6");
        assert_eq!(c.resolve_alias("gpt5"), "gpt-5.2");
        assert_eq!(c.resolve_alias("unknown-model"), "unknown-model");
    }

    #[test]
    fn test_models_for_provider() {
        let c = catalog();
        let openai = c.models_for_provider("openai");
        assert!(!openai.is_empty());
        assert!(openai.iter().any(|m| m.id == "gpt-5.2"));
    }

    #[test]
    fn test_all_models_unique_ids() {
        let c = catalog();
        let mut ids = std::collections::HashSet::new();
        for m in c.all_models() {
            assert!(ids.insert(&m.id), "duplicate model id: {}", m.id);
        }
    }

    #[test]
    fn test_provider_ids() {
        let c = catalog();
        let ids = c.provider_ids();
        assert!(ids.contains(&"anthropic"));
        assert!(ids.contains(&"openai"));
        assert!(ids.contains(&"openrouter"));
        assert!(ids.contains(&"cerebras"));
        assert!(ids.contains(&"ollama"));
    }
}
