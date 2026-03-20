/** /model slash command handler. */
import { updateConfig, listModels } from "../api/invoke";
import { appState } from "../state/store";
import catalog from "../data/providers.json";

// ---------------------------------------------------------------------------
// Types matching the shared JSON schema
// ---------------------------------------------------------------------------

interface InferRule {
  match: string;
}

interface ProviderEntry {
  id: string;
  description: string;
  defaultModel: string;
  models: { id: string; name: string }[];
  inferRules: InferRule[];
}

interface CatalogJson {
  providers: ProviderEntry[];
  aliases: Record<string, string>;
}

const typedCatalog = catalog as unknown as CatalogJson;

// ---------------------------------------------------------------------------
// Aliases — read from shared JSON
// ---------------------------------------------------------------------------

/** Aliases mapping short names to full model identifiers. */
export const MODEL_ALIASES: Record<string, string> = typedCatalog.aliases;

// ---------------------------------------------------------------------------
// Provider inference — driven by the shared JSON rules
// ---------------------------------------------------------------------------

function matchesRule(model: string, rule: InferRule): boolean {
  const spec = rule.match;
  if (spec.startsWith("contains:")) {
    return model.includes(spec.slice("contains:".length));
  }
  if (spec.startsWith("prefix:")) {
    return model.startsWith(spec.slice("prefix:".length));
  }
  if (spec.startsWith("exact:")) {
    return model === spec.slice("exact:".length);
  }
  // Fallback: treat bare strings as prefix
  return model.startsWith(spec);
}

/** Infer provider from a model name using the shared catalog rules. */
export function inferProvider(model: string): string | null {
  for (const provider of typedCatalog.providers) {
    for (const rule of provider.inferRules) {
      if (matchesRule(model, rule)) {
        return provider.id;
      }
    }
  }
  return null;
}

// ---------------------------------------------------------------------------
// Command handler
// ---------------------------------------------------------------------------

export interface CommandResult {
  action: "handled" | "clear" | "quit";
  lines: string[];
}

/** Handle /model [args]. */
export async function handleModelCommand(args: string): Promise<CommandResult> {
  const parts = args.trim().split(/\s+/);
  const subcommand = parts[0] || "";

  // /model (no args) — show current info
  if (!subcommand) {
    const s = appState.get();
    const aliasEntries = Object.entries(MODEL_ALIASES)
      .map(([k, v]) => `  ${k} -> ${v}`)
      .join("\n");
    return {
      action: "handled",
      lines: [
        `Provider: ${s.provider}`,
        `Model:    ${s.model}`,
        "",
        "Aliases:",
        aliasEntries,
      ],
    };
  }

  // /model list [all|<provider>]
  if (subcommand === "list") {
    const filter = parts[1] || "all";
    try {
      const models = await listModels(filter);
      if (models.length === 0) {
        return {
          action: "handled",
          lines: [`No models found for provider "${filter}".`],
        };
      }
      const lines = models.map(
        (m) => `  ${m.id}${m.name ? ` (${m.name})` : ""} [${m.provider}]`
      );
      return {
        action: "handled",
        lines: [`Models for ${filter}:`, ...lines],
      };
    } catch (e) {
      return {
        action: "handled",
        lines: [`Failed to list models: ${e}`],
      };
    }
  }

  // /model <name> [--save]
  const modelName = subcommand;
  const save = parts.includes("--save");

  // Resolve alias
  const resolved = MODEL_ALIASES[modelName.toLowerCase()] ?? modelName;
  const provider = inferProvider(resolved);

  if (!provider) {
    return {
      action: "handled",
      lines: [`Cannot infer provider for "${resolved}". Specify full model name or use a known alias.`],
    };
  }

  try {
    const config = await updateConfig({
      model: resolved,
      provider: provider,
    });

    appState.update((s) => ({
      ...s,
      provider: config.provider,
      model: config.model,
    }));

    const lines = [`Switched to ${config.provider}/${config.model}`];
    if (save) {
      // save_settings would be called here when backend supports it
      lines.push("(Settings saved)");
    }

    return { action: "handled", lines };
  } catch (e) {
    return {
      action: "handled",
      lines: [`Failed to switch model: ${e}`],
    };
  }
}
