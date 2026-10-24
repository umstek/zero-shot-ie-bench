// registry: catalog deployment.adapter → adapter module. The bench CLI and
// apps/playground import from here (docs/plan/03 rule 5).
import { systemone as openrouterSystemone } from "./openrouter.ts";
import type { Adapter } from "./types.ts";

const adapters: Record<string, Adapter> = {
  [openrouterSystemone.id]: openrouterSystemone,
};

export function getAdapter(id: string): Adapter {
  const adapter = adapters[id];
  if (!adapter) throw new Error(`no adapter registered for "${id}"`);
  return adapter;
}

export type { Adapter, AdapterContext } from "./types.ts";
export { AdapterError } from "./types.ts";
