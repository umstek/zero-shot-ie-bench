// registry: catalog deployment.adapter → adapter module. The bench CLI and
// apps/playground import from here (docs/plan/03 rule 5).
import { chat as fastinoChat, systemone as fastinoSystemone } from "./fastino.ts";
import { rerank as openrouterRerank, systemone as openrouterSystemone } from "./openrouter.ts";
import { generate as typellmGenerate } from "./typellm.ts";
import type { Adapter } from "./types.ts";
import { systemone as typesafeSystemone } from "./typesafe.ts";
import { clef as workersaiClef } from "./workersai.ts";

const adapters: Record<string, Adapter> = {
  [openrouterSystemone.id]: openrouterSystemone,
  [openrouterRerank.id]: openrouterRerank,
  [typesafeSystemone.id]: typesafeSystemone,
  [workersaiClef.id]: workersaiClef,
  [fastinoSystemone.id]: fastinoSystemone,
  [fastinoChat.id]: fastinoChat,
  [typellmGenerate.id]: typellmGenerate,
};

export function getAdapter(id: string): Adapter {
  const adapter = adapters[id];
  if (!adapter) throw new Error(`no adapter registered for "${id}"`);
  return adapter;
}

export type { Adapter, AdapterContext } from "./types.ts";
export { AdapterError } from "./types.ts";
