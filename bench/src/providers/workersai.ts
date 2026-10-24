// workers-ai.clef — Cloudflare's Clef decision models: the System One
// wire (engines/jev_client.py) behind the REST v4 envelope. POST
// {base}/accounts/<account>/ai/run/<endpoint> with Bearer
// CLOUDFLARE_AUTH_TOKEN; the account id rides ctx.auth.CLOUDFLARE_ACCOUNT_ID
// (engines/clef_client.py resolves it from the env / `cf auth whoami` —
// the bench resolves auth, the adapter only reads it). sendSystemone
// unwraps the {success, result} envelope and maps the Jev-compatible
// answer rows.
import type { TrackRequestOut, TrackResponseOut } from "@zsib/schema";
import { sendSystemone } from "./systemone.ts";
import { type Adapter, type AdapterContext, AdapterError } from "./types.ts";

export const clef: Adapter = {
  id: "workers-ai.clef",
  async send(req: TrackRequestOut, ctx: AdapterContext): Promise<TrackResponseOut> {
    const account = ctx.auth.CLOUDFLARE_ACCOUNT_ID;
    if (!account) {
      throw new AdapterError("workers-ai.clef needs CLOUDFLARE_ACCOUNT_ID in ctx.auth", null, null);
    }
    return sendSystemone(req, ctx, {
      url: `${ctx.baseUrl}/accounts/${account}/ai/run/${ctx.endpoint}`,
      bearer: ctx.auth.CLOUDFLARE_AUTH_TOKEN,
      provider: "workers-ai",
    });
  },
};
