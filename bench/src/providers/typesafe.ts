// typesafe.systemone — direct System One (Jev). The catalog's provider
// base_url IS the endpoint (https://api.typesafe.ai/v1/systemone), so the
// request goes to ctx.baseUrl verbatim with Bearer TYPESAFE_API_KEY;
// engines/jev_client.py is the wire contract.
//
// Body-model gap (flagged): the deployment pins jev-1.14.0, but its
// endpoint is the full URL, so the pinned api_slug never reaches the
// adapter — the body `model` falls back to the catalog model id
// (typesafe/jev), which TypeSafe's API may not recognize. The deployment
// yaml needs a base_url + slug split (or AdapterContext a model field)
// to restore the jev-latest pin.
import type { TrackRequestOut, TrackResponseOut } from "@zsib/schema";
import { sendSystemone } from "./systemone.ts";
import type { Adapter, AdapterContext } from "./types.ts";

export const systemone: Adapter = {
  id: "typesafe.systemone",
  send: (req: TrackRequestOut, ctx: AdapterContext): Promise<TrackResponseOut> =>
    sendSystemone(req, ctx, {
      url: ctx.baseUrl,
      bearer: ctx.auth.TYPESAFE_API_KEY,
      provider: "typesafe",
    }),
};
