import { describe, expect, test } from "bun:test";
import type { TrackRequestOut } from "@zsib/schema";
import { getAdapter } from "../src/providers/index.ts";
import { systemone } from "../src/providers/openrouter.ts";

const ctx = {
  baseUrl: "https://openrouter.test/api/v1",
  auth: { OPENROUTER_API_KEY: "k" },
  endpoint: "test/decider",
};

const req = {
  model_id: "perplexity/pplx-decider-v1.2",
  items: [
    {
      item_id: "mp-001",
      input: { text: "The food was delicious" },
      question: { type: "choice", options: ["positive", "negative", "neutral"] },
    },
  ],
} as unknown as TrackRequestOut;

describe("openrouter.systemone adapter", () => {
  test("maps items to the systemone wire and answers back", async () => {
    const captured: { c?: { url: string; body: any; headers: Record<string, string> } } = {};
    const fetchImpl = (async (url: string, init: RequestInit) => {
      captured.c = { url, body: JSON.parse(init.body as string), headers: init.headers as never };
      return new Response(
        JSON.stringify({
          answers: { "mp-001": { choice: { pick: "positive", probabilities: [0.9, 0.05, 0.05] } } },
          usage: { total_tokens: 10 },
        }),
        { status: 200 },
      );
    }) as typeof fetch;
    const res = await systemone.send(req, { ...ctx, fetchImpl });
    expect(captured.c?.url).toBe("https://openrouter.test/api/v1/systemone");
    expect(captured.c?.body.state).toBe("The food was delicious");
    expect((captured.c?.body.questions["mp-001"] as any)?.choice?.options).toHaveLength(3);
    expect(captured.c?.headers.authorization).toBe("Bearer k");
    expect(res.results[0]?.prediction).toBe("positive");
    expect(res.results[0]?.prob).toBeCloseTo(0.9);
  });

  test("HTTP failure becomes AdapterError with status", async () => {
    const fetchImpl = (async () =>
      new Response("{}", {
        status: 429,
        headers: { "retry-after": "2" },
      })) as unknown as typeof fetch;
    expect(systemone.send(req, { ...ctx, fetchImpl })).rejects.toThrow(/429/);
  });

  test("registry resolves by adapter id", () => {
    expect(getAdapter("openrouter.systemone").id).toBe("openrouter.systemone");
    expect(() => getAdapter("nope.nothing")).toThrow();
  });
});
