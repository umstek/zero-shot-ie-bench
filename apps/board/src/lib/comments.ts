export const COMMENTS_SITE = "zsib";
export const PROD_COMMENTS_API = "https://comments.umstek.com";
export const STAGING_COMMENTS_API = "https://comments.staging.umstek.com";
export const DEFAULT_TURNSTILE_SITEKEY = "0x4AAAAAAGS31I97vg2qVhDn";

export interface CommentItem {
  id: string;
  parentId: string | null;
  author: string;
  body: string;
  createdAt: number;
  updatedAt: number;
}

export interface ListCommentsResponse {
  schemaVersion: number;
  site: string;
  thread: string;
  comments: CommentItem[];
  nextCursor: string | null;
}

interface ApiErrorEnvelope {
  error?: {
    code?: string;
    message?: string;
  };
}

export class CommentsApiError extends Error {
  readonly status: number;
  readonly code: string;

  constructor(status: number, code: string, message: string) {
    super(message);
    this.name = "CommentsApiError";
    this.status = status;
    this.code = code;
  }
}

/**
 * Convert a board pathname into an edge-comments thread path.
 * edge-comments requires every segment to match `[A-Za-z0-9._~-]+` with no `..`
 * or empty segments, and at least one segment.
 */
export function threadForPathname(pathname: string): string {
  const cleaned = pathname
    .split("/")
    .map((seg) => seg.trim().replace(/[^A-Za-z0-9._~-]/g, "-"))
    .filter((seg) => seg.length > 0 && seg !== "." && seg !== "..");
  if (cleaned.length === 0) return "overview";
  return cleaned.join("/");
}

/**
 * Resolve the edge-comments API origin:
 * - Explicit `envOverride` (e.g. `VITE_COMMENTS_API`) wins.
 * - On `zsib.staging.umstek.com`, target the staging comments worker.
 * - Otherwise target production `comments.umstek.com`.
 */
export function resolveCommentsApiBase(hostname?: string, envOverride?: string): string {
  const override = envOverride?.trim();
  if (override) return override.replace(/\/+$/, "");
  if (hostname === "zsib.staging.umstek.com") return STAGING_COMMENTS_API;
  return PROD_COMMENTS_API;
}

/** Local Miniflare dev runs with ALLOW_INSECURE_WRITES=true and no Turnstile. */
export function isLocalCommentsApi(apiBase: string): boolean {
  try {
    const { hostname } = new URL(apiBase);
    return hostname === "127.0.0.1" || hostname === "localhost";
  } catch {
    return false;
  }
}

export function commentsEndpoint(apiBase: string, thread: string): string {
  const base = apiBase.replace(/\/+$/, "");
  const cleanThread = thread.replace(/^\/+|\/+$/g, "");
  return `${base}/${COMMENTS_SITE}/${cleanThread}/comments`;
}

async function parseError(res: Response): Promise<CommentsApiError> {
  try {
    const data = (await res.json()) as ApiErrorEnvelope;
    const code = data.error?.code ?? "http_error";
    const message = data.error?.message ?? `Request failed (${res.status})`;
    return new CommentsApiError(res.status, code, message);
  } catch {
    return new CommentsApiError(res.status, "http_error", `Request failed (${res.status})`);
  }
}

export async function listComments(
  apiBase: string,
  thread: string,
  options?: { cursor?: string | null; limit?: number; signal?: AbortSignal },
): Promise<ListCommentsResponse> {
  const url = new URL(commentsEndpoint(apiBase, thread));
  if (options?.limit) url.searchParams.set("limit", String(options.limit));
  if (options?.cursor) url.searchParams.set("cursor", options.cursor);

  const res = await fetch(url.toString(), {
    method: "GET",
    credentials: "include",
    signal: options?.signal,
  });
  if (!res.ok) throw await parseError(res);
  return (await res.json()) as ListCommentsResponse;
}

export async function createComment(
  apiBase: string,
  thread: string,
  input: {
    author?: string;
    body: string;
    parentId?: string | null;
    turnstileToken?: string | null;
  },
): Promise<CommentItem> {
  const payload: Record<string, unknown> = {
    body: input.body,
    parentId: input.parentId ?? null,
  };
  const trimmedAuthor = input.author?.trim();
  if (trimmedAuthor) payload.author = trimmedAuthor;
  if (input.turnstileToken) payload.turnstileToken = input.turnstileToken;

  const res = await fetch(commentsEndpoint(apiBase, thread), {
    method: "POST",
    credentials: "include",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw await parseError(res);
  return (await res.json()) as CommentItem;
}

export async function deleteComment(apiBase: string, thread: string, id: string): Promise<void> {
  const res = await fetch(`${commentsEndpoint(apiBase, thread)}/${encodeURIComponent(id)}`, {
    method: "DELETE",
    credentials: "include",
  });
  if (!res.ok) throw await parseError(res);
}
