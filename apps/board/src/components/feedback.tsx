import { useRouterState } from "@tanstack/react-router";
import { type FormEvent, useCallback, useEffect, useId, useMemo, useRef, useState } from "react";
import {
  type CommentItem,
  CommentsApiError,
  createComment,
  DEFAULT_TURNSTILE_SITEKEY,
  deleteComment,
  isLocalCommentsApi,
  listComments,
  resolveCommentsApiBase,
  threadForPathname,
} from "../lib/comments";

const OWN_COMMENTS_KEY = "zsib:own-comments";
const TURNSTILE_SCRIPT_SRC =
  "https://challenges.cloudflare.com/turnstile/v0/api.js?render=explicit";

interface TurnstileApi {
  render: (
    container: HTMLElement,
    options: {
      sitekey: string;
      theme?: "light" | "dark" | "auto";
      callback?: (token: string) => void;
      "expired-callback"?: () => void;
      "error-callback"?: () => void;
    },
  ) => string;
  reset: (widgetId?: string) => void;
  remove: (widgetId: string) => void;
}

declare global {
  interface Window {
    turnstile?: TurnstileApi;
  }
}

let turnstileScriptPromise: Promise<TurnstileApi | null> | null = null;

function loadTurnstile(): Promise<TurnstileApi | null> {
  if (typeof window === "undefined") return Promise.resolve(null);
  if (window.turnstile) return Promise.resolve(window.turnstile);
  if (turnstileScriptPromise) return turnstileScriptPromise;

  turnstileScriptPromise = new Promise((resolve) => {
    const existing = document.querySelector<HTMLScriptElement>(
      `script[src="${TURNSTILE_SCRIPT_SRC}"]`,
    );
    if (existing) {
      existing.addEventListener("load", () => resolve(window.turnstile ?? null), { once: true });
      existing.addEventListener("error", () => resolve(null), { once: true });
      return;
    }
    const script = document.createElement("script");
    script.src = TURNSTILE_SCRIPT_SRC;
    script.async = true;
    script.defer = true;
    script.onload = () => resolve(window.turnstile ?? null);
    script.onerror = () => resolve(null);
    document.head.appendChild(script);
  });

  return turnstileScriptPromise;
}

function readOwnIds(): Set<string> {
  try {
    const raw = localStorage.getItem(OWN_COMMENTS_KEY);
    if (!raw) return new Set();
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? new Set(parsed.filter((v) => typeof v === "string")) : new Set();
  } catch {
    return new Set();
  }
}

function writeOwnIds(ids: Set<string>): void {
  try {
    localStorage.setItem(OWN_COMMENTS_KEY, JSON.stringify([...ids]));
  } catch {
    // Ignore storage quota or private-browsing failures.
  }
}

function fmtTimestamp(epochMs: number): string {
  const date = new Date(epochMs);
  if (Number.isNaN(date.getTime())) return "";
  return date.toISOString().slice(0, 16).replace("T", " ");
}

export function FeedbackSection() {
  const pathname = useRouterState({ select: (s) => s.location.pathname });
  const thread = useMemo(() => threadForPathname(pathname), [pathname]);

  const authorInputId = useId();
  const bodyInputId = useId();

  const [mounted, setMounted] = useState(false);
  const [apiBase, setApiBase] = useState<string>("");
  const [comments, setComments] = useState<CommentItem[]>([]);
  const [nextCursor, setNextCursor] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);

  const [author, setAuthor] = useState("");
  const [body, setBody] = useState("");
  const [replyTo, setReplyTo] = useState<CommentItem | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [submitNotice, setSubmitNotice] = useState<string | null>(null);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [ownIds, setOwnIds] = useState<Set<string>>(() => new Set());

  const [turnstileToken, setTurnstileToken] = useState<string | null>(null);
  const [turnstileFailed, setTurnstileFailed] = useState(false);
  const turnstileContainerRef = useRef<HTMLDivElement | null>(null);
  const turnstileWidgetIdRef = useRef<string | null>(null);

  const envSitekey = import.meta.env.VITE_TURNSTILE_SITEKEY as string | undefined;
  const sitekey = envSitekey !== undefined ? envSitekey.trim() : DEFAULT_TURNSTILE_SITEKEY;
  const needsTurnstile = mounted && sitekey.length > 0 && !isLocalCommentsApi(apiBase);

  // Resolve client-only state after hydration so SSR and initial client markup match.
  useEffect(() => {
    const envApi = import.meta.env.VITE_COMMENTS_API as string | undefined;
    setApiBase(resolveCommentsApiBase(window.location.hostname, envApi));
    setOwnIds(readOwnIds());
    setMounted(true);
  }, []);

  // Fetch comments whenever the route thread or API base changes.
  useEffect(() => {
    if (!mounted || !apiBase) return;
    const controller = new AbortController();
    setLoading(true);
    setLoadError(null);
    setReplyTo(null);
    setSubmitError(null);
    setSubmitNotice(null);

    listComments(apiBase, thread, { signal: controller.signal })
      .then((res) => {
        setComments(res.comments);
        setNextCursor(res.nextCursor);
      })
      .catch((err: unknown) => {
        if (controller.signal.aborted) return;
        setComments([]);
        setNextCursor(null);
        setLoadError(err instanceof Error ? err.message : "Could not load feedback.");
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });

    return () => controller.abort();
  }, [mounted, apiBase, thread]);

  // Mount Cloudflare Turnstile once on the client when required.
  useEffect(() => {
    if (!needsTurnstile) return;
    let cancelled = false;

    loadTurnstile().then((api) => {
      if (cancelled) return;
      const container = turnstileContainerRef.current;
      if (!api || !container) {
        setTurnstileFailed(true);
        return;
      }
      if (turnstileWidgetIdRef.current) return;
      turnstileWidgetIdRef.current = api.render(container, {
        sitekey,
        theme: "light",
        callback: (token) => {
          setTurnstileToken(token);
          setTurnstileFailed(false);
        },
        "expired-callback": () => setTurnstileToken(null),
        "error-callback": () => {
          setTurnstileToken(null);
          setTurnstileFailed(true);
        },
      });
    });

    return () => {
      cancelled = true;
      if (turnstileWidgetIdRef.current && window.turnstile) {
        window.turnstile.remove(turnstileWidgetIdRef.current);
        turnstileWidgetIdRef.current = null;
      }
    };
  }, [needsTurnstile, sitekey]);

  const resetTurnstile = useCallback(() => {
    setTurnstileToken(null);
    if (turnstileWidgetIdRef.current && window.turnstile) {
      window.turnstile.reset(turnstileWidgetIdRef.current);
    }
  }, []);

  async function handleLoadMore() {
    if (!apiBase || !nextCursor || loadingMore) return;
    setLoadingMore(true);
    try {
      const res = await listComments(apiBase, thread, { cursor: nextCursor });
      setComments((prev) => [...prev, ...res.comments]);
      setNextCursor(res.nextCursor);
    } catch (err: unknown) {
      setLoadError(err instanceof Error ? err.message : "Could not load more comments.");
    } finally {
      setLoadingMore(false);
    }
  }

  async function handleSubmit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    if (!apiBase || submitting) return;

    const trimmedBody = body.trim();
    if (!trimmedBody) {
      setSubmitError("Write a message before posting.");
      return;
    }
    if (needsTurnstile && !turnstileToken) {
      setSubmitError("Complete the verification check before posting.");
      return;
    }

    setSubmitting(true);
    setSubmitError(null);
    setSubmitNotice(null);

    try {
      const created = await createComment(apiBase, thread, {
        author,
        body: trimmedBody,
        parentId: replyTo?.id ?? null,
        turnstileToken: needsTurnstile ? turnstileToken : null,
      });
      setComments((prev) => [...prev, created]);
      setOwnIds((prev) => {
        const next = new Set(prev);
        next.add(created.id);
        writeOwnIds(next);
        return next;
      });
      setBody("");
      setReplyTo(null);
      setSubmitNotice("Feedback posted.");
    } catch (err: unknown) {
      if (err instanceof CommentsApiError) {
        setSubmitError(err.message);
      } else {
        setSubmitError(err instanceof Error ? err.message : "Could not post feedback.");
      }
    } finally {
      setSubmitting(false);
      if (needsTurnstile) resetTurnstile();
    }
  }

  async function handleDelete(commentId: string) {
    if (!apiBase || deletingId) return;
    setDeletingId(commentId);
    setSubmitError(null);
    try {
      await deleteComment(apiBase, thread, commentId);
      setComments((prev) => prev.filter((c) => c.id !== commentId));
      setOwnIds((prev) => {
        const next = new Set(prev);
        next.delete(commentId);
        writeOwnIds(next);
        return next;
      });
    } catch (err: unknown) {
      setSubmitError(err instanceof Error ? err.message : "Could not delete comment.");
    } finally {
      setDeletingId(null);
    }
  }

  const { roots, repliesByParent } = useMemo(() => {
    const byId = new Set(comments.map((c) => c.id));
    const rootList: CommentItem[] = [];
    const replyMap = new Map<string, CommentItem[]>();

    for (const c of comments) {
      if (c.parentId && byId.has(c.parentId)) {
        const list = replyMap.get(c.parentId) ?? [];
        list.push(c);
        replyMap.set(c.parentId, list);
      } else {
        rootList.push(c);
      }
    }
    return { roots: rootList, repliesByParent: replyMap };
  }, [comments]);

  function renderComment(c: CommentItem, isReply: boolean) {
    const canDelete = ownIds.has(c.id);
    const replies = repliesByParent.get(c.id) ?? [];

    return (
      <li key={c.id} className={isReply ? "feedback-item feedback-reply" : "feedback-item"}>
        <div className="feedback-meta">
          <strong>{c.author}</strong>
          <span className="muted mono">{fmtTimestamp(c.createdAt)} UTC</span>
          <div className="feedback-item-actions">
            {!isReply && (
              <button
                type="button"
                className="linkish"
                onClick={() => {
                  setReplyTo(c);
                  setSubmitError(null);
                }}
              >
                Reply
              </button>
            )}
            {canDelete && (
              <button
                type="button"
                className="linkish feedback-delete"
                disabled={deletingId === c.id}
                onClick={() => handleDelete(c.id)}
              >
                {deletingId === c.id ? "Deleting…" : "Delete"}
              </button>
            )}
          </div>
        </div>
        <p className="feedback-body">{c.body}</p>
        {replies.length > 0 && (
          <ul className="feedback-list feedback-replies">
            {replies.map((reply) => renderComment(reply, true))}
          </ul>
        )}
      </li>
    );
  }

  return (
    <section className="card feedback" aria-label="Feedback and discussion">
      <div className="feedback-head">
        <h2>
          Feedback{" "}
          {mounted && !loading && (
            <span className="type-count" title={`${comments.length} comments`}>
              {comments.length}
            </span>
          )}
        </h2>
        <span className="muted mono feedback-thread">/{thread}</span>
      </div>
      <p className="muted feedback-intro">
        Spot a bug, missing model, or question about this page? Leave a note — no account required.
      </p>

      {!mounted || loading ? (
        <p className="muted feedback-status">Loading feedback…</p>
      ) : loadError ? (
        <p className="feedback-error" role="alert">
          {loadError}
        </p>
      ) : roots.length === 0 ? (
        <p className="muted feedback-status">No feedback on this page yet.</p>
      ) : (
        <>
          <ul className="feedback-list">{roots.map((c) => renderComment(c, false))}</ul>
          {nextCursor && (
            <button
              type="button"
              className="btn feedback-more"
              disabled={loadingMore}
              onClick={handleLoadMore}
            >
              {loadingMore ? "Loading…" : "Load more"}
            </button>
          )}
        </>
      )}

      <form className="feedback-form" onSubmit={handleSubmit}>
        {replyTo && (
          <div className="feedback-replying">
            <span>
              Replying to <strong>{replyTo.author}</strong>
            </span>
            <button type="button" className="linkish" onClick={() => setReplyTo(null)}>
              Cancel
            </button>
          </div>
        )}

        <div className="feedback-field">
          <label htmlFor={authorInputId} className="muted">
            Name (optional)
          </label>
          <input
            id={authorInputId}
            type="text"
            maxLength={80}
            value={author}
            onChange={(e) => setAuthor(e.target.value)}
            placeholder="Anonymous"
            autoComplete="nickname"
          />
        </div>

        <div className="feedback-field">
          <label htmlFor={bodyInputId} className="muted">
            Message
          </label>
          <textarea
            id={bodyInputId}
            rows={3}
            maxLength={10000}
            required
            value={body}
            onChange={(e) => setBody(e.target.value)}
            placeholder="Share feedback, corrections, or model requests…"
          />
        </div>

        {needsTurnstile && <div ref={turnstileContainerRef} className="feedback-turnstile" />}
        {turnstileFailed && (
          <p className="feedback-error" role="alert">
            Verification widget could not load (blocked or unsupported domain).
          </p>
        )}

        {submitError && (
          <p className="feedback-error" role="alert">
            {submitError}
          </p>
        )}
        {submitNotice && (
          <p className="feedback-notice" role="status">
            {submitNotice}
          </p>
        )}

        <div className="feedback-actions">
          <button
            type="submit"
            className="btn btn-primary"
            disabled={submitting || !mounted || (needsTurnstile && !turnstileToken)}
          >
            {submitting ? "Posting…" : replyTo ? "Post reply" : "Post feedback"}
          </button>
          <span className="muted feedback-powered">
            Powered by{" "}
            <a href="https://github.com/umstek/edge-comments" target="_blank" rel="noreferrer">
              edge-comments
            </a>
          </span>
        </div>
      </form>
    </section>
  );
}
