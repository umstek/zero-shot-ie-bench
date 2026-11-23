import {
  createRootRoute,
  createRoute,
  createRouter,
  Link,
  Outlet,
  type RouterHistory,
  useRouterState,
} from "@tanstack/react-router";
import { Suspense, useEffect } from "react";
import { fmtDate } from "./lib/format";
import { useBoardData } from "./queries/data";
import { ComparePage } from "./routes/compare";
import { ExplorerPage } from "./routes/explore";
import { ModelPage } from "./routes/model.$id";
import { NerPage } from "./routes/ner";
import { PicksPage } from "./routes/picks";

// Code-based route tree (no codegen step): the component files live in
// src/routes/ with the planned names, registered here. Planned-but-unbuilt
// views live in docs/board-roadmap.md — no stub pages on the site.

/** Per-route <title>/description — shared by the client title effect and the
 *  build-time prerenderer so SPA navigation and served HTML always agree. */
export const ROUTE_META: Record<string, { title: string; description: string }> = {
  "/": {
    title: "zsib — decision system picks",
    description:
      "Which decision system is cheapest at ≥90% accuracy, fastest on CPU, best on rare languages — answered from measured runs.",
  },
  "/explore": {
    title: "zsib — deployment explorer",
    description:
      "All benchmarked decision-system deployments with accuracy, p50 latency and measured price per 1k items, faceted in memory.",
  },
  "/compare": {
    title: "zsib — charts: speed, price, accuracy, languages",
    description:
      "Cross-deployment charts: speed vs accuracy, price vs accuracy, cost vs speed with Pareto frontiers, and per-language support.",
  },
  "/ner": {
    title: "zsib — NER: strict micro-F1",
    description:
      "Span extraction on the graded pool: strict micro-F1 per system, ranked, from measured runs.",
  },
};

const NAV = [
  { to: "/", label: "Picks", exact: true },
  { to: "/explore", label: "Explorer", exact: false },
  { to: "/compare", label: "Charts", exact: false },
  { to: "/ner", label: "NER", exact: false },
] as const;

function Nav() {
  return (
    <nav className="nav">
      <span className="brand">
        zsib <span className="muted">board</span>
      </span>
      {NAV.map((item) => (
        <Link
          key={item.to}
          to={item.to}
          activeOptions={{ exact: item.exact }}
          activeProps={{ className: "active" }}
        >
          {item.label}
        </Link>
      ))}
    </nav>
  );
}

/** Keeps the tab title in sync with SPA navigation (model pages refine it
 *  themselves once their data resolves). */
function useRouteTitle() {
  const pathname = useRouterState({ select: (s) => s.location.pathname });
  useEffect(() => {
    const meta = ROUTE_META[pathname];
    document.title =
      meta?.title ??
      (pathname.startsWith("/model/") ? "model — zsib board" : "zsib — decision board");
  }, [pathname]);
}

/** "data as of <date> · N runs" — the freshest run in the lake. */
function DataStamp() {
  const { runs } = useBoardData();
  const latest = runs.reduce((max, r) => Math.max(max, Date.parse(r.started_at) || 0), 0);
  return (
    <>
      Data as of {latest ? fmtDate(new Date(latest).toISOString()) : "—"} · {runs.length} runs.
    </>
  );
}

function RootLayout() {
  useRouteTitle();
  return (
    <div className="shell">
      <Nav />
      <main>
        <Suspense fallback={<p className="loading">Loading lake data…</p>}>
          <Outlet />
        </Suspense>
      </main>
      <footer className="muted">
        Static board over <code>build/*.json</code> (packages/lake). No backend; refresh data with{" "}
        <code>bun run lake</code>.{" "}
        <Suspense fallback={null}>
          <DataStamp />
        </Suspense>
      </footer>
    </div>
  );
}

function RootError({ error }: { error: unknown }) {
  return (
    <section className="card error">
      <h1>Something broke</h1>
      <pre>{error instanceof Error ? error.message : String(error)}</pre>
      <p className="muted">
        If the message says the data files are missing, generate them with <code>bun run lake</code>{" "}
        at the repo root, then reload.
      </p>
    </section>
  );
}

const rootRoute = createRootRoute({
  component: RootLayout,
  errorComponent: RootError,
});

/** Router factory: the browser entry uses the default history; the build-time
 *  prerenderer passes a memory history per rendered path. */
export function createAppRouter(history?: RouterHistory) {
  return createRouter({ routeTree, history });
}

const picksRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/",
  component: PicksPage,
});
const exploreRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/explore",
  component: ExplorerPage,
});
const compareRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/compare",
  component: ComparePage,
});
// Splat: model ids contain "/" (fastino/gliner2.5-base), so /model/$ captures
// the whole tail and params._splat is the model_id.
const modelRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/model/$",
  component: ModelPage,
});

const nerRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/ner",
  component: NerPage,
});

const routeTree = rootRoute.addChildren([
  picksRoute,
  exploreRoute,
  compareRoute,
  nerRoute,
  modelRoute,
]);

export const router = createRouter({ routeTree });

declare module "@tanstack/react-router" {
  interface Register {
    router: typeof router;
  }
}
