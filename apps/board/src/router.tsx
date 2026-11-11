import { createRootRoute, createRoute, createRouter, Link, Outlet } from "@tanstack/react-router";
import { Suspense } from "react";
import { ComparePage } from "./routes/compare";
import { ExplorerPage } from "./routes/explore";
import { ModelPage } from "./routes/model.$id";
import { PicksPage } from "./routes/picks";

// Code-based route tree (no codegen step): the component files live in
// src/routes/ with the planned names, registered here. Planned-but-unbuilt
// views live in docs/board-roadmap.md — no stub pages on the site.

const NAV = [
  { to: "/", label: "Picks", exact: true },
  { to: "/explore", label: "Explorer", exact: false },
  { to: "/compare", label: "Charts", exact: false },
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

function RootLayout() {
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
        <code>bun run lake</code>.
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

const routeTree = rootRoute.addChildren([picksRoute, exploreRoute, compareRoute, modelRoute]);

export const router = createRouter({ routeTree });

declare module "@tanstack/react-router" {
  interface Register {
    router: typeof router;
  }
}
