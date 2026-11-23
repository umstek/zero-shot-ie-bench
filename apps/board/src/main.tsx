import { RouterProvider } from "@tanstack/react-router";
import { StrictMode } from "react";
import { createRoot, hydrateRoot } from "react-dom/client";
import { router } from "./router";
import "./index.css";

const container = document.getElementById("root");
if (!container) throw new Error("#root not found in index.html");

const app = (
  <StrictMode>
    <RouterProvider router={router} />
  </StrictMode>
);

// The build-time prerenderer emits this markup per route; hydration keeps the
// served content for crawlers and attaches interactivity on top. Vite dev and
// the empty dist/index.html SPA fallback start empty and mount with createRoot.
if (container.hasChildNodes()) {
  hydrateRoot(container, app);
} else {
  createRoot(container).render(app);
}
