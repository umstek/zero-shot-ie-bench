import { RouterProvider } from "@tanstack/react-router";
import { StrictMode } from "react";
import { hydrateRoot } from "react-dom/client";
import { router } from "./router";
import "./index.css";

const container = document.getElementById("root");
if (!container) throw new Error("#root not found in index.html");

// The build-time prerenderer emits this markup per route; hydration keeps the
// served content for crawlers and attaches interactivity on top. The layout's
// Suspense boundary lets pending client-side data load without wiping content.
hydrateRoot(
  container,
  <StrictMode>
    <RouterProvider router={router} />
  </StrictMode>,
);
