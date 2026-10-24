// Bun's HTML-import default export (src/server.ts imports src/ui/index.html):
// Bun bundles the page and its <script>/<link> assets at serve time — no
// bundler config needed (docs/plan/02 "Hosted side").
declare module "*.html" {
  const html: string;
  export default html;
}
