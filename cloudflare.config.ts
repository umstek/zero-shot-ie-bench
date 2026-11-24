import { defineConfig } from "cf/config";

export default defineConfig(({ mode }) => {
  const isStaging = mode === "staging";

  return {
    worker: {
      name: isStaging ? "zsib-staging" : "zsib",
      compatibilityDate: "2026-11-23",
      domains: [isStaging ? "zsib.staging.umstek.com" : "zsib.umstek.com"],
      assets: {
        htmlHandling: "auto-trailing-slash",
        notFoundHandling: "single-page-application",
      },
    },
  };
});
