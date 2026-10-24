// Suite schemas — versioned question pools under suites/<track>/<suite-id>/.
// Docs: docs/plan/04-data-model.md.
import { z } from "zod";

export const SuiteMeta = z.looseObject({
  id: z.string().min(1), // "<track>/<suite-id>"
  version: z.number().int().min(1),
  track: z.string().min(1),
  languages: z.array(z.string()).min(1),
  scorer: z.string().min(1), // module in bench/src/tracks/<track>/
  items_license: z.string().nullable(),
});

// Parallel items: the same item_id appears in every language file.
export const SuiteItem = z.looseObject({
  item_id: z.string().min(1),
  lang: z.string().min(1),
  tier: z.enum(["popular", "medium", "rare"]).nullable(),
  input: z.looseObject({
    text: z.string().nullable(),
    images: z.array(z.string()).default([]),
  }),
  question: z.looseObject({
    type: z.string(), // choice | score | noul | ner-labels | …
    options: z.array(z.string()).nullable(),
  }),
  gold: z.unknown(),
});
export type SuiteMetaOut = z.output<typeof SuiteMeta>;
export type SuiteItemOut = z.output<typeof SuiteItem>;
