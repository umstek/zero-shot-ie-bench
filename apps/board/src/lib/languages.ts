// Language metadata, mirroring catalog/languages.yaml (the board reads only the
// built JSON, so tier/ordering info is repeated here on purpose).

/** Benched languages in tier order (popular, medium, rare), template lang last. */
export const LANG_ORDER = ["es", "fr", "zh", "vi", "tr", "uk", "si", "is", "cy", "en"] as const;

export const LANG_NAMES: Record<string, string> = {
  en: "English",
  es: "Spanish",
  fr: "French",
  zh: "Chinese",
  vi: "Vietnamese",
  tr: "Turkish",
  uk: "Ukrainian",
  si: "Sinhala",
  is: "Icelandic",
  cy: "Welsh",
};

export const RARE_LANGS: readonly string[] = ["si", "is", "cy"];

export function langLabel(code: string): string {
  return LANG_NAMES[code] ?? code;
}

/** Languages present in the data, in canonical order, extras appended sorted. */
export function orderLangs(present: Set<string>): string[] {
  const known = [...LANG_ORDER].filter((lang) => present.has(lang));
  const extra = [...present]
    .filter((lang) => !(LANG_ORDER as readonly string[]).includes(lang))
    .sort();
  return [...known, ...extra];
}
