import { Link } from "@tanstack/react-router";
import {
  type ColumnFiltersState,
  columnFacetingFeature,
  columnFilteringFeature,
  createColumnHelper,
  createFacetedRowModel,
  createFacetedUniqueValues,
  createFilteredRowModel,
  createSortedRowModel,
  type FilterFn,
  flexRender,
  rowSortingFeature,
  type SortingState,
  type Table,
  tableFeatures,
  useTable,
} from "@tanstack/react-table";
import { useState } from "react";
import { measuredPerItem } from "../lib/cost";
import { fmtDate, fmtMs, fmtPct, fmtUsdCompact, platformOf, zdrLabel } from "../lib/format";
import { latestMixedPoolRun } from "../lib/latest";
import { useBoardData } from "../queries/data";
import type { Deployment, Run } from "../types";

// Faceted, in-memory filter over build/deployments.json — no backend involved.
// TanStack Table v9: features are registered explicitly (filtering, faceting,
// sorting) with their row-model factories; everything else stays core.

/** Deployment row plus its canonical mixed-pool run — the single source for
 *  accuracy, p50, price and date so one row can never fuse two runs. */
type ExplorerRow = Deployment & {
  mixed: Run | null;
  p50_ms: number | null;
  perItem: number | null;
};

const features = tableFeatures({
  columnFilteringFeature,
  columnFacetingFeature,
  rowSortingFeature,
  filteredRowModel: createFilteredRowModel(),
  sortedRowModel: createSortedRowModel(),
  facetedRowModel: createFacetedRowModel(),
  facetedUniqueValues: createFacetedUniqueValues(),
});

/** Multi-select enum filter: empty selection = show everything. */
const multiSelect: FilterFn<typeof features, ExplorerRow> = (row, columnId, filterValue) => {
  if (!Array.isArray(filterValue) || filterValue.length === 0) return true;
  return filterValue.includes(String(row.getValue(columnId)));
};

const helper = createColumnHelper<typeof features, ExplorerRow>();

const columns = helper.columns([
  helper.accessor("display", {
    header: "Model",
    cell: (info) => (
      <Link to="/model/$" params={{ _splat: info.row.original.model_id }}>
        {info.getValue() ?? info.row.original.model_id}
      </Link>
    ),
  }),
  helper.accessor("family", { header: "Family", filterFn: multiSelect }),
  helper.accessor("kind", { header: "Kind", filterFn: multiSelect }),
  helper.accessor((d) => platformOf(d), {
    id: "platform",
    header: "Provider / runner",
    filterFn: multiSelect,
  }),
  helper.accessor("status", { header: "Status", filterFn: multiSelect }),
  helper.accessor((d) => zdrLabel(d.zdr_safe), {
    id: "zdr",
    header: "ZDR",
    filterFn: multiSelect,
    cell: (info) => (
      <span className={`badge zdr-${info.getValue().replace(" ", "-")}`}>{info.getValue()}</span>
    ),
  }),
  helper.accessor("open_weights", {
    header: "Open weights",
    cell: (info) => (info.getValue() ? "yes" : "no"),
  }),
  helper.accessor("license", { header: "License" }),
  helper.accessor((d) => d.claims.tracks.join(", "), { id: "tracks", header: "Claims (tracks)" }),
  helper.accessor((d) => d.requires?.vram_gb ?? 0, {
    id: "vram",
    header: "VRAM GB",
    cell: (info) => (info.row.original.requires ? info.getValue() : "—"),
  }),
  helper.accessor((d) => d.mixed?.accuracy ?? -1, {
    id: "mixed_accuracy",
    header: "Mixed acc",
    cell: (info) => fmtPct(info.row.original.mixed?.accuracy),
  }),
  // measured figures from the same mixed-pool run; Infinity keeps unpriced /
  // untimed rows at the bottom of an ascending sort instead of faking zero
  helper.accessor((d) => d.p50_ms ?? Number.POSITIVE_INFINITY, {
    id: "p50",
    header: "p50",
    cell: (info) => fmtMs(info.row.original.p50_ms),
  }),
  helper.accessor((d) => d.perItem ?? Number.POSITIVE_INFINITY, {
    id: "price_per_1k",
    header: "$ / 1k",
    cell: (info) =>
      info.row.original.perItem == null ? "—" : fmtUsdCompact(info.row.original.perItem),
  }),
  helper.accessor((d) => d.mixed?.started_at ?? "", {
    id: "last_run",
    header: "Mixed run",
    cell: (info) => {
      const at = info.row.original.mixed?.started_at;
      return at ? fmtDate(at) : "—";
    },
  }),
]);

const FACETS: { id: string; label: string }[] = [
  { id: "kind", label: "kind" },
  { id: "family", label: "family" },
  { id: "platform", label: "provider / runner" },
  { id: "status", label: "status" },
  { id: "zdr", label: "ZDR" },
];

function Facet({
  table,
  columnId,
  label,
}: {
  table: Table<typeof features, ExplorerRow>;
  columnId: string;
  label: string;
}) {
  const column = table.getColumn(columnId);
  if (!column) return null;
  const facets = [...column.getFacetedUniqueValues().entries()].sort((a, b) =>
    String(a[0]).localeCompare(String(b[0])),
  );
  const selected = new Set((column.getFilterValue() as string[] | undefined) ?? []);
  const toggle = (value: string) => {
    const next = new Set(selected);
    if (next.has(value)) next.delete(value);
    else next.add(value);
    column.setFilterValue(next.size === 0 ? undefined : [...next]);
  };
  return (
    <details className="facet">
      <summary>
        {label}
        {selected.size > 0 ? ` (${selected.size})` : ""}
      </summary>
      <ul>
        {facets.map(([value, count]) => {
          const v = String(value);
          return (
            <li key={v || "(none)"}>
              <label>
                <input type="checkbox" checked={selected.has(v)} onChange={() => toggle(v)} />
                {v === "" ? "(none)" : v} <span className="count">{count}</span>
              </label>
            </li>
          );
        })}
      </ul>
    </details>
  );
}

export function ExplorerPage() {
  const { deployments, runs } = useBoardData();
  const [sorting, setSorting] = useState<SortingState>([{ id: "mixed_accuracy", desc: true }]);
  const [columnFilters, setColumnFilters] = useState<ColumnFiltersState>([]);

  const latest = latestMixedPoolRun(runs);
  const data: ExplorerRow[] = deployments.map((d) => {
    const run = latest.get(d.deployment_id) ?? null;
    return {
      ...d,
      mixed: run,
      p50_ms: run?.p50_ms ?? null,
      perItem: run ? measuredPerItem(run) : null,
    };
  });

  const table = useTable({
    features,
    columns,
    data,
    state: { sorting, columnFilters },
    onSortingChange: setSorting,
    onColumnFiltersChange: setColumnFilters,
  });

  const rows = table.getRowModel().rows;

  return (
    <section>
      <header className="page-head">
        <h1>Explorer</h1>
        <p>
          All {deployments.length} catalog deployments, filtered in memory. Accuracy, p50, $/1k and
          the run date all come from one run per system — its latest scored decision/mixed-pool run
          (measured spend only — free tiers show —) — the same selection the picks and charts pages
          use. Multilingual-pool results live on each model's page. Facets count only the rows that
          survive the other active filters.
        </p>
      </header>

      <div className="facet-bar">
        {FACETS.map((f) => (
          <Facet key={f.id} table={table} columnId={f.id} label={f.label} />
        ))}
        {columnFilters.length > 0 && (
          <button type="button" className="linkish" onClick={() => setColumnFilters([])}>
            clear filters
          </button>
        )}
      </div>

      <p className="muted table-status">
        {rows.length} of {deployments.length} deployments
      </p>

      <div className="table-wrap">
        <table className="data-table">
          <thead>
            {table.getHeaderGroups().map((hg) => (
              <tr key={hg.id}>
                {hg.headers.map((header) => (
                  <th key={header.id}>
                    {header.isPlaceholder ? null : (
                      <button
                        type="button"
                        className="th-sort"
                        onClick={header.column.getToggleSortingHandler()}
                      >
                        {flexRender(header.column.columnDef.header, header.getContext())}
                        {{ asc: " ▲", desc: " ▼" }[header.column.getIsSorted() as string] ?? ""}
                      </button>
                    )}
                  </th>
                ))}
              </tr>
            ))}
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.id}>
                {row.getAllCells().map((cell) => (
                  <td key={cell.id}>{flexRender(cell.column.columnDef.cell, cell.getContext())}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
