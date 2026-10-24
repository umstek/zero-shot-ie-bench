// Markdown table utilities shared by bench render and the acceptance
// comparator: parse pipe tables (with the heading that precedes them) and
// render cell matrices back to markdown.

export type MdTable = {
  heading: string; // nearest ## / ### heading above the table
  header: string[];
  rows: string[][];
};

export function parseTables(md: string): MdTable[] {
  const lines = md.split(/\r?\n/);
  const tables: MdTable[] = [];
  let heading = "";
  let current: string[] = [];

  const flush = () => {
    if (current.length === 0) return;
    const block = current;
    current = [];
    // a table = header row, separator row, ≥0 data rows
    if (block.length < 2) return;
    const cells = (line: string) =>
      line
        .replace(/^\|/, "")
        .replace(/\|$/, "")
        .split("|")
        .map((c) => c.trim());
    const header = cells(block[0] ?? "");
    const sep = block[1] ?? "";
    if (!/^\|?[\s:-]+\|/.test(sep) || !sep.includes("-")) return;
    const rows = block.slice(2).map(cells);
    tables.push({ heading, header, rows });
  };

  for (const line of lines) {
    if (line.startsWith("#")) {
      flush();
      heading = line.replace(/^#+\s*/, "");
      continue;
    }
    if (line.trim().startsWith("|")) {
      current.push(line.trim());
    } else {
      flush();
    }
  }
  flush();
  return tables;
}

export function renderTable(header: string[], rows: string[][]): string {
  const line = (cells: string[]) => `| ${cells.join(" | ")} |`;
  const sep = `|${header.map(() => "---").join("|")}|`;
  return [line(header), sep, ...rows.map(line)].join("\n");
}

// strip **bold** and formatting-only noise so cell comparison can be exact
// on content; use when the generated table deliberately carries the same
// formatting as the target
export function normalizeCell(cell: string): string {
  return cell.replace(/\*\*/g, "").trim();
}
