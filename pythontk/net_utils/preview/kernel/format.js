// How the page words a count and a size -- one spelling for the HUD, the
// status line and every script (`viewer.formatBytes`).

export function plural(count, word) {
  return `${count.toLocaleString()} ${word}${count === 1 ? '' : 's'}`;
}

// A size as the page quotes one -- tenths below 10 of a unit. Scripts get it as
// `viewer.formatBytes`, so a panel (the packaged `inspect`) and the status line
// quote one file one way: two spellings disagreed once ("7 KB" / "6.6 KB").
export function formatBytes(bytes) {
  if (bytes < 1024) return `${Math.round(bytes)} B`;
  const [value, unit] = bytes < 1024 * 1024 ? [bytes / 1024, 'KB'] : [bytes / (1024 * 1024), 'MB'];
  return `${value.toFixed(value < 10 ? 1 : 0)} ${unit}`;
}
