// The page's chrome for what it offers beyond the model: PANELS of rows,
// toggles, sliders and buttons, and the WINDOWS they are sorted into.
//
// The page sorts its controls into a few CATEGORIES -- View, Environment,
// Inspect, Export, Rig -- rather than a bar button per feature. The bar
// carries one button per category that has something in it (beside Frame,
// the one action that stays on the bar), and each opens its window, where
// every part of the page that files options under that category -- the
// kernel's own and any script's -- has a SECTION of its own. So the bar stays
// the same few buttons however many scripts a push activates, a phone-sized
// viewport keeps every control in reach, and a script's options land beside
// the page's own for the same thing instead of in a panel nobody can find.
//
// Everything is written as TEXT, never markup: a row's value, a toggle's
// label, a section's title can all come out of the deliverable.

import { el } from './dom.js';

// The categories the page's own parts file under, in bar order. One a script
// names that is not here follows them, in the order it is first named.
export const CATEGORIES = Object.freeze(['View', 'Environment', 'Inspect', 'Export', 'Rig']);

//: category -> its window, made on first ask.
const windows = new Map();

function node(tag, className, text) {
  const made = document.createElement(tag);
  if (className) made.className = className;
  if (text !== undefined) made.textContent = text;
  return made;
}

/**
 * One row of a panel or section (`setRows`): a `[label, value, level]` pair,
 * a `{heading}`, or a line of its own `{text, level}`; `level: 'warn'` tints it.
 * @typedef {[string, *, string?] | {heading: string} | {text: string, level?: string}} PanelRow
 */

// A body of rows, controls and buttons -- a free panel's and a window
// section's alike, so the two cannot drift: rows a script refreshes, under
// them the controls (refreshing rows never takes a slider out from under a
// drag), and a footer of buttons. *changed* hears every change to its size.
function body(parent, changed = () => {}) {
  const rows = node('div', 'rows');
  const controls = node('div', 'rows controls');
  const footer = node('footer');
  parent.append(rows, controls, footer);
  return {
    /** @param {PanelRow[]} entries */
    setRows(entries) {
      rows.replaceChildren(...entries.flatMap((entry) => {
        if (Array.isArray(entry)) {
          const [key, value, level] = entry;
          return [node('div', 'key', key), node('div', `value${level === 'warn' ? ' warn' : ''}`, String(value ?? ''))];
        }
        const line = /** @type {{heading?: string, text?: string, level?: string}} */ (entry);
        if (line.heading) return [node('div', 'heading', line.heading)];
        return [node('div', `line${line.level === 'warn' ? ' warn' : ''}`, line.text ?? '')];
      }));
      changed();
    },
    addButton(label, onClick) {
      const button = node('button', 'ui', label);
      button.type = 'button';
      button.addEventListener('click', onClick);
      footer.append(button);
      changed();
      return button;
    },
    /**
     * A labelled slider under the rows: `{min, max, step, value, format}`
     * (`format(value)` -> the text beside it), `onInput(value)` on every
     * move. Returns `{element, value, setRange(min, max)}`; setting `value`
     * moves the slider without calling `onInput`, so a script can mirror a
     * value it is also driving.
     */
    addSlider(label, { min = 0, max = 1, step = 0.01, value = 0, format = (v) => v.toFixed(2) } = {}, onInput) {
      const input = node('input');
      input.type = 'range';
      input.min = String(min);
      input.max = String(max);
      input.step = String(step);
      input.value = String(value);
      const output = node('output', '', format(Number(input.value)));
      const wrap = node('div', 'slider');
      wrap.append(input, output);
      controls.append(node('div', 'key', label), wrap);
      changed();
      input.addEventListener('input', () => {
        output.textContent = format(Number(input.value));
        onInput?.(Number(input.value));
      });
      return {
        element: wrap,
        get value() { return Number(input.value); },
        set value(v) {
          input.value = String(v);
          output.textContent = format(Number(input.value));
        },
        setRange(lo, hi) {
          input.min = String(lo);
          input.max = String(hi);
        },
      };
    },
    /**
     * A switch under the rows: a checkbox and its label across the row.
     * `{value, title}`; `onChange(checked)` when the user flips it. Returns
     * `{element, value, hidden, disabled}`; setting `value` flips it without
     * calling `onChange`, as a slider's does.
     */
    addToggle(label, { value = false, title = '' } = {}, onChange) {
      const input = node('input');
      input.type = 'checkbox';
      input.checked = Boolean(value);
      const row = node('label', 'toggle');
      if (title) row.title = title;
      row.append(input, document.createTextNode(label));
      controls.append(row);
      changed();
      input.addEventListener('change', () => onChange?.(input.checked));
      return {
        element: row,
        get value() { return input.checked; },
        set value(on) { input.checked = Boolean(on); },
        get hidden() { return row.hidden; },
        set hidden(on) {
          row.hidden = Boolean(on);
          changed();
        },
        get disabled() { return input.disabled; },
        set disabled(on) { input.disabled = Boolean(on); },
      };
    },
  };
}

// A panel of rows in the page's chrome, stacked top-right after the windows,
// for a script with more to say than a status line (`viewer.addPanel`).
// Hidden until shown. A script's options belong in a category's window
// (`windowFor`); a free panel is for a report of its own.
export function addPanel(title) {
  const panel = node('section', 'panel');
  panel.hidden = true;
  panel.append(node('header', '', title));
  const parts = body(panel);
  el.panels.append(panel);
  return {
    element: panel,
    get shown() { return !panel.hidden; },
    show(on = true) { panel.hidden = !on; },
    ...parts,
  };
}

// Where a new category's window and bar button go: after every category
// that comes before it -- CATEGORIES' order, then the order first named.
function placeOf(category) {
  const known = CATEGORIES.indexOf(category);
  return known >= 0 ? known : CATEGORIES.length + windows.size;
}

// The open windows share the column's height, and only the TALLEST gives way
// and scrolls: shrunk alike, a three-row window opened beside Inspect lost
// its first row and its second switch to its own scroll.
function settleHeights() {
  const open = [...windows.values()].filter((win) => win.shown);
  // Its content's whole height, however far the column has squeezed it.
  const height = (win) => win.element.querySelector('.sections').scrollHeight;
  let tallest = null;
  for (const win of open) if (!tallest || height(win) > height(tallest)) tallest = win;
  for (const win of open) win.element.style.flexShrink = win === tallest ? '1' : '0';
}

// A child with no rank -- a free panel, a script's own bar button -- comes
// after every window and category button.
function insertInOrder(parent, element, rank) {
  const rankOf = (child) => (child.dataset.rank === undefined ? Infinity : Number(child.dataset.rank));
  const after = [...parent.children].find((child) => rankOf(child) > rank);
  parent.insertBefore(element, after || null);
}

/**
 * The window of *category* -- see the module comment. The first ask makes it
 * (hidden) and its bar button (shown once a section is added); every later
 * ask with the same name returns the same window, so several scripts file
 * into one. `title` is the bar button's tooltip.
 *
 * Returns `{category, element, button, shown, show(on), toggle(),
 * onShow(fn), setBadge(text), section(title)}`: `onShow(fn)` calls
 * `fn(shown)` whenever it opens or closes; `setBadge(text)` marks the bar
 * button (a job running with the window shut), `''` clears it; `section`
 * adds a block the caller owns -- `{element, shown, show(on), remove(),
 * setRows, addButton, addSlider, addToggle}`, its `title` a heading over it.
 */
export function windowFor(category, { title = '' } = {}) {
  const existing = windows.get(category);
  if (existing) {
    if (title && !existing.button.title) existing.button.title = title;
    return existing;
  }
  const rank = placeOf(category);
  const frame = node('section', 'panel window');
  frame.hidden = true;
  frame.dataset.category = category;
  frame.dataset.rank = String(rank);
  const header = node('header');
  header.append(node('span', 'title', category));
  // The glyph is the stylesheet's, so the header's text stays the category.
  const close = node('button', 'close');
  close.type = 'button';
  close.title = `Close ${category}`;
  close.setAttribute('aria-label', `Close ${category}`);
  header.append(close);
  const sections = node('div', 'sections');
  frame.append(header, sections);
  insertInOrder(el.panels, frame, rank);

  const button = node('button', 'ui category', category);
  button.type = 'button';
  button.hidden = true;
  button.dataset.category = category;
  button.dataset.rank = String(rank);
  if (title) button.title = title;
  insertInOrder(el.categories, button, rank);

  const listeners = [];
  const show = (on = true) => {
    const shown = Boolean(on);
    if (frame.hidden === !shown) return;
    frame.hidden = !shown;
    button.classList.toggle('open', shown);
    button.setAttribute('aria-pressed', String(shown));
    for (const fn of listeners) {
      try {
        fn(shown);
      } catch (error) {
        console.warn(`${category} window listener threw:`, error);
      }
    }
    // After the listeners, which may fill it (Inspect paints as it opens).
    settleHeights();
  };
  button.addEventListener('click', () => show(frame.hidden));
  close.addEventListener('click', () => show(false));

  const win = {
    category,
    element: frame,
    button,
    get shown() { return !frame.hidden; },
    show,
    toggle() { show(frame.hidden); },
    onShow(fn) { listeners.push(fn); },
    setBadge(text) {
      button.textContent = text ? `${category} · ${text}` : category;
      button.classList.toggle('busy', Boolean(text));
    },
    section(heading = '') {
      const block = node('div', 'section');
      if (heading) block.append(node('div', 'heading', heading));
      const parts = body(block, settleHeights);
      sections.append(block);
      button.hidden = false;
      return {
        element: block,
        get shown() { return !frame.hidden && !block.hidden; },
        show(on = true) {
          block.hidden = !on;
          settleHeights();
        },
        // Its window empties with it: an empty window has no business on
        // the bar.
        remove() {
          block.remove();
          if (!sections.children.length) {
            show(false);
            button.hidden = true;
          }
          settleHeights();
        },
        ...parts,
      };
    },
  };
  windows.set(category, win);
  return win;
}
