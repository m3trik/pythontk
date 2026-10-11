// The page's menubar: one bar button per CATEGORY, each opening a dropdown of
// the ITEMS filed under it -- what the page and its scripts offer beyond the
// model, sorted, rather than a bar button per feature.
//
// An item is one of three kinds, and every part of the page that offers
// something builds it from them:
//
//   action   a one-shot command (Export ▸ GLB). A label ending in `…` asks
//            before it acts (`viewer.showDialog`), the desktop convention.
//   toggle   a checkable switch (View ▸ Turntable). The menu stays open on
//            one, so a run of them -- a file per part, say -- is set in one go.
//   window   a persistent panel of rows and controls (windows.js), checked
//            while it is open: a report to read, sliders to drag.
//
// Every item may declare WHEN it applies (`when()`), whether it can act now
// (`enabled()`), a single-key shortcut (`key`), and where it sits (`group`,
// `order`). That is the whole of the context sensitivity: predicates the menu
// asks as it opens, and as the page's context changes (`refreshMenus` -- a
// load, a share's guest flag, a headset session). A bar button shows only
// while its menu has an item that applies, so a deliverable with nothing to
// pose has no Rig on the bar, and one without a probe offers no probe switch.
//
// One table holds every item's shortcut, so a key can be owned once: the
// second item to claim it is told so in the console and is left unbound,
// rather than two features fighting over one keystroke. The dropdown shows
// each item's key beside it, so a shortcut is discoverable where it is used.
//
// Everything is written as TEXT, never markup: a label can come out of the
// deliverable (a file name, a shot name).

import { el, node } from './dom.js';

// The categories the page's own parts file under, in bar order. One a script
// names that is not here follows them, in the order it is first named.
export const CATEGORIES = Object.freeze(['View', 'Environment', 'Inspect', 'Export', 'Rig']);

//: What each of the page's categories holds -- its bar button's tooltip,
//: stated once here rather than by whichever of its contributors asks first.
const TITLES = Object.freeze({
  View: 'How the model is shown',
  Environment: "What lights the model: the bake's reflection probe, or the page's studio",
  Inspect: 'What the preview costs, measured on this device',
  Export: 'Save the model, the view, or a clip',
  Rig: 'Pose the rigs by hand',
});

//: category -> its menu, made on first ask.
const menus = new Map();
//: shortcut key (lowercase) -> the item that owns it.
const shortcuts = new Map();
//: The menu whose dropdown is open, or null.
let openMenu = null;
//: Insertion counter: the last tie-break of an item's place.
let sequence = 0;

// Where a category's bar button goes: CATEGORIES' order, then the order first
// named.
function rankOf(category) {
  const known = CATEGORIES.indexOf(category);
  return known >= 0 ? known : CATEGORIES.length + menus.size;
}

// A child with no rank -- a script's own bar button, a free panel -- comes
// after every category's.
export function insertInOrder(parent, element, rank) {
  const ranked = (child) => (child.dataset.rank === undefined ? Infinity : Number(child.dataset.rank));
  const after = [...parent.children].find((child) => ranked(child) > rank);
  parent.insertBefore(element, after || null);
}

// A predicate's answer, contained: a script's `when` that throws hides its
// item rather than taking the whole menu down with it.
function asks(predicate, fallback) {
  if (typeof predicate !== 'function') return fallback;
  try {
    return Boolean(predicate());
  } catch (error) {
    console.warn('menu predicate threw:', error);
    return false;
  }
}

/**
 * The menu of *category* -- see the module comment. The first ask makes it
 * (its bar button hidden until an item applies); every later ask with the
 * same name returns the same menu, so the page and several scripts file into
 * one. `title` is the bar button's tooltip for a script's own category,
 * kept from the first ask that names one; the page's own carry theirs.
 *
 * Returns `{category, button, element, shown, open(), close(), toggle(),
 * add(kind, spec), setBadge(text, owner), refresh()}`.
 *
 * `add(kind, spec)` files an item: `kind` is `'action'`, `'toggle'` or
 * `'window'`, and `spec` is `{label, title, key, group, order, when, enabled}`
 * plus the kind's own -- `run()` for an action, `{value, onChange(on)}` for a
 * toggle, `{window}` for a window (windows.js files those). Returns the item:
 * `{element, kind, label, value, hidden, disabled, remove()}` -- `label`
 * settable (an action that becomes its own cancel), `value` a toggle's
 * state, settable without calling `onChange`, and `hidden` / `disabled` a
 * manual override on top of `when` / `enabled`.
 *
 * `setBadge(text, owner)` marks the bar button while a job runs with the
 * menu shut, `''` clears *owner*'s -- each owner holds its own, and the
 * button shows the newest standing.
 */
export function menuFor(category, { title = '' } = {}) {
  const existing = menus.get(category);
  if (existing) {
    if (title && !existing.button.title) existing.button.title = title;
    return existing;
  }
  title = TITLES[category] || title;
  const rank = rankOf(category);
  const button = node('button', 'ui category', category);
  button.type = 'button';
  button.hidden = true;
  button.dataset.category = category;
  button.dataset.rank = String(rank);
  button.setAttribute('aria-haspopup', 'menu');
  button.setAttribute('aria-expanded', 'false');
  if (title) button.title = title;
  insertInOrder(el.categories, button, rank);

  const dropdown = node('div', 'menu');
  dropdown.hidden = true;
  dropdown.dataset.category = category;
  dropdown.setAttribute('role', 'menu');
  dropdown.setAttribute('aria-label', category);
  el.menus.append(dropdown);

  const items = [];
  //: Each job's badge, by owner, in the order set: the button wears the
  //: latest. One menu runs several jobs -- a still saved while a playblast
  //: encodes -- and each clears only its own.
  const badges = new Map();

  // The items that apply now, in their place: groups by the lowest `order`
  // among their items, then each group's items by `order` and arrival.
  function applicable() {
    const shown = items.filter((item) => item.applies());
    const groupOrder = new Map();
    for (const item of shown) {
      const low = groupOrder.get(item.group);
      if (low === undefined || item.order < low) groupOrder.set(item.group, item.order);
    }
    return shown.sort((a, b) => (
      (groupOrder.get(a.group) - groupOrder.get(b.group))
      || (a.group < b.group ? -1 : a.group > b.group ? 1 : 0)
      || (a.order - b.order)
      || (a.seq - b.seq)
    ));
  }

  function render() {
    const rows = [];
    let group = null;
    for (const item of applicable()) {
      if (group !== null && item.group !== group) {
        const separator = node('div', 'separator');
        separator.setAttribute('role', 'separator');
        rows.push(separator);
      }
      group = item.group;
      item.sync();
      rows.push(item.element);
    }
    dropdown.replaceChildren(...rows);
  }

  // Opens UP from the bar, which sits on the page's bottom edge, kept inside
  // the viewport however far right its button is.
  function place() {
    const anchor = button.getBoundingClientRect();
    dropdown.style.bottom = `${Math.round(innerHeight - anchor.top + 6)}px`;
    dropdown.style.maxHeight = `${Math.max(120, Math.round(anchor.top - 18))}px`;
    const width = dropdown.offsetWidth;
    dropdown.style.left = `${Math.round(Math.max(12, Math.min(anchor.left, innerWidth - width - 12)))}px`;
  }

  const menu = {
    category,
    button,
    //: Where the category sits -- its windows go in the same place.
    rank,
    element: dropdown,
    get shown() { return !dropdown.hidden; },
    open() {
      if (openMenu && openMenu !== menu) openMenu.close();
      render();
      dropdown.hidden = false;
      button.classList.add('open');
      button.setAttribute('aria-expanded', 'true');
      place();
      openMenu = menu;
    },
    close() {
      if (dropdown.hidden) return;
      dropdown.hidden = true;
      button.classList.remove('open');
      button.setAttribute('aria-expanded', 'false');
      if (openMenu === menu) openMenu = null;
    },
    toggle() {
      if (dropdown.hidden) menu.open(); else menu.close();
    },
    add(kind, spec = {}) {
      const item = makeItem(menu, kind, spec);
      items.push(item);
      if (item.key) {
        const owner = shortcuts.get(item.key);
        if (owner) {
          console.warn(`menu shortcut '${item.key}' is ${owner.menu.category} ▸ `
            + `${owner.label}'s; ${category} ▸ ${item.label} is left without one.`);
          item.key = '';
        } else {
          shortcuts.set(item.key, item);
        }
      }
      item.detach = () => {
        const at = items.indexOf(item);
        if (at >= 0) items.splice(at, 1);
        if (item.key && shortcuts.get(item.key) === item) shortcuts.delete(item.key);
        menu.refresh();
      };
      menu.refresh();
      return item.handle;
    },
    setBadge(text, owner = '') {
      badges.delete(owner); // set again, it is the latest
      if (text) badges.set(owner, String(text));
      const shown = [...badges.values()].pop();
      button.textContent = shown ? `${category} · ${shown}` : category;
      button.classList.toggle('busy', Boolean(shown));
    },
    // Whether anything here applies now decides the bar button; an open
    // dropdown re-renders, so an item that stopped applying leaves it.
    refresh() {
      const any = items.some((item) => item.applies());
      button.hidden = !any;
      if (!any) menu.close();
      else if (!dropdown.hidden) {
        render();
        place();
      }
    },
    // The items in their place, for the keyboard (menuKeydown).
    get focusable() {
      return [...dropdown.querySelectorAll('.item')].filter((item) => !item.disabled);
    },
  };

  button.addEventListener('click', () => menu.toggle());
  // A menubar's convention: with one dropdown open, pointing at another
  // category's button opens that one instead.
  button.addEventListener('pointerenter', () => {
    if (openMenu && openMenu !== menu) menu.open();
  });

  menus.set(category, menu);
  return menu;
}

// One item: its element, kept for its life -- the dropdown re-renders by
// reordering and hiding elements, never by remaking them, so a handle a
// caller holds stays the one on screen.
function makeItem(menu, kind, spec) {
  if (!['action', 'toggle', 'window'].includes(kind)) {
    throw new Error(`menu item kind '${kind}' is not one of action, toggle, window`);
  }
  const {
    label = '', title = '', key = '', group = '', order = 0,
    when, enabled, run, value = false, onChange, window: win,
  } = spec;
  const element = node('button', `item ${kind}`);
  element.type = 'button';
  element.setAttribute('role', kind === 'action' ? 'menuitem' : 'menuitemcheckbox');
  element.dataset.kind = kind;
  if (title) element.title = title;
  const check = node('span', 'check');
  check.setAttribute('aria-hidden', 'true');
  const text = node('span', 'label', label);
  const hint = node('span', 'key', key ? key.toUpperCase() : '');
  element.append(check, text, hint);

  let checked = Boolean(value);
  let hiddenByCaller = false;
  let disabledByCaller = false;

  const item = {
    menu,
    kind,
    label,
    key: String(key || '').toLowerCase(),
    group: String(group),
    order: Number(order) || 0,
    seq: (sequence += 1),
    element,
    applies: () => !hiddenByCaller && asks(when, true),
    usable: () => !disabledByCaller && asks(enabled, true),
    // The element brought up to date with its state as the dropdown shows it.
    sync() {
      const on = kind === 'window' ? Boolean(win?.shown) : checked;
      if (kind !== 'action') element.setAttribute('aria-checked', String(on));
      element.classList.toggle('checked', kind !== 'action' && on);
      element.disabled = !item.usable();
      hint.textContent = item.key ? item.key.toUpperCase() : '';
    },
    // What clicking it, or its key, does.
    activate() {
      if (!item.applies() || !item.usable()) return;
      if (kind === 'toggle') {
        checked = !checked;
        item.sync();
        onChange?.(checked);
        return;
      }
      // Shut BEFORE it acts: an action that asks (`…`) puts a modal up, and
      // a dropdown left open behind it would be one the modal did not take.
      menu.close();
      if (kind === 'window') win?.toggle();
      else run?.();
    },
  };
  element.addEventListener('click', () => item.activate());

  item.handle = {
    element,
    kind,
    get label() { return item.label; },
    set label(next) {
      item.label = String(next);
      text.textContent = item.label;
    },
    get value() { return kind === 'window' ? Boolean(win?.shown) : checked; },
    set value(on) {
      checked = Boolean(on);
      item.sync();
    },
    get hidden() { return hiddenByCaller; },
    set hidden(on) {
      hiddenByCaller = Boolean(on);
      menu.refresh();
    },
    get disabled() { return disabledByCaller; },
    set disabled(on) {
      disabledByCaller = Boolean(on);
      item.sync();
    },
    remove() { item.detach(); },
  };
  return item;
}

/** Re-ask every menu what applies -- after anything the predicates read changed. */
export function refreshMenus() {
  for (const menu of menus.values()) menu.refresh();
}

/** Shut whichever dropdown is open. */
export function closeMenus() {
  openMenu?.close();
}

// The bar buttons that are on show, in bar order -- what Left and Right walk.
function barMenus() {
  return /** @type {HTMLElement[]} */ ([...el.categories.children])
    .filter((button) => !button.hidden && menus.has(button.dataset.category))
    .map((button) => menus.get(button.dataset.category));
}

/**
 * The menus' share of a keydown: walking an open dropdown, then every item's
 * shortcut. True when it took the key, and the page's own then leave it be --
 * Space on a focused item is that item's, never the transport's.
 */
export function menuKeydown(event) {
  if (event.ctrlKey || event.metaKey || event.altKey) return false;
  if (openMenu) {
    const menu = openMenu;
    const items = menu.focusable;
    const at = items.indexOf(document.activeElement);
    const focus = (index) => items[(index + items.length) % items.length]?.focus();
    switch (event.key) {
      case 'ArrowDown':
        event.preventDefault();
        focus(at + 1);
        return true;
      case 'ArrowUp':
        event.preventDefault();
        focus(at < 0 ? -1 : at - 1);
        return true;
      case 'Home':
      case 'End':
        event.preventDefault();
        focus(event.key === 'Home' ? 0 : -1);
        return true;
      case 'ArrowLeft':
      case 'ArrowRight': {
        event.preventDefault();
        const bar = barMenus();
        const next = bar[(bar.indexOf(menu) + (event.key === 'ArrowRight' ? 1 : -1) + bar.length) % bar.length];
        next?.open();
        next?.focusable[0]?.focus();
        return true;
      }
      case 'Escape':
        event.preventDefault();
        menu.close();
        menu.button.focus();
        return true;
      case 'Tab':
        menu.close();
        return false;
      case 'Enter':
      case ' ':
        // The focused item's own activation (a button's), and nothing else.
        if (at >= 0) return true;
        break;
      default:
        break;
    }
  }
  const owner = shortcuts.get(String(event.key || '').toLowerCase());
  if (!owner || !owner.applies() || !owner.usable()) return false;
  event.preventDefault();
  owner.activate();
  return true;
}

// A press outside the open dropdown and its bar button shuts it, as a resize
// does -- the dropdown is placed against the button where it stood.
addEventListener('pointerdown', (event) => {
  if (!openMenu) return;
  const target = /** @type {Node} */ (event.target);
  if (openMenu.element.contains(target) || openMenu.button.contains(target)) return;
  openMenu.close();
}, true);
addEventListener('resize', closeMenus);
