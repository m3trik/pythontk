// The page's own elements, looked up once: what every kernel module writes
// the page's chrome through. A script never reaches them -- it is handed
// `viewer` (api.js) and adds its own controls through that.

export const el = {
  title: document.getElementById('title'),
  status: document.getElementById('status'),
  stats: document.getElementById('stats'),
  issues: document.getElementById('issues'),
  hint: document.getElementById('hint'),
  panels: document.getElementById('panels'),
  frameBtn: /** @type {HTMLButtonElement} */ (document.getElementById('frameBtn')),
  // The bar's category buttons, one per window (windows.js).
  categories: document.getElementById('categories'),
  anim: document.getElementById('anim'),
  clipSelect: /** @type {HTMLSelectElement} */ (document.getElementById('clipSelect')),
  playToggle: /** @type {HTMLButtonElement} */ (document.getElementById('playToggle')),
  scrub: /** @type {HTMLInputElement} */ (document.getElementById('scrub')),
  clipTime: document.getElementById('clipTime'),
  lookdev: document.getElementById('lookdev'),
  normals: document.getElementById('normals'),
  normalScale: /** @type {HTMLInputElement} */ (document.getElementById('normalScale')),
  normalValue: document.getElementById('normalValue'),
  normalSave: /** @type {HTMLButtonElement} */ (document.getElementById('normalSave')),
  dialog: document.getElementById('dialog'),
  dialogForm: /** @type {HTMLFormElement} */ (document.getElementById('dialogForm')),
  dialogTitle: document.getElementById('dialogTitle'),
  dialogFields: document.getElementById('dialogFields'),
  dialogCancel: /** @type {HTMLButtonElement} */ (document.getElementById('dialogCancel')),
  dialogConfirm: /** @type {HTMLButtonElement} */ (document.getElementById('dialogConfirm')),
};
