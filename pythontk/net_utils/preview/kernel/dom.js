// The page's own elements, looked up once: what every kernel module writes
// the page's chrome through -- and the one way the page builds an element
// and saves a file. A script never reaches them -- it is handed `viewer`
// (api.js) and adds its own controls through that.

export const el = {
  title: document.getElementById('title'),
  status: document.getElementById('status'),
  stats: document.getElementById('stats'),
  issues: document.getElementById('issues'),
  hint: document.getElementById('hint'),
  panels: document.getElementById('panels'),
  // The menubar's category buttons (menus.js), and their dropdowns.
  categories: document.getElementById('categories'),
  menus: document.getElementById('menus'),
  // The playing clip's description (captions.js).
  caption: document.getElementById('caption'),
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

// A *tag* element of *className*, its *text* written as TEXT, never markup:
// a label, a row's value, a section's title can all come out of the
// deliverable. What the page's chrome (menus.js, windows.js) is built from.
export function node(tag, className, text) {
  const made = document.createElement(tag);
  if (className) made.className = className;
  if (text !== undefined) made.textContent = text;
  return made;
}

// Save a file this server serves as a download, rather than navigating the
// page to it -- which in a headset would replace the preview with an image
// viewer or a video player. *name* is the file's name on the viewer's disk
// ('' keeps the served one). See `viewer.download` for the origin rule.
export function download(url, name = '') {
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = name;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
}
