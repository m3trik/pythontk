import { el } from './dom.js';

// Resolver of the prompt currently on screen, and the flag for "a dialog is
// open" — the page's keyboard shortcuts read it, because Space toggling a
// checkbox must not also pause the transport behind the modal.
export let dialogResolve = null;

// The page behind a prompt is INERT while it is up, not merely unbound:
// swallowing its shortcuts left Tab free to walk focus out of the modal to the
// controls it covers, the clip picker among them -- where an arrow key changes
// the clip the prompt names. Every top-level node but the dialog, so the
// transport, the bar, the canvas and the VR button go together.
function setPageInert(inert) {
  for (const node of document.body.children) {
    if (node !== el.dialog) /** @type {HTMLElement} */ (node).inert = inert;
  }
}

export function closeDialog(confirmed) {
  if (!dialogResolve) return;
  const resolve = dialogResolve;
  dialogResolve = null;
  el.dialog.hidden = true;
  setPageInert(false);
  resolve(confirmed);
}

// See `viewer.showDialog` for the contract this implements.
export function showDialog({ title = '', fields = [], confirm = 'OK' } = {}) {
  // A second prompt REPLACES the first rather than stacking: there is one
  // modal, and leaving an orphaned promise unresolved would hang whichever
  // script is waiting on it.
  closeDialog(false);
  el.dialogTitle.textContent = title;
  //: key -> a reader for that field's answer, so the resolve below needs no
  //: idea which kind of control each one is.
  const answers = new Map();
  el.dialogFields.replaceChildren(...fields.map((field) => {
    if (field.type === 'note') {
      const note = document.createElement('div');
      note.className = 'note';
      note.textContent = field.label;
      return note;
    }
    const label = document.createElement('label');
    if (field.title) label.title = field.title;
    if (field.type === 'choice') {
      label.className = 'choice';
      const select = document.createElement('select');
      select.className = 'ui';
      for (const choice of field.choices || []) {
        const option = document.createElement('option');
        option.value = choice.value;
        option.textContent = choice.label;
        if (choice.title) option.title = choice.title;
        select.append(option);
      }
      select.value = field.value ?? '';
      // Assigning a value no option carries leaves NOTHING selected, and a
      // prompt that confirms an empty answer hands the caller a value it never
      // offered -- a remembered choice the page has since dropped, say.
      if (select.selectedIndex < 0 && select.options.length) select.selectedIndex = 0;
      label.append(document.createTextNode(field.label), select);
      answers.set(field.key, () => select.value);
      return label;
    }
    const input = document.createElement('input');
    input.type = 'checkbox';
    input.checked = !!field.value;
    // A text NODE rather than innerHTML: a field label can carry a shot name
    // or a description straight out of the deliverable.
    label.append(input, document.createTextNode(field.label));
    answers.set(field.key, () => input.checked);
    return label;
  }));
  el.dialogConfirm.textContent = confirm;
  el.dialog.hidden = false;
  setPageInert(true);
  // The confirm button, not the first field: Enter then does the obvious
  // thing, and the common answer to an options prompt is "yes, as they are".
  el.dialogConfirm.focus();
  return new Promise((resolve) => {
    dialogResolve = (confirmed) => resolve(
      confirmed
        ? Object.fromEntries([...answers].map(([key, read]) => [key, read()]))
        : null
    );
  });
}

el.dialogForm.addEventListener('submit', (event) => {
  event.preventDefault();
  closeDialog(true);
});
el.dialogCancel.addEventListener('click', () => closeDialog(false));
// The backdrop only — a click that started inside the panel and drifted out
// (selecting a label's text) lands on the form, not on this.
el.dialog.addEventListener('mousedown', (event) => {
  if (event.target === el.dialog) closeDialog(false);
});
