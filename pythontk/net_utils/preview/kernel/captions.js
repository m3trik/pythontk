// Clip descriptions: the note the DCC's Shots panel records for a shot, shown
// over the view while the transport is on it -- the half of a shot record a
// reviewer watching the preview cannot otherwise see.
//
// The same answer the playblast burns in (`viewer.descriptionAt`): the
// segment's on FULL SEQUENCE, so the caption changes at each shot boundary
// as the playhead crosses it, and the selected clip's on a single shot. Page
// markup laid over the canvas, so a still or a playblast never carries it --
// a burn-in is the playblast's own choice, asked as it records.
//
// A View switch, offered only while some clip of the model on screen has a
// description, and on by default: nothing shows on a deliverable without
// one. The choice is this browser's (localStorage), so a reviewer who turned
// captions off is not handed them back by the next push.

import { hasDescriptions, mixer, shotDescription } from './animation.js';
import { el } from './dom.js';
import { menuFor } from './menus.js';

//: Where this browser keeps the switch.
const STORE_KEY = 'webxrPreview.captions';

function stored() {
  try {
    return localStorage.getItem(STORE_KEY) !== 'off';
  } catch {
    return true; // storage refused (a private window): the default
  }
}

let enabled = stored();
//: The text on screen, so a frame that changes nothing writes nothing.
let shown = null;

menuFor('View').add('toggle', {
  label: 'Clip descriptions',
  title: "Show the playing shot's description, as the DCC's Shots panel recorded it",
  group: 'animation',
  order: 20,
  value: enabled,
  when: () => mixer !== null && hasDescriptions(),
  onChange: (on) => {
    enabled = on;
    try {
      localStorage.setItem(STORE_KEY, on ? 'on' : 'off');
    } catch {
      // Kept for this tab only.
    }
    updateCaption();
  },
});

// Once per frame, after the transport has posed the model (ui.js): the
// description for where the playhead stands now.
export function updateCaption() {
  const text = enabled && mixer ? shotDescription() : null;
  if (text === shown) return;
  shown = text;
  el.caption.textContent = text || '';
  el.caption.hidden = !text;
}
