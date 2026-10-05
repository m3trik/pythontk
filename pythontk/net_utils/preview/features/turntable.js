/*
  Turntable — rotate the model for a hands-free look on the desktop. It holds
  still while a headset is presenting: the model is shown at its true size, so
  there a turning model is the world turning around the viewer, and the button
  that stops it is out of reach in the page behind the headset.

  One of the two scripts packaged with `PreviewServer.SCRIPTS`, and the smaller
  of the pair on purpose: it is the reference for what a viewer script IS. A
  script is an ES module whose default export is called once, with the page's
  viewer API, the first time the manifest names it. Everything it needs to hook
  into is on that object — nothing here reaches into the page's internals, so
  the viewer can be rewritten around it.

  Activate:  bridge.push(scripts=["turntable"])
*/

//: Degrees per second. Slow enough to read a surface at, fast enough that a
//: full revolution does not outlast the reviewer's patience (~26 s).
const DEGREES_PER_SECOND = 14;

export default function turntable(viewer) {
  let spinning = true;

  // A switch in the View window (`viewer.window`), where the page sorts what
  // changes how the model is shown, rather than a button of its own on the bar.
  const view = viewer.window('View', { title: 'How the model is shown' });
  const toggle = view.section().addToggle(
    'Turntable',
    { value: spinning, title: 'Rotate the model continuously (t)' },
    (on) => { spinning = on; },
  );

  viewer.on('key', (event) => {
    if (event.key === 't') {
      spinning = !spinning;
      toggle.value = spinning;
    }
  });

  // Applied to the pivot, not the model: the pivot is what the viewer owns
  // across loads, so the rotation survives a push instead of being reset to
  // zero with the new model — the whole point of a preview that stays open.
  viewer.on('frame', ({ delta }) => {
    if (!spinning || viewer.renderer.xr.isPresenting) return;
    viewer.pivot.rotation.y += (DEGREES_PER_SECOND * Math.PI / 180) * delta;
  });
}
