/*
  Turntable — rotate the model for a hands-free look, or for a headset the user
  is not holding a controller in.

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

  const button = viewer.addButton('Turntable: on', () => {
    spinning = !spinning;
    button.textContent = `Turntable: ${spinning ? 'on' : 'off'}`;
  });
  button.title = 'Rotate the model continuously (t)';

  viewer.on('key', (event) => {
    if (event.key === 't') button.click();
  });

  // Applied to the pivot, not the model: the pivot is what the viewer owns
  // across loads, so the rotation survives a push instead of being reset to
  // zero with the new model — the whole point of a preview that stays open.
  viewer.on('frame', ({ delta }) => {
    if (spinning) viewer.pivot.rotation.y += (DEGREES_PER_SECOND * Math.PI / 180) * delta;
  });
}
