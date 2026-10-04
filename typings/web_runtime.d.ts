// Declarations the WebXR preview's type check needs and no library supplies.
//
// three.js reaches the page from a CDN through its import map, so the check
// has no declarations for it: these make its names `any`. What the check
// covers is the runtime's own modules and the contracts between them -- the
// kernel's exports, the `viewer` API, the generated record typedefs -- not
// three.js's API. WebXR is not in TypeScript's DOM library either.

declare module 'three';
declare module 'three/addons/*';

declare const XRRigidTransform: any;

interface Navigator {
  readonly xr?: any;
}

interface Window {
  // The boot watchdog's flag (viewer.html), set by kernel/main.js.
  __viewerBooted?: boolean;
}
