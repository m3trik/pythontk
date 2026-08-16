# net_utils

Networking primitives and the transports built on them: credentials, SSH, the live WebXR preview server, and both halves of the ecosystem's plugin-hosted JSON-RPC protocol.

Everything is reachable from the package root (`import pythontk as ptk`). Full signatures: [`API_INDEX.md`](../../API_INDEX.md) / [`API_REGISTRY.md`](../../API_REGISTRY.md).

## Small helpers

- **`NetUtils`** (`_net_utils.py`) — `is_port_open`, `is_port_bindable`, `get_local_ip`, and `connect_rdp` (writes a temp `.rdp`, stores credentials at the exact `TERMSRV/{host}` target Windows reads, launches `mstsc.exe`; Windows-only).
- **`Credentials`** (`credentials.py`) — OS-level secure credential storage with a three-tier priority: the `keyring` library if installed → native Windows Credential Manager via pywin32 → environment variables as the headless/CI fallback. Always importable; each tier is guarded.
- **`SSHClient`** (`ssh_client.py`) — unified wrapper over optional **Paramiko**: secure credential retrieval (delegates to `Credentials`), streaming or captured command output, PTY allocation, file upload/download, context-manager lifecycle. Raises a message-bearing `ImportError` at construction, not at import.

## Live preview server (`preview_server.py`)

The transport half of the [Live WebXR preview](../../docs/webxr_preview.md) pipeline (DCC → FBX → GLB → browser/headset):

- **`PreviewServer`** — threaded localhost static-file server with a live `manifest.json`; `publish()` bumps a version and the already-open page hot-swaps the model without a reload. Localhost is a design decision, not a limitation: `navigator.xr` requires a secure context and `http://localhost` is one by definition, so loopback buys a full `immersive-vr` session with no TLS for every PC-tethered headset. The bundled three.js viewer page (`preview_viewer.html`) ships in this directory.
- **`PreviewDeliverer`** — the `Deliverer` strategy that converts the produced FBX to GLB (via `MeshConvert`) and publishes it; `open_browser="auto"` opens a tab only when nothing is currently watching. `texture_format` selects WebP (default) or KTX2 delivery.
- **`PreviewBridge`** — the `HandoffBridge` subclass supplying glTF-appropriate export defaults (embedded textures) plus the publish/URL surface. It lives here — not mirrored per-DCC — because mayatk and blendertk cannot import each other, and anything written in both drifts in both.

## JSON-RPC (`rpc/`)

Both ends of one protocol, deliberately co-located so the wire format cannot drift:

- **`client.py` — `RpcClient`**, the *outside* half. HTTP JSON-RPC over loopback, stdlib `urllib` only: `GET /health`, `POST /` with `{"op", "kwargs"}`, `POST /describe`. Adapters subclass to bind port / app finder / label. Session-safety guarantee: `shutdown()` only touches a process that `connect()` itself launched — a host app the user opened manually is never killed.
- **`plugin_core.py` — `RpcPlugin`**, the *inside* half, running within Marmoset Toolbag, Substance Painter, or any host that can import a package and keep it alive. `OpRegistry` (decorator-based op table), `MainThreadMarshaller` (hops calls onto the host's Qt main thread; resolves PySide6 → PySide2 → none, so it stays importable without Qt), and the `RpcPlugin` facade serving the routes. Everything host-specific is *data* on the class, so one core serves every host.

  **Stdlib-only, no pythontk imports — by contract.** Installed plugin payloads (mayatk/blendertk's `marmoset_rpc` / `substance_rpc`) carry a verbatim copy as `_rpc_core.py`, staged by `m3trik/scripts/sync_rpc_core.py` (`--check` is a drift CI gate). Never hand-edit a staged copy — edit `plugin_core.py` and re-run the sync.
- **`installer.py` — `PluginInstaller`** — install *strategy* only (the adapter resolves the destination): symlink first (zero drift, live edits), `copytree` fallback. Because the fallback is a snapshot, installs are **content**-checked, not presence-checked — otherwise a machine without symlink rights keeps serving the ops it shipped with and an update surfaces as "unknown op". Bytecode is filtered (the host DCC's Python may not match the workspace Python).
- **`job.py` — `RpcJob`** — one-shot batch pipeline over an `RpcClient`: ping once, run every `Call`, capture per-call ok/value/error (`stop_on_error` to short-circuit). Deliberately does not auto-launch the host.

## Links

- Package overview: [`docs/README.md`](../../docs/README.md)
- WebXR preview pipeline: [`docs/webxr_preview.md`](../../docs/webxr_preview.md)
