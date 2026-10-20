# !/usr/bin/python
# coding=utf-8
"""Look loop: one round of look matching, from a rig by role to a judged frame, in one call.

A round of lighting work used to be ten steps by hand -- bake the lights that
changed, capture them, fit, write the mix, draw the sheet, shoot the exact
frame, score it. :class:`LookLoop` runs them from one *rig file*: the roles of a
rig (a key, a rim, a practical...), each a light payload with its weight, its
colour temperature and whether the fit may move it.

Only what changed is baked: a role's layer is re-baked when its payload differs
from the one its layer was baked from, so a round that only moves weights bakes
nothing and takes seconds. The bake runs in a DCC worker fed through a folder
(``ptk.JobQueue``; mayatk's ``LightLayerBaker``), the capture in a
:class:`LookCapture` service; both are named by the session's ``loop`` section
or the command line.

    python -m pythontk.core_utils.engines.lighting.look_match.loop iterate session.json --rig rig.json --out round1
    python -m pythontk.core_utils.engines.lighting.look_match.loop status session.json --rig rig.json

Rig file, version 1::

    {"version": 1,
     "roles": {"key": {"lights": [...], "kelvin": 4300, "weight": 1.0, "free": true,
                       "fit_kelvin": true},       # the fit may move its colour too
               "glow": {"materials": {...}, "weight": 0.05}},   # an emission role
     "base": {"materials": {...}},                # overrides every bake shares
     "objects": ["LightmapBaker_SET"],             # what the bake covers
     "settings": {"resolution": 256},              # the baker's
     "fit": {"tolerance": 0.1, "regularize": 0.3,  # LookFit's
             "size": [410, 240]}}                  # the fit's working size
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence

from pythontk.core_utils.cli import CLI
from pythontk.core_utils.engines.lighting.look_match._look_match import LookMatch
from pythontk.core_utils.engines.lighting.light_layers import LightLayerScene, LightMix
from pythontk.core_utils.engines.lighting.look_match.session import LookSession
from pythontk.core_utils.engines.lighting.look_match.sheet import LookSheet
from pythontk.core_utils.engines.lighting.look_match.target import LookTarget
from pythontk.file_utils._file_utils import FileUtils
from pythontk.file_utils.job_queue import JobQueue
from pythontk.net_utils.rpc.client import RpcClient


class LookLoop:
    """Runs rounds of look matching against one session.

    Parameters:
        session: The look-match session JSON (``LookMatch.open``).
        queue: The bake worker's queue folder (default: the session's
            ``loop.queue``).
        service: The capture service's URL (default: the session's
            ``loop.service``, else ``http://127.0.0.1:8770/``).
        client: What asks the capture service: anything with ``invoke(op,
            timeout=..., **arguments)`` (default an ``RpcClient`` of *service*,
            which ``LookCapture.serve`` answers; tests pass a stand-in).
        match: The session already open (a panel's ``LookMatch``: its mix and
            its layers' memory); default: opened from *session*.
    """

    RIG_VERSION = 1
    DEFAULT_SERVICE = "http://127.0.0.1:8770/"
    #: Seconds a capture-service op may take (a capture of many layers, an exact frame).
    TIMEOUT = 3600.0

    def __init__(
        self,
        session: str,
        queue: Optional[str] = None,
        service: Optional[str] = None,
        client: Optional[Any] = None,
        match: Optional[LookMatch] = None,
    ):
        self.session = os.path.abspath(session)
        data = LookSession.read(self.session)
        loop = data.get("loop") or {}
        queue = LookSession.resolve(self.session, data, queue or loop.get("queue"))
        self.queue = JobQueue(queue) if queue else None
        #: The scene the worker bakes (the session's ``loop.worker.scene``), or None.
        self.scene = LookSession.resolve(
            self.session, data, (loop.get("worker") or {}).get("scene")
        )
        self.service = service or loop.get("service") or self.DEFAULT_SERVICE
        self.client = client or RpcClient.for_url(
            self.service, app_label="look capture"
        )
        self.match = match if match is not None else LookMatch.open(self.session)
        #: Seconds a bake waits with no worker beating before it gives up: a
        #: caller that just launched one raises it (a GUI Maya opening a
        #: production scene takes minutes).
        self.worker_stale = 120.0

    # ------------------------------------------------------------------ the rig
    @classmethod
    def load_rig(cls, path: str) -> Dict[str, Any]:
        with open(path, encoding="utf-8") as f:
            rig = json.load(f)
        if rig.get("version", 1) > cls.RIG_VERSION:
            raise ValueError(
                f"Rig file version {rig['version']} is newer than this reads."
            )
        if not isinstance(rig.get("roles"), dict) or not rig["roles"]:
            raise ValueError(f"{path} names no roles.")
        return rig

    @staticmethod
    def payload(role: Mapping[str, Any]) -> Dict[str, Any]:
        """What a role's layer is baked from: its lights, or an emission role's materials."""
        out: Dict[str, Any] = {"lights": list(role.get("lights") or [])}
        if role.get("materials"):
            out["materials"] = role["materials"]
        return out

    @staticmethod
    def spec_hash(payload: Mapping[str, Any]) -> str:
        """A payload's identity: its canonical JSON, hashed."""
        text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]

    def stale(self, rig: Mapping[str, Any]) -> List[str]:
        """The roles whose layer is missing, was baked from another payload, or
        from an older save of the session's scene (:attr:`scene`; a layer with
        no scene record is taken as current)."""
        out = []
        for name, role in rig["roles"].items():
            want = self.spec_hash(self.payload(role))
            if name not in self.match.layers:
                out.append(name)
                continue
            info = self.match.layers.info(name)
            if self.spec_hash(self.payload(info.get("spec") or {})) != want:
                out.append(name)
            elif (
                self.scene
                and info.get("scene")
                and not LightLayerScene.matches(info["scene"], self.scene)
            ):
                out.append(name)
        return out

    def mix_of(self, rig: Mapping[str, Any]) -> LightMix:
        """The rig's mix: each role at its weight, colour temperature and
        specular scale (``"specular"``, Arnold's ``aiSpecular``)."""
        return LightMix(
            {
                name: {
                    "weight": float(role.get("weight", 1.0)),
                    "kelvin": role.get("kelvin"),
                    "specular": float(role.get("specular", 1.0)),
                }
                for name, role in rig["roles"].items()
            },
            notes=str(rig.get("notes", "")),
        )

    # ------------------------------------------------------------------ stages
    def bake(
        self,
        rig: Mapping[str, Any],
        names: Sequence[str],
        folder: str,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Bake *names*' layers in the worker, into *folder*; the worker's result.

        Raises:
            RuntimeError: no worker queue, the bake failed, or a layer was refused.
        """
        job = {
            "layers": {n: self.payload(rig["roles"][n]) for n in names},
            "output": folder,
            "base": rig.get("base") or {},
            "settings": rig.get("settings") or {},
        }
        if rig.get("objects"):
            job["objects"] = list(rig["objects"])
        return self._bake_job(job, timeout)

    def _bake_job(self, job: Mapping[str, Any], timeout: Optional[float]) -> Dict[str, Any]:
        """Run one bake job in the worker; its result (a refused layer raises)."""
        if self.queue is None:
            raise RuntimeError(
                "No bake worker: name its queue (--queue, or the session's loop.queue)."
            )
        result = self.queue.wait(
            self.queue.submit(dict(job)), timeout=timeout, stale=self.worker_stale
        )
        refused = {
            n: r["refused"]
            for n, r in (result.get("layers") or {}).items()
            if "refused" in r
        }
        if refused:
            raise RuntimeError(f"The bake refused {refused}.")
        return result

    def stale_layers(self) -> List[str]:
        """The store's layers baked from an older save of the session's scene
        (:attr:`scene`): what bakes again to show the scene as saved. A layer
        with no scene record is taken as current; with no worker scene, none."""
        if not self.scene:
            return []
        layers, out = self.match.layers, []
        for name in layers.names():
            record = layers.info(name).get("scene")
            if record and not LightLayerScene.matches(record, self.scene):
                out.append(name)
        return out

    def rebake(
        self, names: Sequence[str], folder: str, timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """Bake layers *names* again from what the store recorded of them -- each
        layer's ``spec``, under the store's recorded state (its stamp's
        ``existing`` / ``materials``), with the settings it was baked with (a
        job per setting) -- into *folder*, and capture them.

        Raises:
            RuntimeError: no worker queue, or the worker refused a layer.
        """
        layers = self.match.layers
        base = {k: v for k, v in layers.stamp.items() if k in ("existing", "materials")}
        groups: Dict[str, List[str]] = {}
        for name in names:
            settings = layers.info(name).get("settings") or {}
            groups.setdefault(json.dumps(settings, sort_keys=True), []).append(name)
        baked: Dict[str, Any] = {}
        for index, (settings, group) in enumerate(sorted(groups.items())):
            output = os.path.join(folder, str(index))
            job = {
                "layers": {n: self.payload(layers.info(n).get("spec") or {}) for n in group},
                "output": output,
                "base": base,
                "settings": json.loads(settings),
            }
            baked.update(self._bake_job(job, timeout).get("layers") or {})
            self.capture(output)
        return {"layers": baked}

    def capture(self, folder: str) -> Dict[str, Any]:
        """Turn *folder*'s bakes into layers in the capture service, and reload."""
        reply = self.client.invoke("capture", timeout=self.TIMEOUT, source=folder)
        self.client.invoke("reload", timeout=self.TIMEOUT)
        self.match = LookMatch.open(self.session)
        return reply

    # ------------------------------------------------------------------ a round
    def iterate(
        self,
        rig_path: str,
        out: str,
        fit: bool = True,
        exact: bool = True,
        max_iter: int = 20,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """One round: bake what changed, capture it, fit the free roles, judge.

        Writes into *out*: ``mix.json`` and ``rig.json`` (the rig at the fitted
        weights -- the next round's start), ``sheet.png`` (the prediction),
        ``exact.png`` and ``sheet_exact.png`` (the viewer's own frame, judged),
        ``compare.png`` (ours beside the reference, whole and close up: what a
        round is judged by, by eye), ``report.json``.

        Returns:
            The report: per stage seconds, what was baked, the fit's cost, the
            worst regions of the exact frame.
        """
        os.makedirs(out, exist_ok=True)
        rig = self.load_rig(rig_path)
        report: Dict[str, Any] = {"seconds": {}}
        clock = time.time()

        stale = self.stale(rig)
        report["baked"] = stale
        if stale:
            folder = os.path.join(os.path.abspath(out), "bakes")
            self.bake(rig, stale, folder, timeout)
            report["seconds"]["bake"] = round(time.time() - clock, 1)
            clock = time.time()
            self.capture(folder)
            # The state these layers were baked under, for the rig they make
            # (LookMatch.to_rig sets it again wherever the rig is applied).
            base = rig.get("base") or {}
            recorded = {k: base[k] for k in ("existing", "materials") if base.get(k)}
            stamp = self.match.layers.stamp
            changed = sorted(
                k for k, v in recorded.items() if stamp.get(k) not in (None, v)
            )
            if changed:  # the store now holds layers baked under two states
                report["base_changed"] = changed
            if recorded:
                self.match.layers.set_stamp(**recorded)
            report["seconds"]["capture"] = round(time.time() - clock, 1)
            clock = time.time()

        mix = self.mix_of(rig)
        free = [n for n, role in rig["roles"].items() if role.get("free")]
        kelvin = [n for n, role in rig["roles"].items() if role.get("fit_kelvin")]
        if fit and (free or kelvin):
            options = dict(rig.get("fit") or {})
            size = options.pop("size", None)  # a working size: fits are scored often
            result = self.match.fit(
                free,
                mix,
                size=tuple(size) if size else None,
                max_iter=max_iter,
                kelvin=kelvin,
                **options,
            )
            mix = result.mix
            report["fit"] = {
                "cost": [round(result.cost_before, 4), round(result.cost_after, 4)],
                "weights": {
                    n: round(after, 5) for n, (_, after) in result.changes.items()
                },
                "kelvins": {n: after for n, (_, after) in result.kelvins.items()},
            }
            report["seconds"]["fit"] = round(time.time() - clock, 1)
            clock = time.time()
        mix.save(os.path.join(out, "mix.json"))
        fitted = json.loads(json.dumps(rig))
        for name in fitted["roles"]:
            fitted["roles"][name]["weight"] = mix.weight(name)
            if mix.kelvin(name):
                fitted["roles"][name]["kelvin"] = mix.kelvin(name)
        FileUtils.write_json(os.path.join(out, "rig.json"), fitted, indent=1)
        self.match.sheet(os.path.join(out, "sheet.png"), mix=mix)
        if (
            not exact
        ):  # the image to judge by: the prediction when no exact frame is shot
            LookSheet().pair(
                self.match.predict(mix, self.match.layers.size),
                self.match.reference(),
                title=rig.get("notes", ""),
            ).save(os.path.join(out, "compare.png"))
        report["predicted"] = self._worst(self.match.score(mix))
        report["seconds"]["sheet"] = round(time.time() - clock, 1)
        clock = time.time()

        if exact:
            frame = os.path.abspath(os.path.join(out, "exact.png"))
            self.client.invoke(
                "exact", timeout=self.TIMEOUT, mix=mix.to_dict(), mode="lit", out=frame
            )
            self.match.sheet(os.path.join(out, "sheet_exact.png"), mix=mix, frame=frame)
            LookSheet().pair(
                LookTarget.display_at(frame),
                self.match.reference(),
                title=rig.get("notes", ""),
            ).save(os.path.join(out, "compare.png"))
            report["exact"] = self._worst(self.match.score(mix, frame=frame))
            report["seconds"]["exact"] = round(time.time() - clock, 1)
        FileUtils.write_json(os.path.join(out, "report.json"), report, indent=1)
        return report

    @staticmethod
    def _worst(score, count: int = 8) -> List[List[Any]]:
        """The rows furthest from the reference: ``[region, measure, score]``."""
        rows = sorted(score.rows(), key=lambda r: -abs(float(r["score"])))
        return [
            [r["region"], r["measure"], round(float(r["score"]), 3)]
            for r in rows[:count]
        ]

    # ------------------------------------------------------------------ command line
    @classmethod
    def cli(cls, argv: Optional[Sequence[str]] = None) -> int:
        """The command line (``python -m pythontk.core_utils.engines.lighting.look_match.loop``)."""
        return CLI.dispatch(
            "Look loop: one round of look matching from a rig by role.",
            {
                "iterate": CLI.Command(
                    "bake what changed, capture, fit, write the mix, sheet and exact frame",
                    cls._cli_iterate,
                    [
                        ("--out", {"required": True}),
                        ("--no-fit", {"action": "store_true"}),
                        ("--no-exact", {"action": "store_true"}),
                        ("--max-iter", {"type": int, "default": 20}),
                    ],
                ),
                "status": CLI.Command(
                    "the worker, the service and which roles would bake",
                    cls._cli_status,
                ),
            },
            argv,
            common=[
                ("session", {}),
                ("--rig", {"required": True}),
                ("--queue", {"help": "the bake worker's queue folder"}),
                ("--service", {"help": "the capture service's URL"}),
            ],
        )

    @classmethod
    def _cli_status(cls, args) -> None:
        loop = cls(args.session, args.queue, args.service)
        report = {
            "worker": loop.queue.worker() if loop.queue else None,
            "would_bake": loop.stale(cls.load_rig(args.rig)),
            "service": loop.service,
        }
        print(json.dumps(report, indent=1))

    @classmethod
    def _cli_iterate(cls, args) -> None:
        report = cls(args.session, args.queue, args.service).iterate(
            args.rig,
            args.out,
            fit=not args.no_fit,
            exact=not args.no_exact,
            max_iter=args.max_iter,
        )
        print(json.dumps(report, indent=1))


if __name__ == "__main__":
    import sys

    sys.exit(LookLoop.cli())
