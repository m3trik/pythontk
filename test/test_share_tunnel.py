# !/usr/bin/python
# coding=utf-8
"""Tests for :class:`pythontk.ShareTunnel`.

The provider CLIs are faked by a small Python script that prints what the real
ones print -- cloudflared's lines were captured from 2026.9.1 and Tailscale's
"not enabled" prompt from 1.102.2, on 2026-09-23, with addresses and ids
replaced -- so the link detection, the waits, the failure paths and the alias
run against real output shapes with no network and no account. Where the child
lives and dies is covered with the primitive that owns it
(``test_app_launcher``); a live run through real providers is a manual check
(``docs/webxr_preview.md``, *Sharing a link*).
"""

import os
import subprocess
import sys
import time
import unittest
import unittest.mock
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pythontk.core_utils.app_launcher import AppLauncher
from pythontk.file_utils.temp_artifacts import TempArtifacts
from pythontk.net_utils.share_tunnel import ShareTunnel

#: The link the cloudflared fixture announces.
LINK = "https://quiet-river-sample-test.trycloudflare.com"

#: cloudflared 2026.9.1's quick-tunnel start, as captured (addresses and ids
#: replaced). Two decoys lead: the banner names developers.cloudflare.com,
#: and a failed request names api.trycloudflare.com -- neither is the link.
CLOUDFLARED = [
    "2026-09-23T22:33:03Z INF Thank you for trying Cloudflare Tunnel. Doing so, "
    "without a Cloudflare account, is a quick way to experiment and try it out. "
    "If you intend to use Tunnels in production you should use a pre-created "
    "named tunnel by following: "
    "https://developers.cloudflare.com/cloudflare-one/connections/connect-apps",
    "2026-09-23T22:33:03Z INF Requesting new quick Tunnel on trycloudflare.com...",
    "2026-09-23T22:33:07Z INF +----------------------------------------------+",
    "2026-09-23T22:33:07Z INF |  Your quick Tunnel has been created! Visit it at "
    "(it may take some time to be reachable):  |",
    f"2026-09-23T22:33:07Z INF |  {LINK}                                |",
    "2026-09-23T22:33:07Z INF +----------------------------------------------+",
    "2026-09-23T22:33:07Z INF Starting metrics server on 127.0.0.1:20241/metrics",
]
REGISTERED = (
    "2026-09-23T22:33:07Z INF Registered tunnel connection connIndex=0 "
    "connection=00000000-0000-0000-0000-000000000000 event=0 ip=192.0.2.10 "
    "location=test01 protocol=quic"
)
#: The same start on a machine whose firewall denies outbound by default.
BLOCKED = [
    CLOUDFLARED[0],
    CLOUDFLARED[1],
    'failed to request quick Tunnel: Post "https://api.trycloudflare.com/tunnel": '
    "dial tcp 192.0.2.20:443: connectex: An attempt was made to access a socket "
    "in a way forbidden by its access permissions.",
]
#: Tailscale 1.102.2, `serve` on a tailnet that has not enabled it: printed,
#: then waited on until someone follows the link.
TAILSCALE_DISABLED = [
    "",
    "Serve is not enabled on your tailnet.",
    "To enable, visit:",
    "",
    "         https://login.tailscale.com/f/serve?node=nTESTNODE0000CNTRL",
]
#: Tailscale's foreground success, in its documented shape (not captured: the
#: tailnet it was measured on had not enabled Serve).
TAILSCALE_OK = [
    "Available on the internet:",
    "",
    "https://studio-pc.tail0000.ts.net/",
    "|-- proxy http://127.0.0.1:8118",
    "",
    "Press Ctrl+C to exit.",
]

#: The fake CLI: prints the fixture its first argument names, then behaves as
#: the real client would -- keeps running, or exits.
FAKE_CLI = f"""
import os, sys, time
FIXTURES = {{
    "cloudflared": {CLOUDFLARED!r},
    "blocked": {BLOCKED!r},
    "tailscale-disabled": {TAILSCALE_DISABLED!r},
    "tailscale": {TAILSCALE_OK!r},
}}
mode = sys.argv[1]
def say(lines):
    for line in lines:
        print(line, flush=True)
if mode == "cloudflared":
    say(FIXTURES["cloudflared"])
    time.sleep(0.3)
    say([{REGISTERED!r}])
elif mode == "no-ready":
    say(FIXTURES["cloudflared"])
elif mode == "blocked":
    say(FIXTURES["blocked"])
    sys.exit(1)
elif mode == "exit":
    print("Logged out.", flush=True)
    sys.exit(3)
elif mode == "tailscale-later":
    # Waits on the user the way Tailscale does, and carries on by itself once
    # the step is taken -- here, once the file named next exists.
    say(FIXTURES["tailscale-disabled"])
    while not os.path.exists(sys.argv[2]):
        time.sleep(0.05)
    say(["Success."] + FIXTURES["tailscale"])
elif mode == "tailscale-gives-up":
    # A client that names the step and exits rather than wait on it.
    say(FIXTURES["tailscale-disabled"])
    sys.exit(1)
elif mode in FIXTURES:
    say(FIXTURES[mode])
time.sleep(600)
"""


class ShareTunnelTestCase(unittest.TestCase):
    """A fake provider driven through the real start / wait / stop path."""

    @classmethod
    def setUpClass(cls):
        cls.temp = TempArtifacts("test_share_tunnel", policy="scoped")
        cls.cli = cls.temp.path(extension=".py")
        with open(cls.cli, "w", encoding="utf-8") as handle:
            handle.write(FAKE_CLI)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        self.tunnels = []
        self.spawned = []
        real_spawn = AppLauncher.spawn

        def recording_spawn(*args, **kwargs):
            process = real_spawn(*args, **kwargs)
            self.spawned.append(process)
            return process

        patcher = unittest.mock.patch.object(
            AppLauncher, "spawn", side_effect=recording_spawn
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        for tunnel in self.tunnels:
            tunnel.stop()
        for process in self.spawned:
            if process.poll() is None:  # a test that failed mid-way
                process.kill()
            process.wait(timeout=10)

    # -- helpers --------------------------------------------------------
    def _provider(self, mode, like="cloudflared", **overrides):
        """A provider named *mode* whose CLI is the fake, shaped like *like*."""
        spec = dict(ShareTunnel.PROVIDERS[like])
        spec.pop("managed", None)
        spec.update(
            executable=sys.executable,
            paths=(),
            args=["-u", self.cli, mode, "{host}", "{port}"],
            # Not by default: the fixture's link is not in public DNS, and the
            # wait on it has its own test.
            propagates=False,
        )
        spec.update(overrides)
        patcher = unittest.mock.patch.dict(ShareTunnel.PROVIDERS, {mode: spec})
        patcher.start()
        self.addCleanup(patcher.stop)
        return mode

    def _tunnel(self, mode, like="cloudflared", timeout=20.0, **kwargs):
        tunnel = ShareTunnel(
            8118, provider=self._provider(mode, like), timeout=timeout, **kwargs
        )
        self.tunnels.append(tunnel)
        return tunnel

    # -- the link -------------------------------------------------------
    def test_the_link_is_the_providers_not_the_first_url_printed(self):
        tunnel = self._tunnel("cloudflared")
        self.assertEqual(tunnel.start(), LINK)
        self.assertTrue(tunnel.is_running)
        self.assertEqual(tunnel.url, LINK)
        self.assertEqual(tunnel.netloc, "quiet-river-sample-test.trycloudflare.com")
        self.assertTrue(self.spawned[0].bound_to_parent or os.name != "nt")

    def test_the_patterns_find_the_link_and_only_the_link(self):
        """Against every captured line: exactly the link line matches, for
        each provider, and the enable prompt is what the action pattern sees."""
        import re

        cloudflared = re.compile(ShareTunnel.PROVIDERS["cloudflared"]["url"], re.I)
        hits = [line for line in CLOUDFLARED + BLOCKED if cloudflared.search(line)]
        self.assertEqual(len(hits), 1)
        self.assertEqual(cloudflared.search(hits[0]).group(1), LINK)

        for name in ("tailscale_funnel", "tailscale_serve"):
            spec = ShareTunnel.PROVIDERS[name]
            link = re.compile(spec["url"], re.I)
            found = [link.search(line) for line in TAILSCALE_OK + TAILSCALE_DISABLED]
            found = [match.group(1) for match in found if match]
            self.assertEqual(found, ["https://studio-pc.tail0000.ts.net"], name)
            action = re.compile(spec["action"], re.I)
            prompts = [line for line in TAILSCALE_DISABLED if action.search(line)]
            self.assertEqual(len(prompts), 1, name)

    def test_start_waits_for_the_connection_not_just_the_link(self):
        """cloudflared prints the link before the edge accepts a connection;
        a guest following it that early gets an error page."""
        tunnel = self._tunnel("no-ready", timeout=1.5)
        with self.assertRaises(TimeoutError) as caught:
            tunnel.start()
        self.assertIn("no connection", str(caught.exception))
        self.assertFalse(tunnel.is_running)
        self.assertIsNotNone(self.spawned[0].poll(), "the client was left running")

    def test_a_client_that_exits_fails_at_once_with_what_it_printed(self):
        tunnel = self._tunnel("exit", like="tailscale_funnel", timeout=30)
        started = time.monotonic()
        with self.assertRaises(RuntimeError) as caught:
            tunnel.start()
        self.assertLess(time.monotonic() - started, 10, "waited out the timeout")
        self.assertIn("exited (code 3)", str(caught.exception))
        self.assertIn("Logged out.", str(caught.exception))

    def test_a_provider_waiting_on_the_user_fails_at_once_naming_the_step(self):
        """Tailscale prints an enable link and polls until someone follows it;
        whoever pressed Share sees a busy indicator, not that link. It fails at
        once as a StepRequired carrying the page, so a panel can offer it --
        still a RuntimeError, for a caller catching those."""
        tunnel = self._tunnel("tailscale-disabled", like="tailscale_serve", timeout=30)
        started = time.monotonic()
        with self.assertRaises(ShareTunnel.StepRequired) as caught:
            tunnel.start()
        self.assertLess(time.monotonic() - started, 10, "waited out the timeout")
        step = caught.exception
        self.assertIsInstance(step, RuntimeError)
        self.assertEqual(
            step.url, "https://login.tailscale.com/f/serve?node=nTESTNODE0000CNTRL"
        )
        self.assertEqual(step.label, ShareTunnel.PROVIDERS["tailscale_serve"]["label"])
        self.assertIn("one-time step", str(step))
        self.assertIn(step.url, str(step))
        self.assertIsNotNone(self.spawned[0].poll(), "the client was left running")

    def _step_tunnel(self, flag, **kwargs):
        """A Tailscale-shaped start that waits on its step until *flag* exists."""
        name = self._provider(
            "tailscale-later",
            like="tailscale_funnel",
            args=["-u", self.cli, "tailscale-later", flag],
        )
        tunnel = ShareTunnel(8118, provider=name, **kwargs)
        self.tunnels.append(tunnel)
        return tunnel

    def test_a_start_told_of_the_step_waits_for_it_and_returns_the_link(self):
        """The provider carries on by itself once the step is taken, so a
        caller that can show the page (``on_step``) keeps the start waiting --
        past the link's own timeout, since the step is a person's -- and the
        link arrives without the user pressing anything again."""
        import threading

        flag = self.temp.path(extension=".step")
        steps = []
        tunnel = self._step_tunnel(flag, timeout=1.0, on_step=steps.append)

        def take_the_step():
            deadline = time.monotonic() + 20
            while not steps and time.monotonic() < deadline:
                time.sleep(0.05)
            time.sleep(1.5)  # longer than the link's own timeout
            Path(flag).touch()

        taker = threading.Thread(target=take_the_step, daemon=True)
        taker.start()
        self.assertEqual(tunnel.start(), "https://studio-pc.tail0000.ts.net")
        taker.join(20)
        self.assertEqual(len(steps), 1, "the step was reported more than once")
        step = steps[0]
        self.assertIsInstance(step, ShareTunnel.StepRequired)
        self.assertEqual(
            step.url, "https://login.tailscale.com/f/serve?node=nTESTNODE0000CNTRL"
        )
        self.assertEqual(step.label, ShareTunnel.PROVIDERS["tailscale_funnel"]["label"])
        self.assertTrue(tunnel.is_running)

    def test_a_step_not_taken_in_time_ends_the_wait_naming_it(self):
        flag = self.temp.path(extension=".step")  # never created
        steps = []
        tunnel = self._step_tunnel(flag, timeout=1.0, on_step=steps.append)
        with unittest.mock.patch.object(ShareTunnel, "_STEP_WAIT", 1.0):
            with self.assertRaises(ShareTunnel.StepRequired) as caught:
                tunnel.start()
        self.assertEqual([step.url for step in steps], [caught.exception.url])
        self.assertIn(caught.exception.url, str(caught.exception))
        self.assertIsNotNone(self.spawned[0].poll(), "the client was left running")

    def test_a_client_that_stops_at_the_step_names_the_step_not_its_exit(self):
        """A client that will not wait on the user names the step and exits:
        the step is the reason, not the exit code."""
        steps = []
        tunnel = self._tunnel(
            "tailscale-gives-up", like="tailscale_funnel", on_step=steps.append
        )
        with self.assertRaises(ShareTunnel.StepRequired) as caught:
            tunnel.start()
        self.assertEqual(
            caught.exception.url,
            "https://login.tailscale.com/f/serve?node=nTESTNODE0000CNTRL",
        )

    def test_stop_cancels_a_start_waiting_on_the_step(self):
        """A share turned off while it waits on the user ends now, not when the
        wait runs out -- or its client would bring the link up later, for
        whoever took the step, in front of nothing."""
        import threading

        flag = self.temp.path(extension=".step")
        steps, outcome = [], {}
        tunnel = self._step_tunnel(flag, timeout=1.0, on_step=steps.append)

        def start():
            try:
                outcome["url"] = tunnel.start()
            except BaseException as error:  # noqa: BLE001 -- asserted below
                outcome["error"] = error

        starter = threading.Thread(target=start, daemon=True)
        starter.start()
        deadline = time.monotonic() + 20
        while not steps and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertTrue(steps, "the start never reported its step")
        stopped = time.monotonic()
        tunnel.stop()
        starter.join(10)
        self.assertLess(time.monotonic() - stopped, 5, "the stop waited on the step")
        self.assertIsInstance(outcome.get("error"), RuntimeError, outcome)
        self.assertNotIsInstance(outcome["error"], ShareTunnel.StepRequired)
        self.assertIsNotNone(self.spawned[0].poll(), "the client was left running")
        self.assertFalse(tunnel.is_running)

    def test_an_on_step_that_stops_its_own_tunnel_ends_the_start(self):
        """The hooks run under the tunnel's re-entrant lock: one that stops the
        tunnel it was handed must end the start, not leave it polling a
        client that stop just took away."""
        flag = self.temp.path(extension=".step")  # never created
        tunnel = self._step_tunnel(
            flag, timeout=1.0, on_step=lambda step: tunnel.stop()
        )
        with self.assertRaises(RuntimeError) as caught:
            tunnel.start()
        self.assertIn("stopped", str(caught.exception))
        self.assertIsNotNone(self.spawned[0].poll(), "the client was left running")

    def test_an_on_step_that_raises_fails_the_start_and_stops_the_client(self):
        def refuse(step):
            raise ValueError("cannot show the page")

        tunnel = self._tunnel(
            "tailscale-disabled", like="tailscale_serve", on_step=refuse
        )
        with self.assertRaises(ValueError):
            tunnel.start()
        self.assertIsNotNone(self.spawned[0].poll(), "the client was left running")

    def test_a_blocked_client_is_diagnosed_as_the_firewall(self):
        """The raw line is a Go socket error that names neither the cause nor
        the program; the message leads with both."""
        tunnel = self._tunnel("blocked", timeout=30)
        with self.assertRaises(RuntimeError) as caught:
            tunnel.start()
        first = str(caught.exception).splitlines()[0]
        self.assertIn("firewall blocked", first)
        self.assertIn(sys.executable, first)

    def test_no_output_times_out_and_stops_the_client(self):
        tunnel = self._tunnel("silent", timeout=1.0)
        with self.assertRaises(TimeoutError) as caught:
            tunnel.start()
        self.assertIn("(it printed nothing)", str(caught.exception))
        self.assertIsNotNone(self.spawned[0].poll(), "the client was left running")

    def test_a_missing_cli_raises_naming_the_install(self):
        name = self._provider("absent", executable="no-such-tunnel-cli-xyz")
        tunnel = ShareTunnel(8118, provider=name)
        with self.assertRaises(FileNotFoundError) as caught:
            tunnel.start()
        self.assertIn(
            ShareTunnel.PROVIDERS["cloudflared"]["install"], str(caught.exception)
        )
        self.assertEqual(self.spawned, [])

    def test_a_new_link_is_held_until_public_dns_answers(self):
        """A guest who opens a brand-new link too early caches the miss; the
        link is announced once a public resolver has it -- and not waited on
        at all when no resolver can be asked."""
        heard = []
        tunnel = self._tunnel("cloudflared", on_url=heard.append)
        ShareTunnel.PROVIDERS["cloudflared"]["propagates"] = True
        answers = [False, False, True]
        with unittest.mock.patch(
            "pythontk.net_utils._net_utils.NetUtils.resolves_publicly",
            side_effect=lambda host: (heard.append(("dns", host)), answers.pop(0))[1],
        ):
            self.assertEqual(tunnel.start(), LINK)
        host = "quiet-river-sample-test.trycloudflare.com"
        self.assertEqual(heard, [("dns", host)] * 3 + [LINK])

        tunnel.stop()
        with unittest.mock.patch(
            "pythontk.net_utils._net_utils.NetUtils.resolves_publicly",
            return_value=None,
        ) as unaskable:
            self.assertEqual(tunnel.start(), LINK)
        self.assertEqual(unaskable.call_count, 1)

    # -- lifecycle ------------------------------------------------------
    def test_stop_ends_the_client_and_is_idempotent(self):
        tunnel = self._tunnel("cloudflared")
        tunnel.start()
        process = self.spawned[0]
        tunnel.stop()
        self.assertIsNotNone(process.poll())
        self.assertIsNone(tunnel.url)
        self.assertFalse(tunnel.is_running)
        tunnel.stop()

    def test_start_is_idempotent_while_running(self):
        tunnel = self._tunnel("cloudflared")
        self.assertEqual(tunnel.start(), tunnel.start())
        self.assertEqual(len(self.spawned), 1)

    def test_a_client_that_dies_reads_as_not_running(self):
        """How a dropped share shows: no link, rather than a dead one."""
        tunnel = self._tunnel("cloudflared")
        tunnel.start()
        self.spawned[0].kill()
        self.spawned[0].wait(timeout=10)
        self.assertFalse(tunnel.is_running)
        self.assertIsNone(tunnel.url)
        self.assertIsNone(tunnel.netloc)

    def test_the_output_is_kept_for_diagnosis(self):
        tunnel = self._tunnel("cloudflared")
        tunnel.start()
        self.assertIn(REGISTERED, tunnel.output())

    # -- announcing the link --------------------------------------------
    def test_on_url_runs_before_the_alias_and_both_hear_the_stop(self):
        """A server admits the name a guest arrives under in on_url, so it
        must run before the alias sends anyone there."""
        order = []
        tunnel = self._tunnel(
            "cloudflared",
            on_url=lambda url: order.append(("on_url", url)),
            alias=lambda url: order.append(("alias", url)),
        )
        tunnel.start()
        tunnel.stop()
        self.assertEqual(
            order,
            [("on_url", LINK), ("alias", LINK), ("on_url", None), ("alias", None)],
        )

    def test_a_stop_while_the_link_is_announced_leaves_the_alias_saying_it_ended(self):
        """stop() from another thread can land while start() is still writing
        the alias for the link it just got -- an scp takes seconds. The
        announce ran outside the lock, so the stop's "ended" could land first
        and the start's write then pointed the alias at the link that stop had
        just killed: a guest following it met a dead tunnel until the next
        share."""
        import threading

        heard = []
        writing, release = threading.Event(), threading.Event()

        def alias(url):
            if url is not None:  # the slow write of the link
                writing.set()
                release.wait(20)
            heard.append(url)

        tunnel = self._tunnel("cloudflared", alias=alias)
        starter = threading.Thread(target=tunnel.start)
        starter.start()
        self.assertTrue(writing.wait(20), "the start never announced its link")
        stopper = threading.Thread(target=tunnel.stop)
        stopper.start()
        stopper.join(0.5)  # unserialized, the stop runs to its end right here
        release.set()
        starter.join(20)
        stopper.join(20)
        self.assertFalse(tunnel.is_running)
        self.assertEqual(heard[-1:], [None], heard)

    def test_a_hook_that_stops_its_own_tunnel_leaves_the_alias_saying_it_ended(self):
        """The hooks run under the tunnel's lock, re-entrant for this: a hook
        that stops its own tunnel must neither deadlock the start nor have it
        re-announce the link that stop just killed."""
        import threading

        heard = []
        tunnel = ShareTunnel(
            8118,
            provider=self._provider("cloudflared"),
            timeout=20.0,
            on_url=lambda url: url and tunnel.stop(),
            alias=heard.append,
        )
        # Not in self.tunnels: had the start deadlocked, teardown's stop would
        # wait on it forever. The spawned client is killed there regardless.
        starter = threading.Thread(target=tunnel.start, daemon=True)
        starter.start()
        starter.join(20)
        self.assertFalse(starter.is_alive(), "the start deadlocked on its own hook")
        self.assertFalse(tunnel.is_running)
        self.assertEqual(heard, [None])

    def test_an_on_url_that_raises_fails_the_start_and_stops_the_client(self):
        def refuse(url):
            raise ValueError("cannot admit")

        tunnel = self._tunnel("cloudflared", on_url=refuse)
        with self.assertRaises(ValueError):
            tunnel.start()
        self.assertFalse(tunnel.is_running)
        self.assertIsNotNone(self.spawned[0].poll(), "the client was left running")

    def test_an_alias_failure_is_reported_not_raised(self):
        """The share works without its alias: a web root that is offline
        must not cost the link."""

        def offline(url):
            raise OSError("web root offline")

        tunnel = self._tunnel("cloudflared", alias=offline)
        self.assertEqual(tunnel.start(), LINK)
        self.assertIn("web root offline", tunnel.alias_error)

    def test_a_folder_alias_redirects_then_says_the_share_ended(self):
        folder = Path(self.temp.dir_path())
        tunnel = self._tunnel("cloudflared", alias=str(folder))
        tunnel.start()
        page = (folder / "index.html").read_text(encoding="utf-8")
        self.assertIn(f'<meta http-equiv="refresh" content="0; url={LINK}">', page)
        self.assertIn(f'location.replace("{LINK}")', page)

        tunnel.stop()
        page = (folder / "index.html").read_text(encoding="utf-8")
        self.assertNotIn(LINK, page)
        self.assertIn("Nothing is being shared right now", page)
        # It checks back, so a guest who opened it early lands on the next
        # share by itself -- with a fresh query, never a cached copy.
        self.assertIn("location.replace(location.pathname + '?t=' + Date.now())", page)
        self.assertEqual(list(folder.glob("*.part")), [])

    def test_a_missing_alias_folder_is_the_alias_error(self):
        target = Path(self.temp.dir_path()) / "no-such-folder" / "vr.html"
        tunnel = self._tunnel("cloudflared", alias=str(target))
        tunnel.start()
        self.assertIn("does not exist", tunnel.alias_error)
        self.assertFalse(target.parent.exists(), "an alias must not invent folders")

    def test_a_remote_alias_goes_through_scp_and_never_prompts(self):
        calls = []

        def fake_run(app, args, **kwargs):
            with open(args[-2], encoding="utf-8") as handle:
                calls.append((app, list(args), handle.read(), kwargs))
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

        tunnel = self._tunnel(
            "cloudflared", alias="me@web.example:/srv/www/vr/index.html"
        )
        with unittest.mock.patch.object(AppLauncher, "run", side_effect=fake_run):
            tunnel.start()
            # Inside the patch: the stop announces too, and a real scp there
            # would reach for a real host.
            tunnel.stop()
        (app, args, page, kwargs), (_app, _args, ended, _kwargs) = calls
        self.assertIn("Nothing is being shared", ended)
        self.assertEqual(app, "scp")
        self.assertIn("-B", args)  # batch mode: a missing key fails, never prompts
        self.assertEqual(args[-1], "me@web.example:/srv/www/vr/index.html")
        self.assertIn(LINK, page)
        self.assertTrue(kwargs.get("timeout"))
        self.assertFalse(os.path.exists(args[-2]), "the staged page was left behind")

    def test_a_failed_scp_is_the_alias_error(self):
        tunnel = self._tunnel("cloudflared", alias="web.example:/srv/vr.html")
        failed = subprocess.CompletedProcess(
            [], 1, stdout="", stderr="Permission denied"
        )
        with unittest.mock.patch.object(AppLauncher, "run", return_value=failed):
            self.assertEqual(tunnel.start(), LINK)
            self.assertIn("Permission denied", tunnel.alias_error)
            tunnel.stop()  # inside the patch, for the reason above

    def test_a_drive_path_is_never_read_as_a_remote_host(self):
        remote = ShareTunnel._REMOTE.match
        for local in (
            r"C:\www\vr.html",
            "C:/www/vr.html",
            r"\\nas\www\vr.html",
            "https://x.example/vr",
        ):
            self.assertIsNone(remote(local), local)
        for target in (
            "web:/srv/vr.html",
            "me@web.example:/srv/vr.html",
            "web.example:vr.html",
        ):
            self.assertIsNotNone(remote(target), target)

    def test_the_redirect_page_escapes_the_link(self):
        page = ShareTunnel._redirect_page(
            'https://x.example/?q="><script>alert(1)</script>'
        )
        self.assertNotIn("<script>alert", page)
        self.assertNotIn('"><', page)

    # -- providers ------------------------------------------------------
    def test_every_provider_declares_what_a_start_needs(self):
        import re

        for name, spec in ShareTunnel.PROVIDERS.items():
            for key in ("label", "executable", "args", "url", "public", "install"):
                self.assertIn(key, spec, name)
            args = [a.format(host="127.0.0.1", port=8118) for a in spec["args"]]
            self.assertIn("http://127.0.0.1:8118", " ".join(args), name)
            self.assertEqual(re.compile(spec["url"]).groups, 1, name)

    def test_the_default_is_always_a_public_provider(self):
        """A default has to produce a link anyone can open."""
        for name in ShareTunnel.PREFERENCE:
            self.assertTrue(ShareTunnel.PROVIDERS[name]["public"], name)

    def test_the_provider_is_named_then_configured_then_the_first_installed(self):
        installed = {"tailscale_funnel"}
        with (
            unittest.mock.patch.object(
                ShareTunnel, "executable", side_effect=lambda name: name in installed
            ),
            unittest.mock.patch.dict(os.environ, {ShareTunnel.PROVIDER_ENV: ""}),
        ):
            self.assertEqual(
                ShareTunnel.resolve_provider("tailscale_serve"), "tailscale_serve"
            )
            self.assertEqual(ShareTunnel.resolve_provider(), "tailscale_funnel")
            self.assertEqual(ShareTunnel.resolve_provider("auto"), "tailscale_funnel")
            os.environ[ShareTunnel.PROVIDER_ENV] = "tailscale_serve"
            self.assertEqual(ShareTunnel.resolve_provider(), "tailscale_serve")
            os.environ[ShareTunnel.PROVIDER_ENV] = ""
            installed.clear()
            # Nothing installed: the first preference, so the refusal that
            # follows names the easiest install.
            self.assertEqual(ShareTunnel.resolve_provider(), ShareTunnel.PREFERENCE[0])

    def test_an_unknown_provider_is_refused_by_name(self):
        with self.assertRaises(KeyError) as caught:
            ShareTunnel.resolve_provider("ngrokk")
        self.assertIn("cloudflared", str(caught.exception))

    # -- settle ---------------------------------------------------------
    def test_settle_offers_the_download_and_reports_the_install(self):
        installed, asked = [], []
        with (
            unittest.mock.patch.object(ShareTunnel, "executable", return_value=None),
            unittest.mock.patch(
                "pythontk.core_utils.app_installer.AppInstaller.ensure",
                return_value="C:/tools/cloudflared/cloudflared.exe",
            ) as ensure,
        ):
            name = ShareTunnel.settle(
                "cloudflared",
                prompt=lambda question: asked.append(question) or True,
                installed=installed.append,
            )
        self.assertEqual(name, "cloudflared")
        self.assertEqual(len(asked), 1)
        self.assertEqual(installed, ["C:/tools/cloudflared/cloudflared.exe"])
        self.assertEqual(ensure.call_args.args[0], "cloudflared")

    def test_settle_declined_refuses_with_the_fix(self):
        refused = []
        with (
            unittest.mock.patch.object(ShareTunnel, "executable", return_value=None),
            unittest.mock.patch(
                "pythontk.core_utils.app_installer.AppInstaller.ensure"
            ) as ensure,
        ):
            name = ShareTunnel.settle(
                "cloudflared", prompt=lambda question: False, refused=refused.append
            )
        self.assertIsNone(name)
        ensure.assert_not_called()
        self.assertIn("declined", refused[0])
        self.assertIn(ShareTunnel.PROVIDERS["cloudflared"]["install"], refused[0])

    def test_settle_never_installs_a_system_service(self):
        """Tailscale is a VPN service, not a tool binary: named, not fetched."""
        refused = []

        def never(question):
            raise AssertionError("asked to install a system service")

        with (
            unittest.mock.patch.object(ShareTunnel, "executable", return_value=None),
            unittest.mock.patch(
                "pythontk.core_utils.app_installer.AppInstaller.ensure"
            ) as ensure,
        ):
            name = ShareTunnel.settle(
                "tailscale_funnel", prompt=never, refused=refused.append
            )
        self.assertIsNone(name)
        ensure.assert_not_called()
        self.assertIn("https://tailscale.com/download", refused[0])

    def test_settle_refuses_a_mistyped_provider_rather_than_raising(self):
        """A bad PYTHONTK_SHARE_PROVIDER reached the panel's Share action as a
        raw KeyError; it is a refusal like any other, with the known names."""
        refused = []
        with unittest.mock.patch.dict(os.environ, {ShareTunnel.PROVIDER_ENV: "ngrok"}):
            self.assertIsNone(ShareTunnel.settle(refused=refused.append))
        self.assertIn("'ngrok'", refused[0])
        self.assertIn("cloudflared", refused[0])

    def test_settle_asks_nothing_when_the_cli_is_there(self):
        def never(question):
            raise AssertionError("asked although the CLI is installed")

        with unittest.mock.patch.object(
            ShareTunnel, "executable", return_value="C:/x.exe"
        ):
            self.assertEqual(
                ShareTunnel.settle("cloudflared", prompt=never), "cloudflared"
            )


if __name__ == "__main__":
    unittest.main()
