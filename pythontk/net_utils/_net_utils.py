# !/usr/bin/python
# coding=utf-8
from __future__ import annotations
import os
import socket
import ssl
import subprocess
from typing import Optional, Dict, List, Sequence, Tuple


class NetUtils:
    """
    General purpose network utilities.
    """

    @staticmethod
    def listening_ports() -> List[Tuple[int, Optional[int]]]:
        """``[(port, pid), ...]`` for every LISTENING TCP socket, IPv4 and IPv6.

        *pid* is the process holding the socket, or None where this user may
        not see it (another user's process on Linux). The question a launcher
        asks before trusting a port: "is this port held by the process I
        started?" -- a port answering a connect may belong to anyone.

        Windows parses ``netstat -ano``, keyed on the foreign address ``:0``
        (the state word is localized: a German Windows does not print
        "LISTENING"). Linux reads ``/proc/net/tcp{,6}`` and maps each socket's
        inode to the process holding it in ``/proc/<pid>/fd``.

        Raises:
            OSError: This platform's source is unavailable (or the platform is
                not supported) -- so a caller can tell "cannot enumerate" from
                "nothing listens".
            subprocess.CalledProcessError: ``netstat`` failed (Windows).
        """
        import re
        import sys

        if sys.platform == "win32":
            # netstat writes the console's OEM codepage, not the ANSI one a
            # bare text mode decodes with: a localized header (a French Windows
            # writes an "É", cp850 0x90, undefined in cp1252) raised
            # UnicodeDecodeError before a row was read. The rows keyed on below
            # are ASCII either way.
            output = subprocess.check_output(
                ["netstat", "-ano"], encoding="oem", errors="replace"
            )
            # "  TCP    0.0.0.0:7002    0.0.0.0:0    LISTENING    1234"; an
            # ESTABLISHED row's foreign port is never 0.
            row = re.compile(r"^\s*TCP\s+\S+:(\d+)\s+\S+:0\s+\S+\s+(\d+)\s*$")
            return sorted(
                {
                    (int(m.group(1)), int(m.group(2)))
                    for m in map(row.match, output.splitlines())
                    if m
                }
            )
        if not sys.platform.startswith("linux"):
            raise OSError(f"listening_ports is not supported on {sys.platform}")

        ports = {}  # socket inode -> port
        found_table = False
        for table in ("/proc/net/tcp", "/proc/net/tcp6"):
            try:
                with open(table, encoding="ascii") as f:
                    lines = f.read().splitlines()[1:]
            except FileNotFoundError:
                continue  # tcp6 is absent with IPv6 disabled
            found_table = True
            for line in lines:
                # sl local_address rem_address st tx:rx tr:when retrnsmt uid timeout inode
                fields = line.split()
                if len(fields) > 9 and fields[3] == "0A":  # TCP_LISTEN
                    ports[fields[9]] = int(fields[1].rsplit(":", 1)[1], 16)
        if not found_table:
            raise OSError("/proc/net/tcp is not readable")

        owners = {}
        for entry in os.listdir("/proc"):
            if len(owners) == len(ports):
                break
            if not entry.isdigit():
                continue
            fd_dir = f"/proc/{entry}/fd"
            try:
                fds = os.listdir(fd_dir)
            except OSError:
                continue  # gone, or another user's
            for fd in fds:
                try:
                    link = os.readlink(f"{fd_dir}/{fd}")
                except OSError:
                    continue
                if link.startswith("socket:[") and link[8:-1] in ports:
                    owners.setdefault(link[8:-1], int(entry))
        return sorted({(port, owners.get(inode)) for inode, port in ports.items()})

    @staticmethod
    def connect_rdp(
        host: str,
        username: str = None,
        password: str = None,
        width: int = None,
        height: int = None,
        fullscreen: bool = True,
        extra_settings: Dict[str, str] = None,
        save_credentials: bool = True,
    ):
        """
        Connect to a remote desktop using Windows RDP (mstsc.exe).

        Args:
            host (str): Hostname or IP address.
            username (str, optional): Username to log in with.
            password (str, optional): Password. If provided, adds to Windows Credential Manager.
            width (int, optional): Window width.
            height (int, optional): Window height.
            fullscreen (bool): Whether to launch in fullscreen. Defaults to True.
            extra_settings (dict, optional): Dictionary of additional RDP settings (e.g. {'drivestoredirect': '*'})
                                             to merge/override defaults.
            save_credentials (bool, optional): Whether to persist the provided password in Credential Manager.
                                               Defaults to True (standard RDP behavior).
        """
        if os.name != "nt":
            raise OSError("RDP connection is only supported on Windows.")

        # 1. Save credentials at the exact Windows target mstsc reads.
        if username and password and save_credentials:
            target_name = f"TERMSRV/{host}"

            # RDP creds must live at the exact Windows target mstsc reads
            # (TERMSRV/{host}). The keyring-first Credentials store files them
            # under its own service name, so mstsc never finds them and prompts
            # anyway. Write the credential directly with cmdkey (this path is
            # already Windows-only, guarded by os.name != "nt" above).
            subprocess.run(
                [
                    "cmdkey",
                    "/generic:" + target_name,
                    f"/user:{username}",
                    f"/pass:{password}",
                ],
                check=True,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )

        # 2. Build RDP configuration
        defaults = {
            "full address": f"s:{host}",
            "authentication level": "i:0",  # Connect and don't warn me (default for automated scripts)
            "prompt for credentials": "i:0",
            "administrative session": "i:0",
        }

        if username:
            defaults["username"] = f"s:{username}"

        if fullscreen:
            defaults["screen mode id"] = "i:2"
        elif width and height:
            defaults["screen mode id"] = "i:1"
            defaults["desktopwidth"] = f"i:{width}"
            defaults["desktopheight"] = f"i:{height}"

        if extra_settings:
            defaults.update(extra_settings)

        config_lines = [f"{k}:{v}" for k, v in defaults.items()]

        # 3. Create temp file and launch
        # Detached: mstsc reads the file AFTER we return, so it cannot be deleted
        # here on the success path -- and previously nothing ever did, so every
        # connection leaked a .rdp. Allocation now sweeps stale ones by age.
        from pythontk.file_utils.temp_artifacts import TempArtifacts

        rdp_path = TempArtifacts("pythontk_rdp").path(extension=".rdp")
        try:
            with open(rdp_path, "w") as f:
                f.write("\n".join(config_lines))

            subprocess.Popen(["mstsc.exe", rdp_path])
        except Exception as e:
            print(f"Failed to launch RDP: {e}")
            # Clean up immediately if fail, otherwise mstsc needs the file
            if os.path.exists(rdp_path):
                try:
                    os.remove(rdp_path)
                except OSError:
                    pass
            raise

    @staticmethod
    def is_port_open(host: str, port: int, timeout: float = 1.0) -> bool:
        """
        Check if a TCP port is open on a host.

        Args:
            host (str): Hostname or IP address.
            port (int): Port number.
            timeout (float): Connection timeout in seconds.

        Returns:
            bool: True if port is open, False otherwise.
        """
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except (socket.timeout, ConnectionRefusedError, OSError):
            return False

    @staticmethod
    def is_port_bindable(port: int, host: str = "127.0.0.1") -> bool:
        """Check whether a NEW server could bind a TCP port on this machine.

        This is a different question from :meth:`is_port_open`: a hung
        (zombie) process can hold a port *bound but not listening* -- a
        connect probe reads that as free, yet a new server's ``bind()`` still
        fails, so anything launched on that port waits forever. Use this when
        *choosing a port to launch a listener on*; use ``is_port_open`` when
        detecting an existing service to connect to.

        Args:
            port (int): Port number to test.
            host (str): Interface to bind on (default localhost).

        Returns:
            bool: True if a bind succeeded (the port is genuinely free).
        """
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if os.name != "nt":
            # Bind as the server will: HTTPServer sets SO_REUSEADDR on POSIX,
            # which binds straight over a previous run's TIME_WAIT sockets --
            # without it a quick restart reads as "taken". (It still refuses a
            # socket bound without the flag: the zombie case above.)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False
        finally:
            s.close()

    @staticmethod
    def resolves_publicly(
        name: str,
        servers: Sequence[str] = ("1.1.1.1", "8.8.8.8", "9.9.9.9"),
        timeout: float = 3.0,
        port: int = 53,
    ) -> Optional[bool]:
        """Whether public DNS answers *name* right now, asked of *servers* directly.

        Deliberately NOT through this machine's resolver. A name that does not
        exist yet -- a tunnel's brand-new hostname -- is cached as a miss when
        looked up too early, by the OS and the browser alike, and goes on
        failing for minutes after it appears. One A-record query over UDP to a
        public resolver leaves no such trace here.

        Every server is asked until one has the name, because each caches its
        own misses: polled while a name is appearing, one resolver goes on
        answering NXDOMAIN from its cache after the record exists (measured
        2026-09-23: 6 s one run, 18 s the next, asking 1.1.1.1 alone), and a
        second, asked in the same breath, has often already expired its miss.

        Parameters:
            name: The host name.
            servers: Resolver IPs.
            timeout: Seconds to wait for each.
            port: The resolvers' port.

        Returns:
            True when any resolver has an address for it; False when those
            that answered do not (NXDOMAIN, or no address); ``None`` when none
            could be asked -- offline, or outbound DNS blocked. ``None`` is no
            answer, not a "no".
        """
        addresses = NetUtils.public_addresses(name, servers, timeout, port)
        return None if addresses is None else bool(addresses)

    @staticmethod
    def public_addresses(
        name: str,
        servers: Sequence[str] = ("1.1.1.1", "8.8.8.8", "9.9.9.9"),
        timeout: float = 3.0,
        port: int = 53,
    ) -> Optional[List[str]]:
        """The IPv4 addresses public DNS gives *name*, asked of *servers* directly.

        What a stranger's machine resolves the name to -- not this machine's
        resolver, which can answer otherwise: Tailscale's MagicDNS gives a
        tailnet member its own ``100.x`` address for a ``ts.net`` name, so a
        request through it never touches the public ingress a guest uses.
        Asked as :meth:`resolves_publicly` asks (no miss cached here), every
        server in turn until one has the name.

        Parameters:
            name: The host name.
            servers: Resolver IPs.
            timeout: Seconds to wait for each.
            port: The resolvers' port.

        Returns:
            The A records of the first resolver that has any (a CNAME chain
            followed by the resolver); ``[]`` when those that answered have
            none (NXDOMAIN, or no address); ``None`` when none could be asked.
        """
        import random
        import struct

        query_id = random.randrange(1 << 16)
        # Header: id, flags (recursion desired), one question.
        packet = struct.pack(">HHHHHH", query_id, 0x0100, 1, 0, 0, 0)
        for label in name.rstrip(".").encode("idna").split(b"."):
            packet += bytes([len(label)]) + label
        packet += b"\x00" + struct.pack(">HH", 1, 1)  # type A, class IN
        answer = None
        for server in servers:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                    sock.settimeout(timeout)
                    sock.sendto(packet, (server, port))
                    reply = sock.recv(4096)
            except OSError:
                continue
            if (
                len(reply) < 12
                or struct.unpack(">H", reply[:2])[0] != query_id
                or not reply[2] & 0x80  # not a response
            ):
                continue
            rcode = reply[3] & 0x0F
            if rcode == 0:
                found = NetUtils._a_records(reply, len(packet))
                if found:
                    return found
            if rcode in (0, 3):  # no address / NXDOMAIN
                answer = []
            # SERVFAIL, REFUSED ...: this resolver has no answer either way.
        return answer

    @staticmethod
    def _a_records(reply: bytes, offset: int) -> List[str]:
        """The A records in DNS *reply*'s answer section, which starts at
        *offset* (just past the one question, echoed as it was sent).
        A malformed reply yields what was read before the fault."""
        import struct

        found: List[str] = []
        try:
            for _ in range(struct.unpack(">H", reply[6:8])[0]):
                # The owner name: labels, or a 2-byte compression pointer.
                while reply[offset] and reply[offset] & 0xC0 != 0xC0:
                    offset += reply[offset] + 1
                offset += 2 if reply[offset] & 0xC0 == 0xC0 else 1
                kind, _cls, _ttl, length = struct.unpack(
                    ">HHIH", reply[offset : offset + 10]
                )
                offset += 10
                if kind == 1 and length == 4:
                    found.append(socket.inet_ntoa(reply[offset : offset + 4]))
                offset += length
        except (IndexError, struct.error):
            pass
        return found

    @staticmethod
    def tls_context() -> ssl.SSLContext:
        """A verifying client TLS context that trusts what a browser trusts.

        :func:`ssl.create_default_context` loads the Windows certificate
        stores as they stand, expired entries included -- and OpenSSL, handed
        an expired cross-signed copy of a root beside the chain a server
        sends, picks it and fails the handshake. Measured 2026-10-15: the
        stores kept ``ISRG Root X2`` cross-signed by X1 (expired 2025-09-15),
        so every Let's Encrypt site -- a Tailscale ``ts.net`` link among them
        -- failed ``certificate has expired`` from Python while every browser
        opened it. Here the expired entries stay out; the rest of the stores,
        the ``SSL_CERT_FILE`` / ``SSL_CERT_DIR`` overrides and -- when it is
        installed -- ``certifi``'s bundle (a root the stores never received)
        are trusted. Built once per process; treat it as read-only.

        Returns:
            An :class:`ssl.SSLContext` that checks the host name and requires
            a certificate.
        """
        context = NetUtils._tls_context
        if context is None:
            context = NetUtils._tls_context = NetUtils._build_tls_context()
        return context

    _tls_context = None

    @staticmethod
    def _build_tls_context():
        import sys
        import time

        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.set_default_verify_paths()  # all load_default_certs does off Windows
        if sys.platform == "win32":
            now = time.time()
            for store in ("CA", "ROOT"):  # the order load_default_certs reads
                trusted = bytearray()
                for der, encoding, trust in ssl.enum_certificates(store):
                    usable = encoding == "x509_asn" and (
                        trust is True or "1.3.6.1.5.5.7.3.1" in trust  # serverAuth
                    )
                    if usable and not NetUtils._expired(der, now):
                        trusted.extend(der)
                if trusted:
                    context.load_verify_locations(cadata=bytes(trusted))
        try:
            import certifi

            context.load_verify_locations(cafile=certifi.where())
        except (ImportError, OSError, ssl.SSLError):
            pass
        return context

    @staticmethod
    def _expired(der: bytes, now: float) -> bool:
        """Whether CA certificate *der* is past its notAfter -- read by OpenSSL
        itself, through a throwaway context. A certificate it cannot read is
        left out too; one that is no CA (``get_ca_certs`` lists none) is kept."""

        probe = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        try:
            probe.load_verify_locations(cadata=der)
        except ssl.SSLError:
            return True
        listed = probe.get_ca_certs()
        return bool(listed) and ssl.cert_time_to_seconds(listed[0]["notAfter"]) < now

    @staticmethod
    def get_local_ip() -> Optional[str]:
        """
        Get the local IP address of this machine.
        Returns None if it fails.
        """
        try:
            # Uses a dummy connection to determine the interface IP used for routing
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]
        except Exception:
            return None
