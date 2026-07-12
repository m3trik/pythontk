# !/usr/bin/python
# coding=utf-8
"""DCC-agnostic formatters for material / texture info reports.

These render the *record* schema produced by a DCC's ``get_mat_info`` / ``get_texture_info``
(mayatk, blendertk) into plain-text or styled HTML. They are pure string formatting over plain
dicts — no DCC imports — so the shared report surface lives here (SSoT) and each DCC's
``MatUtils`` delegates to it rather than duplicating ~150 lines of formatting.

Record schema (all keys optional unless noted):

- **material record**: ``{"material": str, "type": str, "textures": [texture record, …]}``
- **texture record**: ``{"file_node": str, "path": str, "name": str, "size": int (bytes),
  "width": int, "height": int, "mode": str, "format": str, "bit_depth": str,
  "optimization": {"recommended": bool, "reasons": [str], "error": str}, "error": str}``
- **texture-info record** (``get_texture_info``): ``{"name", "path", "size", "width",
  "height", "mode", "format"}``
"""
import html as _html
import urllib.parse as _urlparse
from typing import Any, Dict, List


class MatReport:
    """Pure record→text/HTML formatters for material & texture info reports."""

    # ---- shared helpers ----------------------------------------------------
    @staticmethod
    def _fmt_size_auto(size_bytes) -> str:
        """Render a byte count using the largest unit that keeps the number in single/triple
        digits — GB for >=1 GB, MB for >=1 MB, KB for >=1 KB, otherwise raw bytes. Texture
        reports span six orders of magnitude (cube faces / LUTs to 4K diffuse) so a fixed unit
        always looks wrong for half the table."""
        if size_bytes is None:
            return "(unknown)"
        try:
            n = float(size_bytes)
        except (TypeError, ValueError):
            return str(size_bytes)
        if n >= 1024**3:
            return f"{n / 1024**3:,.2f} GB"
        if n >= 1024**2:
            return f"{n / 1024**2:,.2f} MB"
        if n >= 1024:
            return f"{n / 1024:,.1f} KB"
        return f"{int(n):,} bytes"

    @staticmethod
    def _path_as_link(path: str) -> str:
        """Wrap *path* in an ``<a href='file:///…'>`` anchor. Display text is the original path
        (HTML-escaped); the href is a URL-encoded ``file://`` URI so spaces, ``&``, and
        parentheses in paths survive Qt's link handler. Returns the escaped path verbatim when
        no anchor target is resolvable (empty input)."""
        if not path:
            return ""
        display = _html.escape(path)
        href_path = path.replace("\\", "/")  # forward slashes are valid on Windows file:// URLs
        href = "file:///" + _urlparse.quote(href_path.lstrip("/"), safe="/:")
        return f"<a href='{href}' style='color:#9cf; text-decoration:none;'>{display}</a>"

    # ---- texture info ------------------------------------------------------
    @classmethod
    def format_texture_info_text(cls, info_list: List[Dict[str, Any]]) -> str:
        """Render ``get_texture_info`` output as a plain-text report."""
        lines: List[str] = []
        sep = "=" * 60
        lines.append(sep)
        lines.append(f"Found {len(info_list)} valid texture(s) in scene.")
        lines.append(sep)
        for info in info_list:
            lines.append(f"Name: {info.get('name')}")
            lines.append(f"  Path:   {info.get('path')}")
            lines.append(f"  Size:   {cls._fmt_size_auto(info.get('size'))}")
            lines.append(f"  Res:    {info.get('width')}x{info.get('height')}")
            lines.append(f"  Mode:   {info.get('mode')}")
            lines.append(f"  Format: {info.get('format')}")
            lines.append("-" * 40)
        return "\n".join(lines)

    @classmethod
    def format_texture_info_html(cls, info_list: List[Dict[str, Any]]) -> str:
        """Render ``get_texture_info`` output as styled HTML. Scene-derived values (names,
        paths) are HTML-escaped — node names / file paths can legitimately contain ``& < >``."""
        head = (
            f"<h2 style='color:#9cf; margin:0 0 6px 0;'>Texture Info</h2>"
            f"<p style='color:#bbb; margin:0 0 8px 0;'>"
            f"Found <b>{len(info_list)}</b> valid texture(s) in scene.</p>"
        )
        body = _html.escape(cls.format_texture_info_text(info_list))
        return head + "<pre style='font-family:monospace; color:#ddd;'>" + body + "</pre>"

    # ---- material info -----------------------------------------------------
    @classmethod
    def format_mat_info_text(cls, records: List[Dict[str, Any]]) -> str:
        """Render ``get_mat_info`` output as a plain-text report."""
        lines: List[str] = []
        sep = "=" * 60
        lines.append(sep)
        lines.append(f"Material Info — {len(records)} material(s)")
        lines.append(sep)
        for rec in records:
            lines.append("")
            lines.append(f"[{rec.get('type')}] {rec.get('material')}")
            textures = rec.get("textures") or []
            if not textures:
                lines.append("  (no textures)")
                continue
            for t in textures:
                lines.append(f"  - {t.get('name')}  ({t.get('file_node')})")
                lines.append(f"      Path:      {t.get('path')}")
                if "error" in t:
                    lines.append(f"      Error:     {t['error']}")
                    continue
                if "width" in t or "mode" in t or "format" in t:
                    lines.append(
                        f"      Res:       {t.get('width')}x{t.get('height')}  "
                        f"Mode: {t.get('mode')}  BitDepth: {t.get('bit_depth')}  "
                        f"Format: {t.get('format')}"
                    )
                lines.append(f"      File size: {cls._fmt_size_auto(t.get('size'))}")
                opt = t.get("optimization")
                if opt is None:
                    continue
                if "error" in opt:
                    lines.append(f"      Optimize:  (error: {opt['error']})")
                elif opt.get("recommended"):
                    lines.append("      Optimize:  YES")
                    for r in opt.get("reasons", []):
                        lines.append(f"                 - {r}")
                else:
                    lines.append("      Optimize:  no change recommended")
        return "\n".join(lines)

    @classmethod
    def format_mat_info_html(cls, records: List[Dict[str, Any]]) -> str:
        """Render ``get_mat_info`` output as styled HTML. Inline colours flag optimization
        status (yellow = recommended, red = error, dim = no change). Scene-derived strings are
        HTML-escaped; paths are wrapped as ``file://`` links so the host viewer can open the
        containing folder on click."""
        esc = _html.escape
        head = (
            f"<h2 style='color:#9cf; margin:0 0 6px 0;'>Material Info</h2>"
            f"<p style='color:#bbb; margin:0 0 8px 0;'>"
            f"<b>{len(records)}</b> material(s)</p>"
        )
        chunks: List[str] = [head]
        for idx, rec in enumerate(records):
            if idx > 0:  # visual separator between materials (skip before the first)
                chunks.append(
                    "<hr style='border:none; border-top:1px solid #444; margin:10px 0 0 0;'/>"
                )
            chunks.append(
                f"<p style='margin:8px 0 2px 0;'>"
                f"<span style='color:#888;'>[{esc(str(rec.get('type', '')))}]</span> "
                f"<b style='color:#fff;'>{esc(str(rec.get('material', '')))}</b></p>"
            )
            textures = rec.get("textures") or []
            if not textures:
                chunks.append(
                    "<pre style='color:#888; margin:0 0 0 16px;'>(no textures)</pre>"
                )
                continue
            body_lines: List[str] = []
            for t in textures:
                body_lines.append(
                    f"<span style='color:#ddd;'>- {esc(str(t.get('name', '')))}</span>  "
                    f"<span style='color:#888;'>({esc(str(t.get('file_node', '')))})</span>"
                )
                body_lines.append(
                    f"    Path:      {cls._path_as_link(str(t.get('path', '')))}"
                )
                if "error" in t:
                    body_lines.append(
                        f"    <span style='color:#e58;'>Error:     {esc(str(t['error']))}</span>"
                    )
                    continue
                if "width" in t or "mode" in t or "format" in t:
                    body_lines.append(
                        f"    Res:       {t.get('width')}x{t.get('height')}  "
                        f"Mode: {esc(str(t.get('mode', '')))}  "
                        f"BitDepth: {esc(str(t.get('bit_depth', '')))}  "
                        f"Format: {esc(str(t.get('format', '')))}"
                    )
                body_lines.append(f"    File size: {cls._fmt_size_auto(t.get('size'))}")
                opt = t.get("optimization")
                if opt is None:
                    continue
                if "error" in opt:
                    body_lines.append(
                        f"    <span style='color:#e58;'>Optimize:  (error: {esc(str(opt['error']))})</span>"
                    )
                elif opt.get("recommended"):
                    body_lines.append("    <span style='color:#ec5;'>Optimize:  YES</span>")
                    for r in opt.get("reasons", []):
                        body_lines.append(
                            f"               <span style='color:#ec5;'>- {esc(str(r))}</span>"
                        )
                else:
                    body_lines.append(
                        "    <span style='color:#888;'>Optimize:  no change recommended</span>"
                    )
            chunks.append(
                "<pre style='font-family:monospace; margin:0 0 0 16px;'>"
                + "\n".join(body_lines)
                + "</pre>"
            )
        return "".join(chunks)


# --------------------------------------------------------------------------------------------
# Notes
# --------------------------------------------------------------------------------------------
