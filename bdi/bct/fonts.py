"""Font management for BCT plots."""

try:
    from mpl_fontkit import install as _fk_install
except ImportError:
    _fk_install = None


def ensure_fonts() -> None:
    """Ensure Roboto Condensed is available to matplotlib's font manager.

    Uses mpl_fontkit to download from Google Fonts if not already installed.
    Safe to call multiple times; skips download if the font is cached.
    """
    if _fk_install is not None:
        try:
            _fk_install("Roboto Condensed", verbose=False)
        except Exception:
            pass  # Network issues, etc. — fall back to matplotlib defaults
