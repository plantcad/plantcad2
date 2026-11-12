"""Expose pinned fitters and the original scientific wheel to CV subprocesses."""

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)
