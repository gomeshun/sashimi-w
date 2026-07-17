"""Public opt-in façade for the ITAMAE-backed SASHIMI-W implementation.

Importing this module leaves the established :mod:`sashimi_w` API unchanged.
The lower-case alias mirrors its public class name while selecting the
parallel implementation with an explicit WDM power convention, independent
physics modes, and structured ITAMAE catalogs.
"""

from sashimi_w_itamae_migration import ItamaeSubhalos

subhalos = ItamaeSubhalos

__all__ = ["ItamaeSubhalos", "subhalos"]
