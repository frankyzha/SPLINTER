"""ShapeSPLINTER: nonlinear (grouped) multiway splits."""

from .contiguous import _SPLINTER


class _ShapeSPLINTER(_SPLINTER):
    """SPLINTER with nonlinear (grouped) splits ordered by a LightGBM reference ensemble.

    Same parameters as ``_SPLINTER``; the class attribute
    ``shape_mode = True`` switches the solver to grouped splits.
    """

    shape_mode = True
