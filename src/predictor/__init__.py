"""
src.predictor
-------------
Standalone ML next-day BUY/SELL signal module. See docs/PREDICTOR.md.

Pipeline stages (each a separate module, each with a file contract):
    data.py      load universe parquets + benchmark (point-in-time safe)
    features.py  build the (date, symbol) feature panel from cached daily bars
    labels.py    two heads — dir1d (next-day direction) + swing (triple-barrier win)
    model.py     LightGBM + logistic baseline; train / calibrate / predict / score

Design rules (from docs/PREDICTOR.md):
  • Features at row T use only data <= close[T]; labels are strictly forward (T+1..).
  • Missing features stay NaN (LightGBM-native); never imputed to a misleading 0.
  • Every model is reproducible from the panel + config; models are data.
"""

PREDICTOR_VERSION = "0.1"
