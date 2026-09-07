# Freqtrade plus the ML stack the strategy needs at inference time.
# Pinned to a stable tag so a silent upstream bump cannot change backtests.
FROM freqtradeorg/freqtrade:stable

USER ftuser
COPY requirements-ml.txt /tmp/requirements-ml.txt
RUN pip install --user --no-cache-dir -r /tmp/requirements-ml.txt
