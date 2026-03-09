# Deep Models

Public objects from `aberrant.model.deep`:

- `Autoencoder`
- `KitNET`

Notes:
- `Autoencoder` requires `torch` (`aberrant[dl]`).
- `KitNET` uses an online, phased warm-up (`feature_map_warmup`,
  `detector_warmup`, `ready`) and returns a continuous anomaly score.
