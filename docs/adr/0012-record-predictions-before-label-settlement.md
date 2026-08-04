# Record predictions before label settlement

A forecast can contribute to Temporal Forward Evidence only when its digest-bound Prediction Receipt was durably created before the declared label-availability instant. Label settlement is a separate append-only event that must reference that receipt after maturity. Historical replay and predictions reconstructed after targets are visible remain development evidence even if their time-series splits are otherwise leakage-safe.
