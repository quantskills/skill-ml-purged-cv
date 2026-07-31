# Use an explicit validation dataset

The canonical input is a Validation Dataset containing stable sample identities, explicit Information Intervals, authoritative Session Axis membership, features, targets, and optional asset identities. Pandas and MultiIndex inputs enter through an explicit Pandas Adapter; a DataFrame-first API with column-name, index-level, or hidden-attribute inference was rejected because ambiguous metadata would undermine fail-closed leakage guarantees.
