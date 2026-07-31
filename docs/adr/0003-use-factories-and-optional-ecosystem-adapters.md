# Use factories and optional ecosystem adapters

The Leakage-Safe Evaluator will accept estimator and transformation protocols through Fold Factories so every validation assignment receives new unfitted objects. The core runtime will depend on NumPy, while pandas and scikit-learn integration will be optional adapters; accepting arbitrary pre-fitted objects or hard-coupling the core to scikit-learn was rejected because those choices either permit hidden cross-fold state or unnecessarily constrain model ecosystems.
