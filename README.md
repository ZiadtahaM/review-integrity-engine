# Review Integrity Engine: Automated Fraud Classification

## Executive Summary
In an ecosystem driven by consumer sentiment, review integrity is paramount. This system serves as a robust NLP pipeline designed to distinguish between authentic customer experiences and synthetic, computer-generated content (CG). By leveraging feature engineering and optimized classification algorithms, this solution provides an automated defense layer against e-commerce reputation manipulation.

## Architectural Approach
The system follows a modular architectural pattern, separating data processing, model inference, and the presentation layer.

*   **Pipeline Architecture:** The system employs a modular TF-IDF vectorization pipeline that ensures consistent feature mapping during both training and production inference, mitigating data drift.
*   **Classification Strategy:** The core engine utilizes a Logistic Regression classifier—chosen for its high interpretability, computational efficiency, and robust performance on sparse, high-dimensional textual datasets.
*   **Decoupled Presentation:** The GUI is implemented as a lightweight, event-driven interface, decoupling the core analytics engine from user interaction.

## Performance Metrics
The model achieved an **F1-Score of 0.88** on unseen test data, demonstrating high precision in distinguishing between original (OR) and computer-generated (CG) reviews. The classification process maintains sub-millisecond inference latency, making it suitable for real-time review filtering integrations.

## Repository Structure
- `train_and_save.py`: The training pipeline. Employs modular data serialization to ensure model portability and environment reproducibility.
- `app.py`: The production-ready inference interface.
- `*.pkl`: Serialized model state and feature vocabulary (feature-engineered artifacts).
# review-integrity-engine
