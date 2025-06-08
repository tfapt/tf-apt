##From One Attack Domain to Another: Contrastive
Transfer Learning with Siamese Networks for APT
Detection.

## Abstract

Advanced Persistent Threats (APTs) pose a major cybersecurity challenge due to their stealth, persistence, and adaptability. Traditional machine learning models struggle with APT detection due to imbalanced datasets, complex feature spaces, and limited real-world attack traces. Additionally, APT tactics vary by target, making generalizable detection models difficult to develop. Many proposed anomaly detection methods suffer from key limitations, since they lack transferability when trained in a closed-world setting, performing well on their training dataset but failing when applied to new, unseen attack scenarios. This necessitates a robust transfer learning approach to adapt knowledge across different attack scenarios.

To address these challenges, we propose a hybrid framework integrating Transfer Learning, Explainable AI (XAI), Contrastive Learning, and Siamese Networks. Our pipeline employs an attention-based autoencoder for knowledge transfer, while SHAP-based feature selection optimizes transferability by retaining only the most informative features, reducing computational overhead. Contrastive learning enhances anomaly separability, and Siamese networks align feature spaces, mitigating feature drift.

We evaluate our approach using real-world cybersecurity data from the DARPA Transparent Computing program, augmented with synthetic attack traces to simulate diverse cyber threats. By combining transfer learning, XAI-driven feature selection, contrastive learning, and synthetic data augmentation, our framework establishes a new benchmark for scalable, explainable, and transferable APT detection.

## Architecture
