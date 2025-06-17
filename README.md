# QAQA-main
QAQA: a quick unsupervised action quality assessment method

# Skeleton-Based Action Quality Assessment with Anomaly-Aware DTW Optimization for Intelligent Sports Education

This research presents a skeleton-based action quality assessment method with anomaly-aware Dynamic Time Warping (DTW) optimization for intelligent sports education. Addressing the limitations of traditional methods—where regression models rely heavily on high-quality annotated data and unsupervised methods suffer from accuracy degradation in long sequences—we propose an indirect scoring framework integrating action anomaly detection and a Quick Action Quality Assessment (QAQA) algorithm.

### Core Methods

1. **Anomaly Detection Module**：Based on the DBSCAN clustering algorithm, this module dynamically adjusts scoring thresholds by analyzing acceleration outliers between frames, enhancing the robustness of sports action evaluations1.
2. **QAQA Algorithm**：Using a multi-resolution approach (coarsening, projection, refinement), the algorithm reduces computational complexity to O(n), addressing efficiency bottlenecks in long-sequence assessments2.
3. **Dedicated Dataset**：A novel dataset for traditional Chinese Qigong, including 22 sub-actions from Baduanjin and Yijinjing, validates the method's effectiveness.

### Experimental Results

The method outperforms traditional approaches in both execution efficiency and scoring accuracy, with the Spearman rank correlation coefficient improving by approximately 48% compared to the best supervised methods4
