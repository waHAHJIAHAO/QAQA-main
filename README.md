# QAQA-main
QAQA: a quick unsupervised action quality assessment method

# Skeleton-Based Action Quality Assessment with Anomaly-Aware DTW Optimization for Intelligent Sports Education

This research presents a skeleton-based action quality assessment method with anomaly-aware Dynamic Time Warping (DTW) optimization for intelligent sports education. Addressing the limitations of traditional methods—where regression models rely heavily on high-quality annotated data and unsupervised methods suffer from accuracy degradation in long sequences—we propose an indirect scoring framework integrating action anomaly detection and a Quick Action Quality Assessment (QAQA) algorithm.
![image](/asset/pipeline.png)
### Core Methods：

1. **Anomaly Detection Module**：Based on the DBSCAN clustering algorithm, this module dynamically adjusts scoring thresholds by analyzing acceleration outliers between frames, enhancing the robustness of sports action evaluations1.
2. **QAQA Algorithm**：Using a multi-resolution approach (coarsening, projection, refinement), the algorithm reduces computational complexity to O(n), addressing efficiency bottlenecks in long-sequence assessments2.
3. **Dedicated Dataset**：A novel dataset for traditional Chinese Qigong, including 22 sub-actions from Baduanjin and Yijinjing, validates the method's effectiveness.

### Experimental Results：

The method outperforms traditional approaches in both execution efficiency and scoring accuracy, with the Spearman rank correlation coefficient improving by approximately 48% compared to the best supervised methods4

### Chinese traditional Qigong Dataset：

The dataset can be obtained through Google Drive below after filling out the application form.
dataset: https://drive.google.com/drive/folders/1bPnoIrRZLBV8fPS1wo5XqhktYgJVjEy3?usp=sharing
Each movement category contains a standard template movement demonstrated by a Qigong expert and multiple movements for scoring.

| ID  | Action category | Amount |
| --- | --- | --- |
| -   | Template video | 22  |
| 1   | Double Hands Hold Up the Sky to Regulate the Triple Burner | 398 |
| 2   | Draw the Bow to the Left and Right as if Shooting an Eagle | 401 |
| 3   | Single Lift to Regulate the Spleen and Stomach | 398 |
| 4   | Look Back to Alleviate the Five Strains and Seven Injuries | 398 |
| 5   | Shake the Head and Wiggle the Tail to Eliminate Heart Fire | 42  |
| 6   | Climb the Feet with Both Hands to Strengthen the Kidneys and Lower Back | 98  |
| 7   | Clench the Fists and Gaze Fiercely to Increase Strength | 351 |
| 8   | Lift the Back to Alleviate the Seven Dizzinesses and a Hundred Ailments | 354 |
| 9   | Gathering the Energy | 354 |
| 10  | Vajrapani Presents the Pestle | 9   |
| 11  | Pluck the Stars and Shift the Big Dipper Posture | 230 |
| 12  | Pull the Tail of Nine Cows Posture | 227 |
| 13  | Extend the Claws and Spread the Wings Posture | 176 |
| 14  | Nine Ghosts Draw the Sabre | 171 |
| 15  | Three Levels Fall to the Ground Posture | 46  |
| 16  | Double Hands Hold Up the Sky to Regulate the Triple Burner-extend | 377 |
| 17  | Shake the Head and Wiggle the Tail to Eliminate Heart Fire-extend | 381 |
| 18  | Tow-extend | 344 |
| 19  | Three-extend | 345 |
| 20  | One-extend | 345 |
| 21  | Azure Dragon Extends the Claw Posture | 43  |
| 22  | Ready Position | 399 |
