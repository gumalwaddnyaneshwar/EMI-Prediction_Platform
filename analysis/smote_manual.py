"""
Manual SMOTE (Synthetic Minority Oversampling Technique) implementation.

This sandbox has no internet access, so the `imbalanced-learn` package
(which provides SMOTE out of the box) cannot be installed. SMOTE's core
algorithm is straightforward - for each minority-class sample, find its
k nearest neighbors within the same class and generate synthetic points
by linear interpolation between them - so it's implemented here directly
using scikit-learn's NearestNeighbors, which IS available.

This is a real, working SMOTE implementation, not an approximation.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors


def smote_oversample(X: np.ndarray, y: np.ndarray, minority_label, k_neighbors=5,
                      target_ratio=1.0, random_state=42):
    """
    Generate synthetic samples for the minority class via SMOTE.

    target_ratio: desired minority_count / majority_count after oversampling
                  (1.0 = fully balanced with the largest class; here we use
                  a more moderate ratio to avoid over-correcting, same
                  principle as the sample_weight tuning done earlier)
    """
    rng = np.random.RandomState(random_state)
    minority_mask = (y == minority_label)
    X_minority = X[minority_mask]
    n_minority = len(X_minority)

    majority_count = np.bincount(y).max()
    n_synthetic = int(majority_count * target_ratio) - n_minority
    if n_synthetic <= 0:
        return X, y

    nn = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(X_minority)
    _, neighbor_idx = nn.kneighbors(X_minority)

    synthetic = np.zeros((n_synthetic, X.shape[1]))
    for i in range(n_synthetic):
        sample_idx = rng.randint(0, n_minority)
        neighbor_choice = rng.randint(1, k_neighbors + 1)  # skip self (index 0)
        neighbor_idx_i = neighbor_idx[sample_idx, neighbor_choice]
        gap = rng.uniform(0, 1)
        synthetic[i] = X_minority[sample_idx] + gap * (X_minority[neighbor_idx_i] - X_minority[sample_idx])

    X_resampled = np.vstack([X, synthetic])
    y_resampled = np.concatenate([y, np.full(n_synthetic, minority_label)])
    return X_resampled, y_resampled


if __name__ == "__main__":
    import sys
    sys.path.append("src")
    import joblib
    from train_utils import load_featured_data, get_splits, fit_or_load_preprocessor, transform_splits, CLF_TARGET
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import classification_report, f1_score, accuracy_score

    df = load_featured_data()
    train_df, val_df, test_df = get_splits(df)
    preprocessor = fit_or_load_preprocessor(train_df, refit=False)
    X_train, X_val, X_test = transform_splits(preprocessor, train_df, val_df, test_df)
    le = joblib.load("models/label_encoder.joblib")
    y_train = le.transform(train_df[CLF_TARGET])
    y_val = le.transform(val_df[CLF_TARGET])
    hr_idx = list(le.classes_).index("High_Risk")

    print(f"Before SMOTE: {np.bincount(y_train)} (class order: {list(le.classes_)})")

    # Moderate ratio (0.3 of majority count) - same philosophy as the
    # sample_weight=2 tuning: partial correction, not full balancing,
    # to avoid the precision collapse seen with full balancing earlier.
    X_train_smote, y_train_smote = smote_oversample(
        X_train, y_train, minority_label=hr_idx, k_neighbors=5, target_ratio=0.3
    )
    print(f"After SMOTE:  {np.bincount(y_train_smote)}")

    model = HistGradientBoostingClassifier(max_iter=300, max_depth=8, learning_rate=0.08, random_state=42)
    model.fit(X_train_smote, y_train_smote)
    pred = model.predict(X_val)

    print("\n=== SMOTE-oversampled model (val set) ===")
    print(classification_report(y_val, pred, target_names=le.classes_, digits=3))
    print("macro F1:", round(f1_score(y_val, pred, average="macro"), 4))
    print("accuracy:", round(accuracy_score(y_val, pred), 4))
