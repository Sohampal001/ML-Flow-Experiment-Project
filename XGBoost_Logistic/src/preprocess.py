import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def cap_outliers_iqr(train_df, test_df):
    """
    Cap outliers using IQR method.
    Caps are calculated from TRAIN data and applied to both.
    """
    train_capped = train_df.copy()
    test_capped = test_df.copy()

    for col in train_df.columns:
        Q1 = train_df[col].quantile(0.25)
        Q3 = train_df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        train_capped[col] = train_capped[col].clip(lower_bound, upper_bound)
        test_capped[col] = test_capped[col].clip(lower_bound, upper_bound)

    return train_capped, test_capped


def preprocess_data(
    raw_data_path: str,
    processed_data_dir: str,
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Preprocess raw data:
    - Remove duplicates
    - Convert quality to good/bad label
    - Train-test split
    - Outlier capping (IQR)
    - Standard scaling (no leakage)
    - Save train/test CSV
    """

    
    df = pd.read_csv(raw_data_path)
    df = df.drop_duplicates()

   
    df["quality_label"] = df["quality"].apply(
        lambda x: 0 if x >= 6 else 1
    )

    
    X = df.drop(columns=["quality", "quality_label"])
    y = df["quality_label"]

   
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )


    X_train_capped, X_test_capped = cap_outliers_iqr(X_train, X_test)


    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train_capped)
    X_test_scaled = scaler.transform(X_test_capped)

    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

    # -----------------------
    # Combine features & target
    # -----------------------
    train_df = pd.concat(
        [X_train_scaled, y_train.reset_index(drop=True)],
        axis=1
    )
    test_df = pd.concat(
        [X_test_scaled, y_test.reset_index(drop=True)],
        axis=1
    )

    # -----------------------
    # Save processed data
    # -----------------------
    os.makedirs(processed_data_dir, exist_ok=True)

    train_path = os.path.join(processed_data_dir, "train.csv")
    test_path = os.path.join(processed_data_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("Preprocessing completed")
    print("Quality converted to binary label (Good=0, Bad=1)")
    print("Outliers capped using IQR")
    print("StandardScaler applied after split")
    print(f"Train saved: {train_path}")
    print(f"Test saved: {test_path}")


if __name__ == "__main__":
    preprocess_data(
        raw_data_path=r"D:\ml_flow_project\XGBoost_Logistic\data\raw\winequality-red.csv",
        processed_data_dir=r"D:\ml_flow_project\XGBoost_Logistic\data\preprocessed"
    )
