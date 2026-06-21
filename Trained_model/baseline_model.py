import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ----------------------------
# Configuration
# ----------------------------

DATA_PATH = "sms_preprocessed.csv"
TEXT_COL = "clean_message"
LABEL_COL = "label_num"

RANDOM_STATE = 42
TEST_SIZE = 0.2


# ----------------------------
# Dummy dataset (fallback)
# ----------------------------

def load_dummy_data():

    data = {
        "text": [
            "Congratulations! You have won a free iPhone, click here to claim now",
            "Your bank account has been suspended, verify immediately at this link",
            "Hey, are we still meeting for lunch tomorrow?",
            "URGENT: Your package could not be delivered, update payment info",
            "Don't forget the team meeting at 5pm today",
            "You've been selected for a $1000 gift card, claim before it expires",
            "Can you send me the notes from yesterday's class?",
            "Verify your PayPal account now or it will be permanently locked",
            "Happy birthday! Hope you have a great day",
            "Click this link to reset your password immediately, urgent action required",
            "Let's catch up this weekend, it's been a while",
            "Your Netflix subscription payment failed, update billing details now",
            "Reminder: assignment submission deadline is tomorrow night",
            "You have an unclaimed refund, click to receive your money",
            "Mom asked if you're coming home for dinner tonight",
            "Security alert: unusual login detected, confirm your identity now"
        ],
        "label": [
            1, 1, 0, 1,
            0, 1, 0, 1,
            0, 1, 0, 1,
            0, 1, 0, 1
        ]
    }

    return pd.DataFrame(data)


# ----------------------------
# Load dataset
# ----------------------------

def load_data():

    # ------------------------
    # SMS DATASET
    # ------------------------
    sms_df = pd.read_csv(
        "sms_preprocessed.csv",
        encoding="latin-1"
    )

    sms_df = sms_df[[TEXT_COL, LABEL_COL]].dropna()

    sms_df.columns = ["text", "label"]

    if sms_df["label"].dtype == object:

        sms_df["label"] = (
            sms_df["label"]
            .str.lower()
            .map({
                "spam": 1,
                "phishing": 1,
                "1": 1,
                "ham": 0,
                "legitimate": 0,
                "0": 0
            })
        )

    sms_df = sms_df.dropna(subset=["label"])

    sms_df["label"] = sms_df["label"].astype(int)

    # ------------------------
    # EMAIL DATASET
    # ------------------------
    email_df = pd.read_csv(
        "preprocessed_email_dataset.csv"
    )

    email_df = email_df[["tokens", "label"]].dropna()

    email_df["tokens"] = (
        email_df["tokens"]
        .astype(str)
        .str.replace("[", "", regex=False)
        .str.replace("]", "", regex=False)
        .str.replace("'", "", regex=False)
        .str.replace(",", " ", regex=False)
    )

    email_df.columns = ["text", "label"]

    email_df["label"] = email_df["label"].astype(int)

    # ------------------------
    # COMBINE DATASETS
    # ------------------------
    df = pd.concat(
        [sms_df, email_df],
        ignore_index=True
    )

    df = df.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    print("\nCombined Dataset Shape:")
    print(df.shape)

    print("\nClass Distribution:")
    print(df["label"].value_counts())

    return df


# ----------------------------
# Main
# ----------------------------

def main():

    df = load_data()

    print("\nDataset shape:")
    print(df.shape)

    print("\nClass Distribution:")
    print(df["label"].value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["label"]
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=10000,
        ngram_range=(1, 2)
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    models = {

    "Naive Bayes": MultinomialNB(),

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
        class_weight="balanced"
    )
}

    results = []

    trained_models = {}

    for name, model in models.items():

        print(f"\n========== {name} ==========")

        model.fit(X_train_vec, y_train)

        preds = model.predict(X_test_vec)

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)

        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall   : {rec:.4f}")
        print(f"F1 Score : {f1:.4f}")

        print("Confusion Matrix:")
        print(confusion_matrix(y_test, preds))

        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1": f1
        })

        trained_models[name] = model

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="F1",
        ascending=False
    )

    print("\n===== MODEL SUMMARY =====")
    print(results_df)

    # --------------------------------
    # Save Logistic Regression
    # --------------------------------

    best_name = "Logistic Regression"

    best_model = trained_models["Logistic Regression"]

    joblib.dump(
        best_model,
        "best_baseline_model.pkl"
    )

    joblib.dump(
        vectorizer,
        "tfidf_vectorizer.pkl"
    )

    results_df.to_csv(
        "results.csv",
        index=False
    )

    print("\n[SAVED]")
    print("Model      -> best_baseline_model.pkl")
    print("Vectorizer -> tfidf_vectorizer.pkl")
    print("Metrics    -> results.csv")
    print(f"Saved Model: {best_name}")


if __name__ == "__main__":
    main()
