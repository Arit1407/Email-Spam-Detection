import joblib
import numpy as np

# load trained pipeline
model = joblib.load("models/spam_model.pkl")


def explain_prediction(text, top_n=5):
    """
    Return top contributing words for one prediction
    using TF-IDF value * Logistic Regression coefficient
    """

    tfidf = model.named_steps["tfidf"]
    clf = model.named_steps["model"]

    # transform input text
    vec = tfidf.transform([text])

    # tfidf values
    values = vec.toarray()[0]

    # vocabulary words
    feature_names = tfidf.get_feature_names_out()

    # contribution score
    contributions = values * clf.coef_[0]

    # highest contribution indexes
    top_idx = np.argsort(contributions)[::-1]

    important_words = []

    for i in top_idx:
        if values[i] > 0:
            important_words.append({
                "word": feature_names[i],
                "score": round(float(contributions[i]), 4)
            })

        if len(important_words) == top_n:
            break

    return important_words