import string

import matplotlib.pyplot as plt
import nltk
import pandas as pd
from keras.layers import Dense, InputLayer, TextVectorization
from keras.models import Sequential
from nltk.corpus import stopwords
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from tensorflow import keras

nltk.download("stopwords")


STOP_WORDS = set(stopwords.words("english"))


def remove_punctuation(text: str) -> str:
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def remove_stopwords(text: str) -> str:
    words = text.split()
    filtered_words = [word for word in words if word not in STOP_WORDS]
    return " ".join(filtered_words)


def remove_short_words(text: str) -> str:
    words = text.split()
    filtered_words = [word for word in words if len(word) > 3]
    return " ".join(filtered_words)


def remove_numbers(text: str) -> str:
    text = text.translate(str.maketrans("", "", string.digits))
    return text


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop_duplicates()
    df["text"] = df["text"].apply(remove_punctuation)
    df["text"] = df["text"].apply(remove_stopwords)
    df["text"] = df["text"].apply(remove_short_words)
    df["text"] = df["text"].apply(remove_numbers)
    df["text"] = df["text"].str.lower()
    return df


def main():
    # Préparation des données.
    print("\nPréparation des données...")
    df = load_data("emails.csv")
    df = clean_data(df)

    # Création des jeux de données d'entrainement et de test.
    df_train, df_test = train_test_split(df, test_size=0.3)
    X_train = df_train.text.values
    y_train = df_train.spam.values
    X_test = df_test.text.values
    y_test = df_test.spam.values

    # Numérisation de la variable text.
    vectorize_layer = TextVectorization(output_mode="count")
    vectorize_layer.adapt(X_train)
    X_train_vectorized = vectorize_layer(X_train)
    X_test_vectorized = vectorize_layer(X_test)

    # Création du model.
    feature_size = X_train_vectorized.shape[1]
    model = Sequential()
    input_layer = InputLayer(shape=(feature_size,))
    first_hidden_layer = Dense(64, activation="relu")
    second_hidden_layer = Dense(32, activation="relu")
    output_layer = Dense(1, activation="sigmoid")
    model.add(input_layer)
    model.add(first_hidden_layer)
    model.add(second_hidden_layer)
    model.add(output_layer)
    model.summary()

    # Entrainement du model.
    print("\nEntrainement...")
    loss = keras.losses.BinaryCrossentropy()
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])
    history = model.fit(
        X_train_vectorized, y_train, batch_size=64, epochs=10, validation_split=0.2
    )

    # Affichage de des courbes d'apprentissages.
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="validation")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("loss_curve.png")
    plt.close()

    # Taux de précision.
    loss, accuracy = model.evaluate(X_test_vectorized, y_test)
    print(f"\nTaux de précision : {accuracy * 100:.2f}%")

    # Matrice de confusion.
    y_pred = model.predict(X_test_vectorized)
    y_pred = (y_pred > 0.5).astype(int)
    cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
    print(classification_report(y_true=y_test, y_pred=y_pred))
    ConfusionMatrixDisplay(confusion_matrix=cm).plot()
    plt.savefig("confusion_matrix.png")
    plt.close()

    print("Courbes d'apprentissages sauvegardé : 'loss_curve.png'")
    print("Matrice de confusion sauvegardé     : 'confusion_matrix.png'")


if __name__ == "__main__":
    main()
