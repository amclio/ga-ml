import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Input, Dropout
from keras.layers.experimental.preprocessing import CategoryEncoding, StringLookup
import matplotlib.pyplot as plt

df_loaded = pd.read_pickle("data.pkl")

df_loaded["totals.bounces"] = df_loaded["totals.bounces"].fillna(0).astype(int)


y = df_loaded["totals.bounces"].values

categorical_cols = [
    "trafficSource.medium",
    "trafficSource.source",
    "trafficSource.isTrueDirect",
    "hits.isEntrance",
    "hits.hour",
    "hits.hitNumber",
    "hits.isInteraction",
    "device.deviceCategory",
    "device.browser",
    "device.language",
    "geoNetwork.continent",
    "geoNetwork.country",
]

X = df_loaded[categorical_cols]

X_train_df, X_test_df, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Train shape: {X_train_df.shape}")
print(f"Test shape: {X_test_df.shape}")


for col in categorical_cols:
    X_train_df[col] = X_train_df[col].astype(str)
    X_test_df[col] = X_test_df[col].astype(str)

inputs = []
encoded_features = []

for col in categorical_cols:
    input_data = Input(shape=(1,), dtype=tf.string, name=col)
    lookup = StringLookup(output_mode="int", mask_token="")
    lookup.adapt(X_train_df[col].values)
    encoded_data = lookup(input_data)
    encoded_data = CategoryEncoding(
        output_mode="multi_hot", num_tokens=lookup.vocabulary_size()
    )(encoded_data)
    inputs.append(input_data)
    encoded_features.append(encoded_data)

encoded_all = tf.keras.layers.concatenate(encoded_features)
encoder_model = tf.keras.Model(inputs=inputs, outputs=encoded_all)

X_train_encoded = encoder_model.predict(
    {col: X_train_df[col].values for col in categorical_cols}
)
X_test_encoded = encoder_model.predict(
    {col: X_test_df[col].values for col in categorical_cols}
)

model = Sequential()
model.add(Dense(12, input_dim=X_train_encoded.shape[1], activation="relu"))

model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))

model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))

model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))

model.add(Dense(1, activation="sigmoid"))

model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"],
)

history = model.fit(
    X_train_encoded,
    y_train,
    epochs=15,
    batch_size=127,
    validation_split=0.2,
)

accuracy = model.evaluate(X_test_encoded, y_test)[1]
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# plt.figure(figsize=(12, 6))
# plt.plot(history.history["loss"], label="Training Loss")
# plt.plot(history.history["val_loss"], label="Validation Loss")
# plt.title("Training and Validation Loss Over Epochs")
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.legend()
# plt.show()

# plt.figure(figsize=(12, 6))
# plt.plot(history.history["accuracy"])
# plt.plot(history.history["val_accuracy"])
# plt.title("Model Accuracy")
# plt.ylabel("Accuracy")
# plt.xlabel("Epoch")
# plt.legend(["Train", "Validation"], loc="upper left")
# plt.tight_layout()
# plt.show()
