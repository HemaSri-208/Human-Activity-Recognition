import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

OUTPUT_DIR = Path("./har_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

DATA_DIR  = Path(r"C:\Users\aalla\Downloads\PycharmProjects\PycharmProjects\human\.venv\New_ImageHumanActivityRecog_2026\Dataset")
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR  = DATA_DIR / "test"

MODEL_FILE = Path(r"C:\Users\aalla\Downloads\PycharmProjects\PycharmProjects\human\.venv\New_ImageHumanActivityRecog_2026\ModelFiles\har_efficientnet_model.h5")
BEST_MODEL_FILE = Path(r"C:\Users\aalla\Downloads\PycharmProjects\PycharmProjects\human\.venv\New_ImageHumanActivityRecog_2026\ModelFiles\har_efficientnet_best_model.h5")

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50

train_df = pd.read_csv(DATA_DIR / "Training_set.csv")
test_df  = pd.read_csv(DATA_DIR / "Testing_set.csv")

train_df["filename"] = train_df["filename"].astype(str).str.strip()
test_df["filename"]  = test_df["filename"].astype(str).str.strip()

def build_image_map(folder):
    image_map = {}
    for p in Path(folder).glob("*"):
        image_map[p.name.lower()] = str(p)
        image_map[p.stem.lower()] = str(p)
    return image_map

train_map = build_image_map(TRAIN_DIR)
test_map  = build_image_map(TEST_DIR)

def resolve_path(name, image_map):
    name = name.strip().lower()
    return image_map.get(name, None)

train_df["filepath"] = train_df["filename"].apply(lambda x: resolve_path(x, train_map))
test_df["filepath"]  = test_df["filename"].apply(lambda x: resolve_path(x, test_map))

train_df = train_df[train_df["filepath"].notnull()].reset_index(drop=True)
test_df  = test_df[test_df["filepath"].notnull()].reset_index(drop=True)

if len(train_df) == 0:
    raise ValueError("Train dataset is empty.")
if len(test_df) == 0:
    raise ValueError("Test dataset is empty.")

CLASSES = sorted(train_df["label"].unique())
label2idx = {c:i for i,c in enumerate(CLASSES)}
idx2label = {v:k for k,v in label2idx.items()}

train_df["label_idx"] = train_df["label"].map(label2idx)

train_df, val_df = train_test_split(
    train_df,
    test_size=0.2,
    stratify=train_df["label"],
    random_state=42
)

def preprocess_image(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = img / 255.0
    return img, label

def preprocess_test_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = img / 255.0
    return img

train_ds = tf.data.Dataset.from_tensor_slices(
    (train_df["filepath"].values, train_df["label_idx"].values)
)
train_ds = train_ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.shuffle(1024).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices(
    (val_df["filepath"].values, val_df["label_idx"].values)
)
val_ds = val_ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

test_ds = tf.data.Dataset.from_tensor_slices(test_df["filepath"].values)
test_ds = test_ds.map(preprocess_test_image, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

if MODEL_FILE.exists():
    model = tf.keras.models.load_model(str(MODEL_FILE))
else:
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base_model.trainable = True

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.1)(x)
    x = layers.RandomZoom(0.1)(x)
    x = base_model(x, training=True)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(len(CLASSES), activation="softmax")(x)

    model = models.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    early_stop = callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=7,
        restore_best_weights=True
    )

    checkpoint = callbacks.ModelCheckpoint(
        str(BEST_MODEL_FILE),
        monitor='val_accuracy',
        save_best_only=True
    )

    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[early_stop, checkpoint, reduce_lr]
    )

    model.save(str(MODEL_FILE))

    np.save(r"C:\Users\aalla\Downloads\PycharmProjects\PycharmProjects\human\.venv\New_ImageHumanActivityRecog_2026\Reports\train_loss.npy", history.history["loss"])
    np.save(r"C:\Users\aalla\Downloads\PycharmProjects\PycharmProjects\human\.venv\New_ImageHumanActivityRecog_2026\Reports\val_loss.npy", history.history["val_loss"])
    np.save(r"C:\Users\aalla\Downloads\PycharmProjects\PycharmProjects\human\.venv\New_ImageHumanActivityRecog_2026\Reports\train_acc.npy", history.history["accuracy"])
    np.save(r"C:\Users\aalla\Downloads\PycharmProjects\PycharmProjects\human\.venv\New_ImageHumanActivityRecog_2026\Reports\val_acc.npy", history.history["val_accuracy"])

    plt.figure()
    plt.plot(history.history["accuracy"], label="train_accuracy")
    plt.plot(history.history["val_accuracy"], label="val_accuracy")
    plt.legend()
    plt.savefig(r"C:\Users\aalla\Downloads\PycharmProjects\PycharmProjects\human\.venv\New_ImageHumanActivityRecog_2026\Reports\accuracy_graph.png")

    plt.figure()
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.legend()
    plt.savefig(r"C:\Users\aalla\Downloads\PycharmProjects\PycharmProjects\human\.venv\New_ImageHumanActivityRecog_2026\Reports\loss_graph.png")

preds = model.predict(test_ds)
pred_labels = np.argmax(preds, axis=1)

test_df["predicted_label"] = [idx2label[i] for i in pred_labels]

test_df.to_csv(
    r"C:\Users\aalla\Downloads\PycharmProjects\PycharmProjects\human\.venv\New_ImageHumanActivityRecog_2026\Reports\test_predictions.csv",
    index=False
)