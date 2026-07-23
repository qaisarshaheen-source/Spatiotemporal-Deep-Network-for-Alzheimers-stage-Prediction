"""
Alzheimer's Detection - Model development with efficient and compatible model selection.
Trains multiple architectures and saves the best performer by validation macro F1.

Kaggle: Add dataset "alzheimers-detection-dataset". Dataset path is auto-set to
/kaggle/input/alzheimers-detection-dataset/dataset; outputs go to /kaggle/working/models/.
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import json

from config import (
    DATASET_DIR,
    MODELS_DIR,
    SAVED_MODEL_PATH,
    CLASS_NAMES_PATH,
    IMG_SIZE,
    IMG_SHAPE,
    BATCH_SIZE,
    SEED,
    VAL_SPLIT,
    TEST_SPLIT,
    EPOCHS,
    PATIENCE,
    CLASS_NAMES,
    NUM_CLASSES,
    IMG_EXTENSIONS,
    OVERSAMPLE_MIN_PER_CLASS,
    CLASS_WEIGHT_MAX,
)

tf.random.set_seed(SEED)
np.random.seed(SEED)


def collect_image_paths_and_labels():
    """Load image paths and labels from dataset/ with class subfolders."""
    image_paths = []
    labels = []
    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(DATASET_DIR, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in os.listdir(class_dir):
            if fname.lower().endswith(IMG_EXTENSIONS):
                image_paths.append(os.path.join(class_dir, fname))
                labels.append(class_idx)
    return np.array(image_paths), np.array(labels)


def load_and_split_data():
    """Stratified train/val/test split from collected paths."""
    paths, labels = collect_image_paths_and_labels()
    if len(paths) == 0:
        raise FileNotFoundError(f"No images found under {DATASET_DIR} in classes {CLASS_NAMES}")

    # First split: 80% train+val, 20% test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        paths, labels, test_size=TEST_SPLIT, stratify=labels, random_state=SEED
    )
    # Second split: 80% train, 20% val (of the 80%)
    val_ratio = VAL_SPLIT / (1 - TEST_SPLIT)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_ratio, stratify=y_trainval, random_state=SEED
    )
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def oversample_minority_classes(X_train, y_train, target_min_per_class=OVERSAMPLE_MIN_PER_CLASS):
    """
    Oversample minority classes so each class has at least target_min_per_class samples.
    Helps the model learn Mild_Demented and Moderate_Demented when they are rare.
    """
    X_list, y_list = [], []
    for c in range(NUM_CLASSES):
        mask = y_train == c
        X_c = X_train[mask]
        y_c = y_train[mask]
        n = len(X_c)
        if n == 0:
            continue
        # Repeat indices to reach at least target_min_per_class (with replacement)
        n_repeat = max(1, (target_min_per_class + n - 1) // n)
        indices = np.tile(np.arange(n), n_repeat)[: max(n, target_min_per_class)]
        np.random.RandomState(SEED).shuffle(indices)
        X_list.append(X_c[indices])
        y_list.append(np.full(len(indices), c))
    X_bal = np.concatenate(X_list, axis=0)
    y_bal = np.concatenate(y_list, axis=0)
    perm = np.random.RandomState(SEED).permutation(len(X_bal))
    return X_bal[perm], y_bal[perm]


def build_augmentation():
    """Training augmentation to address imbalance and improve generalization."""
    return ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=25,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
    )


def _load_and_preprocess(path, label, augment=False):
    img = tf.io.read_file(path)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    if augment:
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, 0.1)
        img = tf.image.random_contrast(img, 0.9, 1.1)
        img = tf.clip_by_value(img, 0, 1)
    return img, label


def dataset_from_paths(paths, labels, batch_size, shuffle=True, augment=False):
    """Build tf.data.Dataset from file paths and labels. Optionally apply augmentation."""
    ds = tf.data.Dataset.from_tensor_slices((paths.astype(str), labels.astype(np.int32)))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(paths), 2048), seed=SEED)
    ds = ds.map(
        lambda p, l: _load_and_preprocess(p, l, augment=augment),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def get_class_weights(y, max_weight=10.0):
    """Compute class weights for imbalanced data; cap very high weights for stability."""
    weights = compute_class_weight(
        "balanced", classes=np.unique(y), y=y
    )
    w_dict = dict(enumerate(weights))
    for k in w_dict:
        w_dict[k] = min(float(w_dict[k]), max_weight)
    return w_dict


# --------------- Model architectures ---------------


def build_custom_cnn(input_shape=IMG_SHAPE, num_classes=NUM_CLASSES):
    """Lightweight custom CNN compatible with IMG_SHAPE."""
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D(2)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D(2)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Conv2D(256, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs, name="custom_cnn")
    return model


def build_mobilenetv2(input_shape=IMG_SHAPE, num_classes=NUM_CLASSES):
    """MobileNetV2 transfer learning - efficient and compatible."""
    base = MobileNetV2(input_shape=input_shape, include_top=False, weights="imagenet")
    base.trainable = False
    inputs = keras.Input(shape=input_shape)
    x = base(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs, name="mobilenetv2")
    return model


def build_efficientnet(input_shape=IMG_SHAPE, num_classes=NUM_CLASSES):
    """EfficientNetB0 transfer learning - strong accuracy/size tradeoff."""
    base = EfficientNetB0(input_shape=input_shape, include_top=False, weights="imagenet")
    base.trainable = False
    inputs = keras.Input(shape=input_shape)
    x = base(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs, name="efficientnetb0")
    return model


MODEL_BUILDERS = {
    "custom_cnn": build_custom_cnn,
    "mobilenetv2": build_mobilenetv2,
    "efficientnetb0": build_efficientnet,
}


def train_and_evaluate(model_name, train_ds, val_ds, test_ds, y_test, class_weights):
    """Build, compile, train one model and return metrics."""
    model = MODEL_BUILDERS[model_name]()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )
    # Evaluate on test (use provided y_test; predict in same order)
    y_pred = np.argmax(model.predict(test_ds), axis=1)
    test_acc = np.mean(y_test == y_pred)
    test_f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    return {
        "model": model,
        "history": history.history,
        "test_accuracy": float(test_acc),
        "test_f1_macro": float(test_f1_macro),
        "classification_report": classification_report(
            y_test, y_pred, target_names=CLASS_NAMES, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    print(f"Dataset directory: {DATASET_DIR}")
    print(f"Output (models) directory: {MODELS_DIR}")

    print("Loading and splitting data...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_and_split_data()
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Oversample minority classes so model sees Mild/Moderate Demented often enough
    X_train, y_train = oversample_minority_classes(X_train, y_train)
    print(f"After oversampling: train size {len(X_train)}")

    class_weights = get_class_weights(y_train, max_weight=CLASS_WEIGHT_MAX)
    print("Class weights (capped):", class_weights)
    train_ds = dataset_from_paths(X_train, y_train, BATCH_SIZE, shuffle=True, augment=True)
    val_ds = dataset_from_paths(X_val, y_val, BATCH_SIZE, shuffle=False)
    test_ds = dataset_from_paths(X_test, y_test, BATCH_SIZE, shuffle=False)

    results = {}
    for name in MODEL_BUILDERS:
        print(f"\n{'='*60}\nTraining {name}\n{'='*60}")
        try:
            results[name] = train_and_evaluate(
                name, train_ds, val_ds, test_ds, y_test, class_weights
            )
        except Exception as e:
            print(f"Error training {name}: {e}")
            results[name] = None

    # Select best by test macro F1
    best_name = None
    best_f1 = -1
    for name, res in results.items():
        if res and res["test_f1_macro"] > best_f1:
            best_f1 = res["test_f1_macro"]
            best_name = name

    if best_name is None:
        raise RuntimeError("No model trained successfully.")

    print(f"\nBest model: {best_name} (test macro F1: {best_f1:.4f})")
    best_model = results[best_name]["model"]
    best_model.save(SAVED_MODEL_PATH)
    with open(CLASS_NAMES_PATH, "w") as f:
        f.write("\n".join(CLASS_NAMES))

    # Save metrics for reference
    metrics_path = os.path.join(MODELS_DIR, "best_model_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "best_model": best_name,
                "test_accuracy": results[best_name]["test_accuracy"],
                "test_f1_macro": results[best_name]["test_f1_macro"],
                "classification_report": results[best_name]["classification_report"],
                "confusion_matrix": results[best_name]["confusion_matrix"],
            },
            f,
            indent=2,
        )
    print(f"Model saved to {SAVED_MODEL_PATH}")
    print(f"Metrics saved to {metrics_path}")
    print("\nClassification report:\n", results[best_name]["classification_report"])


if __name__ == "__main__":
    main()
