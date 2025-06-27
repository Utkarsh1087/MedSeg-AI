"""
MedSeg-AI: Medical Image Segmentation Training Pipeline
Author: Utkarsh1087 (https://github.com/Utkarsh1087)
"""

import os
import sys
import argparse
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
import importlib

# Resolve project base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_DIR = os.path.join(BASE_DIR, "segmentation_models")
DEFAULT_DATASET_PATH = os.path.join(BASE_DIR, "dataset")
DEFAULT_SAVE_MODEL_PATH = os.path.join(BASE_DIR, "save_trained_models")


def random_flip(image, mask):
    """Random horizontal flip for data augmentation."""
    if random.random() > 0.5:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)
    return image, mask


def random_rotate(image, mask):
    """Random rotation within [-30, 30] degrees with reflection border."""
    angle = random.randint(-30, 30)
    h, w = image.shape[:2]
    m_rot = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    image = cv2.warpAffine(image, m_rot, (w, h), borderMode=cv2.BORDER_REFLECT)
    mask = cv2.warpAffine(mask, m_rot, (w, h), borderMode=cv2.BORDER_REFLECT)
    return image, mask


def adjust_brightness(image):
    """Random brightness adjustment."""
    factor = 0.7 + random.uniform(0, 0.6)
    return np.clip(image * factor, 0, 255).astype(np.uint8)


def load_dataset_with_augmentation(image_dir, mask_dir, img_width=256, img_height=256, augmentation_factor=10):
    """
    Loads paired medical images and ground truth masks, applying data augmentation.
    """
    images = []
    masks = []
    img_files = sorted(os.listdir(image_dir))

    if not img_files:
        raise FileNotFoundError(f"No images found in dataset directory: {image_dir}")

    for img_file in img_files:
        img_path = os.path.join(image_dir, img_file)
        mask_path = os.path.join(mask_dir, img_file)

        if not os.path.exists(mask_path):
            print(f"Warning: Mask file not found for {img_file}, skipping.")
            continue

        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            print(f"Warning: Failed to read {img_file} or mask, skipping.")
            continue

        # Original normalized image and mask
        img_base = cv2.resize(img, (img_width, img_height)) / 255.0
        mask_base = cv2.resize(mask, (img_width, img_height)) / 255.0
        mask_base = np.expand_dims(mask_base, axis=-1)
        images.append(img_base)
        masks.append(mask_base)

        # Generate augmented samples
        for _ in range(max(1, augmentation_factor - 1)):
            img_aug, mask_aug = img.copy(), mask.copy()
            img_aug, mask_aug = random_flip(img_aug, mask_aug)
            img_aug, mask_aug = random_rotate(img_aug, mask_aug)
            img_aug = adjust_brightness(img_aug)

            img_aug = cv2.resize(img_aug, (img_width, img_height)) / 255.0
            mask_aug = cv2.resize(mask_aug, (img_width, img_height)) / 255.0
            mask_aug = np.expand_dims(mask_aug, axis=-1)

            images.append(img_aug)
            masks.append(mask_aug)

    return np.array(images, dtype=np.float32), np.array(masks, dtype=np.float32)


def dice_coefficient(y_true, y_pred, smooth=1.0):
    """Dice coefficient metric for binary segmentation evaluation."""
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred, smooth=1.0):
    """Dice loss function for segmentation model optimization."""
    return 1.0 - dice_coefficient(y_true, y_pred, smooth=smooth)


def get_model(model_name="unet", input_size=(256, 256, 3), model_dir=None):
    """Dynamically loads and instantiates the segmentation model architecture."""
    if model_dir and model_dir not in sys.path:
        sys.path.insert(0, model_dir)
    model_module = importlib.import_module(model_name)
    return model_module.build_model(input_size)


def train_pipeline(args):
    model_dir = os.path.abspath(args.model_dir)
    dataset_path = os.path.abspath(args.dataset_path)
    save_model_path = os.path.abspath(args.save_model_path)
    os.makedirs(save_model_path, exist_ok=True)

    image_dir = os.path.join(dataset_path, "images")
    mask_dir = os.path.join(dataset_path, "masks")

    print(f"Loading dataset from: {dataset_path}")
    x_data, y_data = load_dataset_with_augmentation(
        image_dir=image_dir,
        mask_dir=mask_dir,
        img_width=args.img_width,
        img_height=args.img_height,
        augmentation_factor=args.augmentation_factor,
    )
    print(f"Total dataset samples (including augmentations): {x_data.shape[0]}")

    if x_data.shape[0] > 1:
        x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.2, random_state=42)
    else:
        x_train, y_train = x_data, y_data
        x_val, y_val = x_data, y_data

    print(f"Training samples: {len(x_train)}, Validation samples: {len(x_val)}")

    print(f"Loading model architecture '{args.model_name}'...")
    model = get_model(args.model_name, input_size=(args.img_height, args.img_width, 3), model_dir=model_dir)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr), loss=dice_loss, metrics=[dice_coefficient])
    model.summary()

    print(f"Starting model training for {args.epochs} epochs with batch size {args.batch_size}...")
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size
    )

    model_output_path = os.path.join(save_model_path, f"{args.model_name}_trained.h5")
    model.save(model_output_path)
    print(f"Model successfully saved at: {model_output_path}")

    # Plot & save Loss Curve
    plt.figure(figsize=(8, 6))
    plt.plot(history.history["loss"], label="Train Loss", linewidth=2)
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="Validation Loss", linewidth=2)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("MedSeg-AI: Training and Validation Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    loss_plot_path = os.path.join(save_model_path, "training_loss.png")
    plt.savefig(loss_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Loss curve saved at: {loss_plot_path}")

    # Plot & save Dice Coefficient Curve
    plt.figure(figsize=(8, 6))
    plt.plot(history.history["dice_coefficient"], label="Train Dice Coefficient", linewidth=2)
    if "val_dice_coefficient" in history.history:
        plt.plot(history.history["val_dice_coefficient"], label="Validation Dice Coefficient", linewidth=2)
    plt.xlabel("Epochs")
    plt.ylabel("Dice Coefficient")
    plt.title("MedSeg-AI: Training and Validation Dice Coefficient")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    dice_plot_path = os.path.join(save_model_path, "training_dice_coefficient.png")
    plt.savefig(dice_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Dice coefficient curve saved at: {dice_plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MedSeg-AI: Train Medical Image Segmentation Model")
    parser.add_argument("--model_name", type=str, default="unet", help="Segmentation model architecture name")
    parser.add_argument("--model_dir", type=str, default=DEFAULT_MODEL_DIR, help="Directory containing model architectures")
    parser.add_argument("--dataset_path", type=str, default=DEFAULT_DATASET_PATH, help="Directory containing dataset images and masks")
    parser.add_argument("--save_model_path", type=str, default=DEFAULT_SAVE_MODEL_PATH, help="Directory to save model weights and plots")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for Adam optimizer")
    parser.add_argument("--img_height", type=int, default=256, help="Target image height")
    parser.add_argument("--img_width", type=int, default=256, help="Target image width")
    parser.add_argument("--augmentation_factor", type=int, default=10, help="Number of augmentations per image")

    args = parser.parse_args()
    train_pipeline(args)
