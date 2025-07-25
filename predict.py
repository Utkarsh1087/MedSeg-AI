"""
MedSeg-AI: Medical Image Segmentation Inference & Prediction Pipeline
Author: Utkarsh1087 (https://github.com/Utkarsh1087)
"""

import os
import argparse
import numpy as np
import cv2
import tensorflow as tf

# Resolve project base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SAVE_MODEL_PATH = os.path.join(BASE_DIR, "save_trained_models")
DEFAULT_INPUT_IMAGE_PATH = os.path.join(BASE_DIR, "input_images")
DEFAULT_PREDICTED_MASK_PATH = os.path.join(BASE_DIR, "predicted_masks")


def dice_coefficient(y_true, y_pred, smooth=1.0):
    """Dice coefficient metric for binary segmentation evaluation."""
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred, smooth=1.0):
    """Dice loss function for segmentation model optimization."""
    return 1.0 - dice_coefficient(y_true, y_pred, smooth=smooth)


def load_and_preprocess_image(image_path, target_size=(256, 256)):
    """
    Loads an input image, preserves its original aspect ratio / dimensions,
    and returns a normalized tensor ready for inference.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Failed to load image at: {image_path}")

    orig_height, orig_width = image.shape[:2]
    image_resized = cv2.resize(image, target_size) / 255.0
    return image_resized, orig_height, orig_width


def run_inference(args):
    save_model_path = os.path.abspath(args.save_model_path)
    input_image_path = os.path.abspath(args.input_image_path)
    predicted_mask_path = os.path.abspath(args.predicted_mask_path)
    os.makedirs(predicted_mask_path, exist_ok=True)

    model_file = os.path.join(save_model_path, f"{args.model_name}_trained.h5")
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model file not found at: {model_file}")

    print(f"Loading trained weights from: {model_file}...")
    custom_objects = {
        "dice_loss": dice_loss,
        "dice_coefficient": dice_coefficient
    }
    model = tf.keras.models.load_model(model_file, custom_objects=custom_objects)
    print(f"Model '{args.model_name}' loaded successfully.")

    input_images = sorted(os.listdir(input_image_path))
    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    image_files = [f for f in input_images if f.lower().endswith(valid_extensions)]

    if not image_files:
        raise FileNotFoundError(f"No valid images found in: {input_image_path}")

    print(f"Found {len(image_files)} images for prediction.")

    for img_name in image_files:
        img_path = os.path.join(input_image_path, img_name)
        img_processed, orig_h, orig_w = load_and_preprocess_image(
            img_path, target_size=(args.img_size, args.img_size)
        )
        img_tensor = np.expand_dims(img_processed, axis=0)

        # Predict segmentation mask
        predicted_mask = model.predict(img_tensor, verbose=0)[0]

        # Apply threshold to produce binary segmentation mask
        binary_mask = (predicted_mask > args.threshold).astype(np.uint8) * 255

        # Resize mask back to original image dimensions using nearest-neighbor interpolation
        mask_resized = cv2.resize(binary_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        mask_save_path = os.path.join(predicted_mask_path, img_name)
        cv2.imwrite(mask_save_path, mask_resized)
        print(f"Generated mask for {img_name} -> {mask_save_path}")

    print(f"Inference complete! All predicted masks saved in: {predicted_mask_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MedSeg-AI: Medical Image Segmentation Inference")
    parser.add_argument("--model_name", type=str, default="unet", help="Segmentation model architecture name")
    parser.add_argument("--save_model_path", type=str, default=DEFAULT_SAVE_MODEL_PATH, help="Directory containing saved model weights")
    parser.add_argument("--input_image_path", type=str, default=DEFAULT_INPUT_IMAGE_PATH, help="Directory containing input images")
    parser.add_argument("--predicted_mask_path", type=str, default=DEFAULT_PREDICTED_MASK_PATH, help="Directory to save predicted masks")
    parser.add_argument("--img_size", type=int, default=256, help="Input dimension for model inference")
    parser.add_argument("--threshold", type=float, default=0.5, help="Binarization probability threshold")

    args = parser.parse_args()
    run_inference(args)
