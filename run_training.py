#!/usr/bin/env python3

import os
import argparse
import torch
from SWINUNETR_ARTF.train import SWINUNETRTrainer

def parse_args():
    parser = argparse.ArgumentParser(description="Train SwinUNETR model with enhanced small structure focus")
    
    # Dataset parameters
    parser.add_argument("--dataset_dir", type=str, required=True, 
                        help="Path to the dataset directory")
    parser.add_argument("--fold", type=int, default=0, 
                        help="Fold number to use for training/validation split")
    parser.add_argument("--output_dir", type=str, default="./output", 
                        help="Directory to save model outputs")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=2, 
                        help="Batch size for training")
    parser.add_argument("--max_epochs", type=int, default=200, 
                        help="Maximum number of epochs to train")
    parser.add_argument("--learning_rate", type=float, default=1e-4, 
                        help="Initial learning rate")
    parser.add_argument("--num_workers", type=int, default=4, 
                        help="Number of worker processes for data loading")
    
    # Model parameters
    parser.add_argument("--patch_size", type=str, default="32,160,256", 
                        help="Patch size for training (comma-separated)")
    parser.add_argument("--feature_size", type=int, default=48, 
                        help="Feature size for the SwinUNETR model")
    parser.add_argument("--roi", action="store_true", 
                        help="Whether to use ROI masks")
    parser.add_argument("--LR", action="store_true", 
                        help="Whether to use left/right distinction")
    
    # Logging and debugging
    parser.add_argument("--verbose", action="store_true", 
                        help="Enable verbose logging")
    parser.add_argument("--wandb", action="store_true", 
                        help="Enable Weights & Biases logging")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Convert patch size from string to tuple
    args.patch_size = tuple(map(int, args.patch_size.split(',')))
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create the trainer
    trainer = SWINUNETRTrainer(args)
    
    # Train the model
    best_dice = trainer.train_model()
    
    print(f"Training completed! Best validation Dice score: {best_dice:.4f}")
    print(f"Model saved to: {os.path.join(args.output_dir, 'best_model.pt')}")

if __name__ == "__main__":
    main() 