import os
import glob
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm
import random
import scipy.signal

def augment_audio_advanced(file_path, output_path):
    """
    Apply advanced augmentations to simulate 'Hard' AI samples:
    1. Pitch Smoothing (make it too perfect)
    2. Robotic Artifacts (Ring modulation)
    3. Synthetic Reverb
    4. Spectral Smoothing
    """
    try:
        y, sr = librosa.load(file_path, sr=None)
        
        aug_type = random.choice(['robotic', 'smooth', 'reverb', 'combined'])
        
        if aug_type == 'robotic':
            # Ring Modulation (multiply by sine wave) - creates metallic sound
            mod_freq = random.uniform(20, 100)
            modulator = np.sin(2 * np.pi * mod_freq * np.arange(len(y)) / sr)
            y = y * (0.8 + 0.2 * modulator) # Light effect
            
        elif aug_type == 'smooth':
            # Spectral smoothing (remove micro-fluctuations)
            # Simple way: median filter
            y = scipy.signal.medfilt(y, kernel_size=3)
            
        elif aug_type == 'reverb':
            # Synthetic reverb (simple convolution with decay)
            decay = np.exp(-np.linspace(0, 5, int(sr*0.2)))
            noise = np.random.normal(0, 1, len(decay))
            ir = decay * noise
            y = scipy.signal.convolve(y, ir, mode='same')
            
        elif aug_type == 'combined':
            # Slight pitch shift + gain
            steps = random.uniform(-0.5, 0.5)
            y = librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)
            gain = random.uniform(0.8, 1.2)
            y = y * gain
            
        # Normalize
        y = librosa.util.normalize(y)
        
        sf.write(output_path, y, sr)
        return True
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main():
    ai_dir = "data/train/ai_generated"
    if not os.path.exists(ai_dir):
        print(f"Directory not found: {ai_dir}")
        return

    # Process all AI files
    files = glob.glob(os.path.join(ai_dir, "**/*.mp3"), recursive=True)
    files.extend(glob.glob(os.path.join(ai_dir, "**/*.wav"), recursive=True))
    
    print(f"Augmenting {len(files)} AI files with ADVANCED techniques...")
    
    count = 0
    # Create 1000 hard samples for deep training
    limit = 1000 
    
    for i, f in enumerate(tqdm(files, desc="Augmenting Data")):
        if count >= limit:
            break
            
        # Create augmented filename
        parent = os.path.dirname(f)
        name = os.path.basename(f)
        new_name = f"hard_aug_{count}_{name}" # Unique name
        output_path = os.path.join(parent, new_name)
        
        # Skip if already exists
        if os.path.exists(output_path):
            continue
            
        if augment_audio_advanced(f, output_path):
            count += 1
            
    print(f"Created {count} HARD augmented samples.")

if __name__ == "__main__":
    main()
