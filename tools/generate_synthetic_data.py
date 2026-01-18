import json
import random
import numpy as np

# ==========================================
#           CONFIGURATION
# ==========================================
OUTPUT_FILE = "morse_sequences.jsonl"
NUM_SAMPLES_PER_CHAR = 1000  # 500 examples per character

# 1. ITU-R M.1677-1 OFFICIAL DICTIONARY
# This includes the standard letters, numbers, and punctuation.
MORSE_CODE_DICT = {
    # Letters
    'A': '.-', 'B': '-...', 'C': '-.-.', 'D': '-..', 'E': '.', 
    'F': '..-.', 'G': '--.', 'H': '....', 'I': '..', 'J': '.---', 
    'K': '-.-', 'L': '.-..', 'M': '--', 'N': '-.', 'O': '---', 
    'P': '.--.', 'Q': '--.-', 'R': '.-.', 'S': '...', 'T': '-', 
    'U': '..-', 'V': '...-', 'W': '.--', 'X': '-..-', 'Y': '-.--', 
    'Z': '--..',
    
    # Numbers
    '0': '-----', '1': '.----', '2': '..---', '3': '...--', '4': '....-', 
    '5': '.....', '6': '-....', '7': '--...', '8': '---..', '9': '----.',
    
    # Punctuation (Standard ITU-R M.1677-1)
    '.': '.-.-.-',  # Period
    ',': '--..--',  # Comma
    '?': '..--..',  # Question Mark
    "'": '.----.',  # Apostrophe
    '/': '-..-.',   # Slash
    '(': '-.--.',   # Open Parenthesis
    ')': '-.--.-',  # Close Parenthesis
    ':': '---...',  # Colon
    '=': '-...-',   # Double Dash / Equals
    '+': '.-.-.',   # Plus
    '-': '-....-',  # Hyphen
    '"': '.-..-.',  # Quote
    '@': '.--.-.'   # At Sign
}

# 2. SIMPLE TIMING CONFIGURATION (Seconds)
# We ignore WPM formulas and just use "Short vs Long" logic.
DOT_MEAN = 0.20   # Increased slightly (0.15 is very fast for a human)
DASH_MEAN = 0.70  # Standard human dash
JITTER = 0.15     # HIGH VARIANCE -> Solves the 'F'/'C' confusion

def generate_duration(symbol):
    """
    Generates a duration with significant random noise to mimic
    imperfect human timing (SOP 4 Generalization).
    """
    if symbol == '.':
        # Generate random time centered around 0.20s
        duration = np.random.normal(DOT_MEAN, JITTER)
        # Clamp Max at 0.45s to leave a gap before the Dash starts
        return max(0.05, min(duration, 0.45)) 
        
    elif symbol == '-':
        # Generate random time centered around 0.70s
        duration = np.random.normal(DASH_MEAN, JITTER)
        # Clamp Min at 0.55s to ensure distinct separation from Dot
        return max(0.55, min(duration, 1.5)) 
        
    return 0.0

def main():
    print(f"Generating data using official ITU-R M.1677-1 codes for {len(MORSE_CODE_DICT)} symbols...")
    
    samples = []
    
    for char, code in MORSE_CODE_DICT.items():
        for _ in range(NUM_SAMPLES_PER_CHAR):
            
            # Generate the float sequence based on the code
            raw_durations = [generate_duration(symbol) for symbol in code]
            
            sample = {
                "raw_durations": raw_durations,
                "morse_seq": code,
                "label": char
            }
            samples.append(sample)

    # Shuffle the dataset
    random.shuffle(samples)

    # Save to file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    print(f"Successfully generated {len(samples)} samples.")
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()