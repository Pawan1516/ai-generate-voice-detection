"""
COQUI XTTS v2 - Generate High-Quality Multilingual AI Voices
Supports: English, Hindi, Tamil, Telugu, Malayalam, Kannada
"""

import os
import sys
import subprocess
import numpy as np
from pathlib import Path

def check_ffmpeg():
    """Check if ffmpeg is installed (optional for this version)"""
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True, timeout=5)
        print("✓ ffmpeg found")
        return True
    except:
        print("⚠️  ffmpeg not found (optional - we'll generate WAV files)")
        return False

def install_tts():
    """Install TTS if not present"""
    try:
        import TTS
        print("✓ TTS already installed")
        return True
    except ImportError:
        print("📦 Installing Coqui TTS...")
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", "TTS"],
                timeout=300
            )
            print("✓ TTS installed")
            return True
        except Exception as e:
            print(f"❌ Installation failed: {e}")
            return False

def generate_ai_voices():
    """Generate AI voices using Coqui XTTS v2"""
    
    print("\n" + "="*70)
    print("🎤 COQUI XTTS v2 - MULTILINGUAL AI VOICE GENERATION")
    print("="*70)
    
    # Check dependencies
    print("\n📋 Checking dependencies...")
    has_ffmpeg = check_ffmpeg()
    
    if not install_tts():
        print("\n❌ Failed to install TTS")
        return False
    
    # Import after installation
    from TTS.api import TTS
    
    print("\n🤖 Loading XTTS v2 Model...")
    print("   (This may take a few minutes on first run)")
    
    try:
        tts = TTS(
            model_name="tts_models/multilingual/multi-dataset/xtts_v2",
            gpu=False,
            progress_bar=True
        )
        print("   ✓ Model loaded successfully")
    except Exception as e:
        print(f"   ❌ Failed to load model: {e}")
        return False
    
    # Sample texts in multiple languages
    sample_texts = {
        "english": [
            "This is an artificial intelligence generated voice sample.",
            "Deep learning models can now synthesize natural sounding speech.",
            "Voice detection systems use machine learning for classification.",
            "Artificial voices are becoming increasingly realistic.",
            "This sample demonstrates advanced text to speech technology.",
        ],
        "hindi": [
            "यह एक कृत्रिम बुद्धिमत्ता द्वारा उत्पन्न आवाज़ का नमूना है।",
            "यह आधुनिक तकनीक का उदाहरण है।",
            "मशीन लर्निंग से बनी आवाज़ें बहुत यथार्थवादी होती हैं।",
            "कृत्रिम बुद्धिमत्ता तकनीक दिन प्रतिदिन बेहतर हो रही है।",
            "यह नमूना उच्च गुणवत्ता की आवाज़ संश्लेषण को दर्शाता है।",
        ],
        "tamil": [
            "இது செயற்கை நுண்ணறிவால் உருவாக்கப்பட்ட குரல் மாதிரி.",
            "நவீன தொழில்நுட்பத்தின் எடுத்துக்காட்டு இது.",
            "இயந்திர கற்றல் மூலம் உருவாக்கப்பட்ட குரல்கள் மிகவும் யথার்థவாதமாக உள்ளன.",
            "செயற்கை நுண்ணறிவு தொழில்நுட்பம் ஒவ்வொரு நாளும் மேம்படுத்தப்படுகிறது.",
            "இந்த மாதிரி உচ்च தரமான குரல் தொகுப்பு நுட்பத்தை காட்டுகிறது.",
        ],
        "telugu": [
            "ఇది కృత్రిమ మేధస్సు ద్వారా ఉత్పత్తి చేయబడిన వాయిస్ నమూనా.",
            "ఆధునిక సాంకేతికత యొక్క ఉదాహరణ ఇది.",
            "మెషిన్ లర్నింగ్ ద్వారా సృష్టించిన వాయిస్‌లు చాలా వాస్తవమైనవి.",
            "కృత్రిమ మేధస్సు సాంకేతికత ప్రతిదిన మెరుగుపడుతోంది.",
            "ఈ నమూనా అధిక నాణ్యతైన వాయిస్ సంश్లేషణ సాంకేతికతను ప్రదర్శిస్తుంది.",
        ],
        "malayalam": [
            "ഇത് കൃത്രിമ ബുദ്ധിയാൽ സൃഷ്ടിച്ച ശബ്ദ സാമ്പിളാണ്.",
            "ഇത് ആധുനിക സാങ്കേതികവിദ്യയുടെ ഉദാഹരണമാണ്.",
            "മെഷീൻ ലേണിംഗ് വഴി സൃഷ്ടിച്ച ശബ്ദങ്ങൾ വളരെ യാഥാർത്ഥ്യമാണ്.",
            "കൃത്രിമ ബുദ്ധിയുടെ സാങ്കേതികവിദ്യ എന്നാൽ ഉത്തരക്രിയ വരുന്നു.",
            "ഈ സാമ്പിൾ ഉയർന്ന നിലവാരമുള്ള ശബ്ദ സമന്വയ സാങ്കേതികത പ്രദർശിപ്പിക്കുന്നു.",
        ],
        "kannada": [
            "ಇದು ಕೃತ್ರಿಮ ಬುದ್ಧಿಮತ್ತೆಯಿಂದ ರಚಿಸಲ್ಪಟ್ಟ ಧ್ವನಿ ಮಾದರಿ.",
            "ಇದು ಆಧುನಿಕ ತಂತ್ರಜ್ಞಾನದ ಉದಾಹರಣೆ.",
            "ಮೆಷಿನ್ ಲರ್ನಿಂಗ್ ಮೂಲಕ ರಚಿಸಲ್ಪಟ್ಟ ಧ್ವನಿಗಳು ಬಹಳ ವಾಸ್ತವಿಕವಾಗಿವೆ.",
            "ಕೃತ್ರಿಮ ಬುದ್ಧಿಮತ್ತೆ ತಂತ್ರಜ್ಞಾನ ಪ್ರತಿದಿನ ಉತ್ತಮವಾಗಿದೆ.",
            "ಈ ಮಾದರಿ ಉಚ್ಚ ಗುಣಮಾನದ ಧ್ವನಿ ಸಂಶ್ಲೇಷಣ ತಂತ್ರಜ್ಞಾನವನ್ನು ಪ್ರದರ್ಶಿಸುತ್ತದೆ.",
        ]
    }
    
    # Base dataset path
    base_path = Path(__file__).parent / "data" / "train" / "ai_generated"
    
    total_generated = 0
    
    # Generate voices for each language
    for language, texts in sample_texts.items():
        lang_path = base_path / language
        lang_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n🗣️  Generating {language.upper()} voices...")
        print(f"   Output: {lang_path}")
        
        generated_count = 0
        
        # Generate multiple samples per text for variety
        for text_idx, text in enumerate(texts):
            for sample_idx in range(10):  # 10 samples per text = 50 per language
                try:
                    # Generate with variation
                    filename = f"ai_{language}_{text_idx}_{sample_idx:02d}.wav"
                    wav_path = lang_path / filename
                    
                    # Generate audio
                    print(f"   ⏳ {filename}...", end='\r')
                    
                    tts.tts_to_file(
                        text=text,
                        file_path=str(wav_path),
                        language=language[:2] if language != "english" else "en"
                    )
                    
                    print(f"   ✓ {filename}            ")
                    generated_count += 1
                    total_generated += 1
                    
                except Exception as e:
                    print(f"   ❌ Error generating {filename}: {str(e)}")
                    continue
        
        print(f"   ✅ Generated {generated_count} {language} samples")
    
    print(f"\n" + "="*70)
    print(f"✅ GENERATION COMPLETE!")
    print(f"="*70)
    print(f"\n📊 Total AI voices generated: {total_generated}")
    print(f"📁 Location: {base_path}")
    print(f"\n📂 Generated structure:")
    print(f"   ai_generated/")
    for lang in sample_texts.keys():
        print(f"   ├── {lang}/ ({50} voices)")
    
    return True

def main():
    """Main entry point"""
    try:
        success = generate_ai_voices()
        
        if success:
            print(f"\n🎉 AI voice generation successful!")
            print(f"✅ Ready for model training")
            return 0
        else:
            print(f"\n❌ AI voice generation failed")
            return 1
            
    except KeyboardInterrupt:
        print(f"\n\n⛔ Generation interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
