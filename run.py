#!/usr/bin/env python3
"""
Enhanced AI Tutor System Runner
Simple script to start the application with all new features
"""

import os
import sys

def main():
    print("🎓 Starting Enhanced AI Tutor System...")
    print("=" * 60)
    print("✨ New Features:")
    print("   🔊 Mute/Unmute speech functionality")
    print("   📝 Enhanced quiz with answer reveal and scoring")
    print("   🧠 Improved LLM responses for comprehensive answers")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not os.path.exists('app.py'):
        print("❌ Error: app.py not found!")
        print("Please run this script from the ai_tutor_enhanced directory")
        sys.exit(1)
    
    # Check if requirements are installed
    try:
        import flask
        import flask_cors
        import requests
        print("✅ Dependencies verified")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        sys.exit(1)
    
    print("🚀 Starting Flask server...")
    print("📱 Open your browser to: https://127.0.0.1:5000")
    print("🎤 Make sure to allow microphone permissions!")
    print("🔊 Use the mute/unmute button to control speech output!")
    print("=" * 60)
    
    # Import and run the app
    from app import app
    app.run(host='0.0.0.0', port=5000, debug=True, ssl_context="adhoc")

if __name__ == '__main__':
    main()

