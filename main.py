#!/usr/bin/env python3

from common.modules import sys, traceback
from gui.experiment_launcher import run

def main():
    """Main Entry Point for Application"""
    try:
        # Start the Flask web server
        run(host='127.0.0.1', port=5000, debug=False)
    except Exception as e:
        print(f"Error starting application: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()