"""
ChromaDB Fix Script
Run this if you get "Could not connect to tenant" errors
"""
import os
import shutil

chroma_path = "./chroma_db"
if os.path.exists(chroma_path):
    print("Removing old ChromaDB data...")
    shutil.rmtree(chroma_path)
    print("Done. Restart the server.")
else:
    print("No old data found.")