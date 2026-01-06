import struct
import os

file_path = "nodes.dat"

def interpret_chunk(chunk):
    # Try interpreting as 8 floats (32 bytes)
    floats = struct.unpack('<8f', chunk)
    # Try interpreting as 8 ints
    ints = struct.unpack('<8i', chunk)
    
    print(f"Floats: {floats}")
    print(f"Ints:   {ints}")

try:
    file_size = os.path.getsize(file_path)
    count = file_size // 32
    print(f"File size: {file_size}, Records (32b): {count}")
    
    with open(file_path, 'rb') as f:
        for i in range(10):
            print(f"\nRecord {i}:")
            chunk = f.read(32)
            if not chunk: break
            interpret_chunk(chunk)
            
except Exception as e:
    print(f"Error: {e}")
