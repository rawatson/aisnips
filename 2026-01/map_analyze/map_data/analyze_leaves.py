import struct
import os

file_path = "nodes.dat"
leaf_start_index = 87381

def interpret_chunk(chunk):
    floats = struct.unpack('<8f', chunk)
    ints = struct.unpack('<8i', chunk)
    return floats, ints

try:
    with open(file_path, 'rb') as f:
        f.seek(leaf_start_index * 32)
        print(f"Reading Leaf Nodes starting at index {leaf_start_index}")
        
        for i in range(10):
            chunk = f.read(32)
            if not chunk: break
            floats, ints = interpret_chunk(chunk)
            print(f"Leaf {i+leaf_start_index}:")
            # UV Size
            print(f"  Pos: ({floats[0]:.5f}, {floats[1]:.5f}) Size: {floats[2]:.5f}")
            # Data
            print(f"  Data Ints: {ints[3:]}")
            print(f"  Data Floats: {floats[3:]}")
            
except Exception as e:
    print(f"Error: {e}")
