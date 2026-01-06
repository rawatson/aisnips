import os

file_path = "nodes.dat"

def hexdump(data, width=16):
    for i in range(0, len(data), width):
        chunk = data[i:i+width]
        hex_part = ' '.join(f'{b:02x}' for b in chunk)
        ascii_part = ''.join(chr(b) if 32 <= b < 127 else '.' for b in chunk)
        print(f'{i:08x}  {hex_part:<{width*3}}  |{ascii_part}|')

try:
    with open(file_path, 'rb') as f:
        header = f.read(256)
        print(f"File size: {os.path.getsize(file_path)} bytes")
        print("First 256 bytes:")
        hexdump(header)
        
        # Check for repeating patterns or structures if it's large
        f.seek(0)
        full_data = f.read()
        
        # Maybe it's a list of structs? Let's check the size divisibility.
        size = len(full_data)
        print(f"\nTotal size: {size}")
        for stride in [4, 8, 12, 16, 20, 24, 32, 64]:
            if size % stride == 0:
                print(f"Size is divisible by {stride} (count: {size//stride})")

except Exception as e:
    print(f"Error reading file: {e}")
