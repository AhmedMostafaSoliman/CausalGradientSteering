import os
import sys
from pathlib import Path

def concat_log_files(dir_path):
    """Concatenate out.txt files from immediate subdirectories into all_logs.txt"""
    base_dir = Path(dir_path)
    
    if not base_dir.exists() or not base_dir.is_dir():
        print(f"Error: {dir_path} is not a valid directory")
        return
    
    output_file = base_dir / "all_logs.txt"
    
    with open(output_file, 'w') as outfile:
        for subdir in base_dir.iterdir():
            if subdir.is_dir():
                out_txt_path = subdir / "out.txt"
                if out_txt_path.exists():
                    print(f"Adding {out_txt_path}")
                    outfile.write(f"\n{'='*50}\n")
                    outfile.write(f"LOG FROM: {subdir.name}\n")
                    outfile.write(f"{'='*50}\n")
                    
                    with open(out_txt_path, 'r') as infile:
                        outfile.write(infile.read())
                    outfile.write("\n")
    
    print(f"All logs concatenated into {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python concat_log_files.py <directory_path>")
        sys.exit(1)
    
    concat_log_files(sys.argv[1])
