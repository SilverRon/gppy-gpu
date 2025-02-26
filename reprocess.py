#!/usr/bin/env /usr/local/anaconda3/bin/python

from gppy.reprocess import reprocess_folder
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Reprocess data in a folder')
    parser.add_argument('-folder', type=str, help='Folder to reprocess')
    parser.add_argument('-overwrite', type=bool, help='Overwrite existing processed files')
    args = parser.parse_args()
    
    reprocess_folder(args.folder, overwrite=args.overwrite)