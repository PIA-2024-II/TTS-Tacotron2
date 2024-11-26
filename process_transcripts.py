import os
import csv

def process_transcripts(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8', newline='') as outfile:
        reader = csv.reader(infile, delimiter='|')
        writer = csv.writer(outfile, delimiter='|')
        for row in reader:
            audio_path, text = row
            # Validate and process Mapudungun special characters if needed
            writer.writerow([audio_path, text])

def split_dataset(input_file, train_file, val_file, train_ratio=0.8):
    with open(input_file, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()
    train_size = int(len(lines) * train_ratio)
    with open(train_file, 'w', encoding='utf-8') as train_out, open(val_file, 'w', encoding='utf-8') as val_out:
        train_out.writelines(lines[:train_size])
        val_out.writelines(lines[train_size:])

if __name__ == "__main__":
    input_file = 'data/transcripts pc lucho.txt'
    output_file = 'data/metadata.csv'
    train_file = 'data/train.csv'
    val_file = 'data/val.csv'
    
    process_transcripts(input_file, output_file)
    split_dataset(output_file, train_file, val_file)