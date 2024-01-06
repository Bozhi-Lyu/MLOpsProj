import torch
import pandas as pd
import torch

def make_dataset(raw_dir = "./data/raw/", processed_dir = "./data/processed/"):
    
    print("Processing dataset...")
    print("Raw directory:", raw_dir)

    data = pd.read_csv(raw_dir + "train.csv")
    print("Headers:\n", data.head())

    labels = torch.from_numpy(data['emotion'].values).unsqueeze(1)
    pixels_column = data.iloc[:, 1].tolist()
    pixels = torch.tensor([list(map(int, row.split())) for row in pixels_column])/255.0

    print("Data procession completed.")
    print("Labels:", labels.shape)
    print("Pixels:", pixels.shape)

    print("Processed directory:", processed_dir)
    torch.save(labels, processed_dir + "train_labels.pt")
    torch.save(pixels, processed_dir + "normalized_train_images.pt")
    print("Processing dataset completed.")


if __name__ == '__main__':
    # Get the data and process it
    make_dataset()

