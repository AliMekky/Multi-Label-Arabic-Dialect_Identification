import os
import numpy as np
import torch
from torch.utils.data import DataLoader

# Mock data paths
mock_base_path = "mock_data"
os.makedirs(mock_base_path, exist_ok=True)

# Create mock metadata for testing
mock_images = np.random.randint(0, 256, (10, 3, 448, 448), dtype=np.uint8)  # 10 random images
mock_labels = np.random.randint(0, 2, (10, 20))  # 10 random multilabel vectors for 20 classes
mock_labels_obs = np.copy(mock_labels)

# Save mock data
np.save(os.path.join(mock_base_path, "formatted_train_images.npy"), mock_images)
np.save(os.path.join(mock_base_path, "formatted_val_images.npy"), mock_images)
np.save(os.path.join(mock_base_path, "formatted_train_labels.npy"), mock_labels)
np.save(os.path.join(mock_base_path, "formatted_val_labels.npy"), mock_labels)
np.save(os.path.join(mock_base_path, "formatted_train_labels_obs.npy"), mock_labels_obs)
np.save(os.path.join(mock_base_path, "formatted_val_labels_obs.npy"), mock_labels_obs)

# Parameter dictionary for testing
P = {
    "dataset": "pascal",
    "split_seed": 42,
    "val_frac": 0.2,  # 20% validation
    "ss_seed": 123,  # For subsampling
    "ss_frac_train": 1.0,
    "ss_frac_val": 1.0,
    "train_set_variant": "clean",
    "val_set_variant": "observed",
    "use_feats": False,
    "train_feats_file": "",
    "val_feats_file": "",
}

# Test the code
if __name__ == "__main__":
    # Test get_metadata
    meta = get_metadata(P["dataset"])
    print("Metadata:", meta)

    # Test get_transforms
    tx = get_transforms()
    print("Transforms (train):", tx["train"])

    # Test generate_split
    rng = np.random.RandomState(P["split_seed"])
    idx_train, idx_val = generate_split(10, P["val_frac"], rng)
    print("Train indices:", idx_train)
    print("Validation indices:", idx_val)

    # Test load_data
    data = load_data(mock_base_path, P)
    print("Loaded data keys:", data.keys())
    print("Sample train image shape:", data["train"]["images"][0].shape)
    print("Sample train label:", data["train"]["labels"][0])

    # Test multilabel class
    tx = get_transforms()
    dataset_obj = multilabel(P, tx)
    datasets = dataset_obj.get_datasets()

    print("Train dataset length:", len(datasets["train"]))
    print("Validation dataset length:", len(datasets["val"]))
    print("Test dataset length:", len(datasets["test"]))

    # DataLoader for testing
    train_loader = DataLoader(datasets["train"], batch_size=2, shuffle=True)
    for batch in train_loader:
        print("Batch images shape:", batch["image"].shape)
        print("Batch labels:", batch["label_vec_true"])
        break  # Test one batch
