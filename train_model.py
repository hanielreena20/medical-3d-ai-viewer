import os
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

import medmnist
from medmnist import INFO


# ============================================================
# CONFIGURATION
# ============================================================

BATCH_SIZE = 8
EPOCHS = 30
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 1e-4

PATIENCE = 6

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("3D MEDICAL AI TRAINING")
print("=" * 60)

print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("Device:", DEVICE)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    print(
        "GPU Memory:",
        round(
            torch.cuda.get_device_properties(0).total_memory
            / 1024**3,
            2
        ),
        "GB"
    )

else:

    print("WARNING: CUDA is not available.")


# ============================================================
# DATASET
# ============================================================

DATA_FLAG = "organmnist3d"

info = INFO[DATA_FLAG]

DataClass = getattr(
    medmnist,
    info["python_class"]
)


train_dataset = DataClass(
    split="train",
    download=True
)

val_dataset = DataClass(
    split="val",
    download=True
)

test_dataset = DataClass(
    split="test",
    download=True
)


print()
print("Dataset:", DATA_FLAG)
print("Train:", len(train_dataset))
print("Validation:", len(val_dataset))
print("Test:", len(test_dataset))


# ============================================================
# DATASET WRAPPER
# ============================================================

class Medical3DDataset(Dataset):

    def __init__(
        self,
        dataset,
        augment=False
    ):

        self.dataset = dataset

        self.augment = augment


    def __len__(self):

        return len(self.dataset)


    def __getitem__(self, index):

        image, label = self.dataset[index]

        image = np.asarray(image)

        image = np.squeeze(image)

        # ----------------------------------------------------
        # IMPORTANT:
        # MedMNIST3D returns values in [0,1]
        # ----------------------------------------------------

        image = image.astype(
            np.float32
        )

        # ----------------------------------------------------
        # Add channel dimension
        #
        # 28 x 28 x 28
        #
        # →
        #
        # 1 x 28 x 28 x 28
        # ----------------------------------------------------

        image = torch.from_numpy(
            image
        ).unsqueeze(0)


        # ----------------------------------------------------
        # DATA AUGMENTATION
        # ----------------------------------------------------

        if self.augment:

            # Random flip X
            if np.random.rand() > 0.5:

                image = torch.flip(
                    image,
                    dims=[1]
                )


            # Random flip Y
            if np.random.rand() > 0.5:

                image = torch.flip(
                    image,
                    dims=[2]
                )


            # Random flip Z
            if np.random.rand() > 0.5:

                image = torch.flip(
                    image,
                    dims=[3]
                )


            # Small intensity variation
            if np.random.rand() > 0.5:

                factor = np.random.uniform(
                    0.9,
                    1.1
                )

                image = image * factor


        image = torch.clamp(
            image,
            0,
            1
        )


        label = int(
            np.asarray(label).squeeze()
        )


        return (
            image,
            torch.tensor(
                label,
                dtype=torch.long
            )
        )


# ============================================================
# CREATE DATASETS
# ============================================================

train_data = Medical3DDataset(
    train_dataset,
    augment=True
)

val_data = Medical3DDataset(
    val_dataset,
    augment=False
)

test_data = Medical3DDataset(
    test_dataset,
    augment=False
)


# ============================================================
# DATA LOADERS
# ============================================================

train_loader = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

test_loader = DataLoader(
    test_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)


# ============================================================
# 3D CNN
# ============================================================

class Organ3DCNN(nn.Module):

    def __init__(
        self,
        num_classes=11
    ):

        super().__init__()


        self.features = nn.Sequential(

            # ----------------------------------------------
            # BLOCK 1
            # ----------------------------------------------

            nn.Conv3d(
                1,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm3d(32),

            nn.ReLU(inplace=True),

            nn.MaxPool3d(2),

            nn.Dropout3d(0.10),


            # ----------------------------------------------
            # BLOCK 2
            # ----------------------------------------------

            nn.Conv3d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm3d(64),

            nn.ReLU(inplace=True),

            nn.MaxPool3d(2),

            nn.Dropout3d(0.15),


            # ----------------------------------------------
            # BLOCK 3
            # ----------------------------------------------

            nn.Conv3d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm3d(128),

            nn.ReLU(inplace=True),

            nn.MaxPool3d(2),

            nn.Dropout3d(0.20)
        )


        # --------------------------------------------------
        # Global average pooling
        #
        # This greatly reduces overfitting compared with
        # Flatten -> huge fully-connected layer.
        # --------------------------------------------------

        self.pool = nn.AdaptiveAvgPool3d(
            (1, 1, 1)
        )


        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                128,
                64
            ),

            nn.ReLU(inplace=True),

            nn.Dropout(0.40),

            nn.Linear(
                64,
                num_classes
            )
        )


    def forward(self, x):

        x = self.features(x)

        x = self.pool(x)

        x = self.classifier(x)

        return x


# ============================================================
# MODEL
# ============================================================

NUM_CLASSES = len(
    info["label"]
)

model = Organ3DCNN(
    num_classes=NUM_CLASSES
)

model = model.to(
    DEVICE
)


print()
print(model)


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# LEARNING RATE SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2
)


# ============================================================
# SAVE PATH
# ============================================================

os.makedirs(
    "models",
    exist_ok=True
)

MODEL_PATH = (
    "models/organ3d_cnn_best.pth"
)


# ============================================================
# TRAINING VARIABLES
# ============================================================

best_val_accuracy = 0.0

epochs_without_improvement = 0


# ============================================================
# TRAINING LOOP
# ============================================================

for epoch in range(EPOCHS):


    # ========================================================
    # TRAIN
    # ========================================================

    model.train()

    running_loss = 0.0

    correct = 0

    total = 0


    for images, labels in train_loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )


        optimizer.zero_grad(
            set_to_none=True
        )


        outputs = model(
            images
        )


        loss = criterion(
            outputs,
            labels
        )


        loss.backward()


        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=2.0
        )


        optimizer.step()


        running_loss += (
            loss.item()
            * images.size(0)
        )


        predictions = torch.argmax(
            outputs,
            dim=1
        )


        correct += (
            predictions == labels
        ).sum().item()


        total += labels.size(0)


    train_loss = (
        running_loss / total
    )

    train_accuracy = (
        correct / total
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    val_loss_total = 0.0

    val_correct = 0

    val_total = 0


    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True
            )


            outputs = model(
                images
            )


            loss = criterion(
                outputs,
                labels
            )


            val_loss_total += (
                loss.item()
                * images.size(0)
            )


            predictions = torch.argmax(
                outputs,
                dim=1
            )


            val_correct += (
                predictions == labels
            ).sum().item()


            val_total += labels.size(0)


    val_loss = (
        val_loss_total /
        val_total
    )

    val_accuracy = (
        val_correct /
        val_total
    )


    # ========================================================
    # SCHEDULER
    # ========================================================

    scheduler.step(
        val_accuracy
    )


    current_lr = (
        optimizer.param_groups[0]["lr"]
    )


    # ========================================================
    # DISPLAY
    # ========================================================

    print(
        f"\nEpoch [{epoch + 1}/{EPOCHS}]"
    )

    print(
        f"Train Loss : {train_loss:.4f}"
    )

    print(
        f"Train Acc  : {train_accuracy * 100:.2f}%"
    )

    print(
        f"Val Loss   : {val_loss:.4f}"
    )

    print(
        f"Val Acc    : {val_accuracy * 100:.2f}%"
    )

    print(
        f"LR         : {current_lr:.7f}"
    )


    # ========================================================
    # BEST MODEL
    # ========================================================

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        epochs_without_improvement = 0


        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "num_classes":
                    NUM_CLASSES,

                "best_val_accuracy":
                    best_val_accuracy
            },
            MODEL_PATH
        )


        print(
            "✓ Best model saved"
        )


    else:

        epochs_without_improvement += 1


    # ========================================================
    # EARLY STOPPING
    # ========================================================

    if (
        epochs_without_improvement
        >= PATIENCE
    ):

        print()
        print(
            "Early stopping triggered."
        )

        break


# ============================================================
# LOAD BEST MODEL
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


# ============================================================
# TEST
# ============================================================

test_correct = 0

test_total = 0


# Per-class statistics
class_correct = np.zeros(
    NUM_CLASSES
)

class_total = np.zeros(
    NUM_CLASSES
)


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )


        outputs = model(
            images
        )


        predictions = torch.argmax(
            outputs,
            dim=1
        )


        test_correct += (
            predictions == labels
        ).sum().item()


        test_total += labels.size(0)


        for label, prediction in zip(
            labels.cpu().numpy(),
            predictions.cpu().numpy()
        ):

            class_total[label] += 1

            if label == prediction:

                class_correct[label] += 1


test_accuracy = (
    test_correct /
    test_total
)


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 60)
print("FINAL MODEL RESULTS")
print("=" * 60)

print(
    f"Best Validation Accuracy: "
    f"{best_val_accuracy * 100:.2f}%"
)

print(
    f"Test Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

print()
print("PER-CLASS ACCURACY")
print("-" * 60)


for index in range(
    NUM_CLASSES
):

    name = info["label"][
        str(index)
    ]

    if class_total[index] > 0:

        accuracy = (
            class_correct[index]
            /
            class_total[index]
        )

        print(
            f"{index:2d} | "
            f"{name:15s} | "
            f"{accuracy * 100:.2f}%"
        )


print()
print("=" * 60)

print(
    f"Best model saved to:"
    f" {MODEL_PATH}"
)

print("=" * 60)