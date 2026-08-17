import os
import numpy as np
import streamlit as st
import plotly.graph_objects as go

import torch
import torch.nn as nn

import nibabel as nib
import medmnist
from medmnist import INFO

from scipy.ndimage import gaussian_filter
from skimage import measure


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Medical 3D AI Viewer",
    page_icon="🧬",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🧬 AI-Powered 3D Medical Scan Viewer")

st.markdown(
    """
    **Interactive 3D medical visualization with AI organ classification**
    """
)


# ============================================================
# ORGAN LABELS
# ============================================================

ORGAN_LABELS = {
    0: "liver",
    1: "kidney-right",
    2: "kidney-left",
    3: "femur-right",
    4: "femur-left",
    5: "bladder",
    6: "heart",
    7: "lung-right",
    8: "lung-left",
    9: "spleen",
    10: "pancreas"
}


# ============================================================
# 3D CNN
# ============================================================

class Organ3DCNN(nn.Module):

    def __init__(self, num_classes=11):

        super().__init__()

        self.features = nn.Sequential(

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
# LOAD AI MODEL
# ============================================================

@st.cache_resource
def load_ai_model():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = Organ3DCNN(
        num_classes=11
    )


    checkpoint = torch.load(
        "models/organ3d_cnn_best.pth",
        map_location=device
    )


    model.load_state_dict(
        checkpoint["model_state_dict"]
    )


    model.to(device)

    model.eval()

    return model, device


# ============================================================
# LOAD MEDMNIST
# ============================================================

@st.cache_data
def load_medmnist():

    data_flag = "organmnist3d"

    info = INFO[data_flag]

    DataClass = getattr(
        medmnist,
        info["python_class"]
    )

    dataset = DataClass(
        split="train",
        download=True
    )

    return dataset


# ============================================================
# LOAD NIFTI
# ============================================================

def load_nifti(uploaded_file):

    temp_path = "temp_scan.nii.gz"

    with open(
        temp_path,
        "wb"
    ) as f:

        f.write(
            uploaded_file.getbuffer()
        )


    nii = nib.load(
        temp_path
    )

    volume = nii.get_fdata()

    return np.asarray(
        volume
    )


# ============================================================
# CREATE 3D MESH
# ============================================================

def create_mesh(
    volume,
    threshold
):

    volume = volume.astype(
        np.float32
    )


    minimum = volume.min()

    maximum = volume.max()


    if maximum > minimum:

        volume = (
            volume - minimum
        ) / (
            maximum - minimum
        )


    volume = gaussian_filter(
        volume,
        sigma=1
    )


    verts, faces, _, _ = (
        measure.marching_cubes(
            volume,
            level=threshold
        )
    )


    return verts, faces


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "⚙️ Scan Controls"
)


source = st.sidebar.radio(
    "Scan Source",
    [
        "OrganMNIST3D",
        "Upload Scan"
    ]
)


# ============================================================
# DATASET SCAN
# ============================================================

volume = None

scan_name = None


if source == "OrganMNIST3D":

    dataset = load_medmnist()


    scan_index = st.sidebar.slider(
        "Scan Number",
        0,
        len(dataset) - 1,
        0
    )


    volume = dataset[
        scan_index
    ][0]


    volume = np.asarray(
        volume
    )


    volume = np.squeeze(
        volume
    )


    scan_name = (
        f"OrganMNIST3D Scan {scan_index}"
    )


# ============================================================
# UPLOAD SCAN
# ============================================================

else:

    uploaded_file = st.sidebar.file_uploader(
        "Upload medical scan",
        type=[
            "nii",
            "nii.gz",
            "npy"
        ]
    )


    if uploaded_file:

        try:

            if uploaded_file.name.endswith(
                ".npy"
            ):

                volume = np.load(
                    uploaded_file
                )

            else:

                volume = load_nifti(
                    uploaded_file
                )


            volume = np.asarray(
                volume
            )


            volume = np.squeeze(
                volume
            )


            scan_name = (
                uploaded_file.name
            )


        except Exception as e:

            st.error(
                f"Unable to load scan: {e}"
            )

            st.stop()


# ============================================================
# VALIDATE
# ============================================================

if volume is None:

    st.info(
        "Select a scan to begin."
    )

    st.stop()


if volume.ndim != 3:

    st.error(
        f"Expected 3D volume. "
        f"Received shape {volume.shape}"
    )

    st.stop()


# ============================================================
# SCAN INFO
# ============================================================

st.subheader(
    "📊 Scan Information"
)


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Scan",
        scan_name
    )


with col2:

    st.metric(
        "Width",
        volume.shape[2]
    )


with col3:

    st.metric(
        "Height",
        volume.shape[1]
    )


with col4:

    st.metric(
        "Depth",
        volume.shape[0]
    )


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3 = st.tabs(
    [
        "🎚️ Slice Viewer",
        "🧊 3D Viewer",
        "🧠 AI Analysis"
    ]
)


# ============================================================
# TAB 1 — SLICE VIEWER
# ============================================================

with tab1:

    st.header(
        "🎚️ Interactive Slice Viewer"
    )


    axis = st.selectbox(
        "Viewing Plane",
        [
            "Axial",
            "Coronal",
            "Sagittal"
        ]
    )


    if axis == "Axial":

        max_index = (
            volume.shape[2] - 1
        )

        index = st.slider(
            "Axial Slice",
            0,
            max_index,
            max_index // 2
        )

        image = volume[
            :,
            :,
            index
        ]


    elif axis == "Coronal":

        max_index = (
            volume.shape[1] - 1
        )

        index = st.slider(
            "Coronal Slice",
            0,
            max_index,
            max_index // 2
        )

        image = volume[
            :,
            index,
            :
        ]


    else:

        max_index = (
            volume.shape[0] - 1
        )

        index = st.slider(
            "Sagittal Slice",
            0,
            max_index,
            max_index // 2
        )

        image = volume[
            index,
            :,
            :
        ]


    st.image(
        image,
        caption=f"{axis} Slice {index}",
        clamp=True,
        width="stretch"
    )


# ============================================================
# TAB 2 — 3D VIEWER
# ============================================================

with tab2:

    st.header(
        "🧊 Interactive 3D Reconstruction"
    )


    threshold = st.slider(
        "Surface Threshold",
        0.01,
        0.99,
        0.50,
        0.01
    )


    if st.button(
        "🔄 Generate 3D Model"
    ):

        with st.spinner(
            "Generating 3D model..."
        ):

            try:

                verts, faces = create_mesh(
                    volume,
                    threshold
                )


                x, y, z = verts.T

                i, j, k = faces.T


                mesh = go.Mesh3d(

                    x=x,
                    y=y,
                    z=z,

                    i=i,
                    j=j,
                    k=k,

                    color="lightblue",

                    opacity=0.85,

                    lighting=dict(
                        ambient=0.35,
                        diffuse=0.8,
                        specular=0.5,
                        roughness=0.4
                    ),

                    lightposition=dict(
                        x=100,
                        y=100,
                        z=100
                    )
                )


                fig = go.Figure(
                    data=[mesh]
                )


                fig.update_layout(

                    title=(
                        "Interactive 3D "
                        "Medical Scan"
                    ),

                    scene=dict(
                        aspectmode="data"
                    ),

                    height=700,

                    margin=dict(
                        l=0,
                        r=0,
                        t=50,
                        b=0
                    )
                )


                st.plotly_chart(
                    fig,
                    use_container_width=True
                )


                st.success(
                    f"3D model generated: "
                    f"{len(verts):,} vertices"
                )


            except Exception as e:

                st.error(
                    f"3D generation failed: {e}"
                )


# ============================================================
# TAB 3 — AI ANALYSIS
# ============================================================

with tab3:

    st.header(
        "🧠 AI Organ Classification"
    )


    st.write(
        """
        A trained 3D CNN analyzes the current
        volume and predicts the represented organ.
        """
    )


    if not os.path.exists(
        "models/organ3d_cnn_best.pth"
    ):

        st.error(
            "Trained model not found."
        )

        st.stop()


    if st.button(
        "🧠 Analyze Current Scan"
    ):

        with st.spinner(
            "Running 3D CNN..."
        ):

            model, device = (
                load_ai_model()
            )


            # ------------------------------------------
            # PREPROCESS
            # ------------------------------------------

            ai_volume = (
                volume.astype(
                    np.float32
                )
            )


            # Normalize
            minimum = (
                ai_volume.min()
            )

            maximum = (
                ai_volume.max()
            )


            if maximum > minimum:

                ai_volume = (
                    ai_volume - minimum
                ) / (
                    maximum - minimum
                )


            # ------------------------------------------
            # TENSOR
            # ------------------------------------------

            tensor = torch.from_numpy(
                ai_volume
            ).float()


            tensor = tensor.unsqueeze(
                0
            ).unsqueeze(
                0
            )


            tensor = tensor.to(
                device
            )


            # ------------------------------------------
            # PREDICTION
            # ------------------------------------------

            with torch.no_grad():

                output = model(
                    tensor
                )


                probabilities = torch.softmax(
                    output,
                    dim=1
                )


            predicted_class = torch.argmax(
                probabilities,
                dim=1
            ).item()


            confidence = probabilities[
                0,
                predicted_class
            ].item()


            predicted_organ = (
                ORGAN_LABELS[
                    predicted_class
                ]
            )


        # ------------------------------------------
        # RESULT
        # ------------------------------------------

        st.success(
            f"Predicted Organ: "
            f"{predicted_organ.upper()}"
        )


        col1, col2 = st.columns(2)


        with col1:

            st.metric(
                "Prediction",
                predicted_organ.upper()
            )


        with col2:

            st.metric(
                "Confidence",
                f"{confidence * 100:.2f}%"
            )


        # ------------------------------------------
        # PROBABILITIES
        # ------------------------------------------

        st.subheader(
            "📊 Class Probabilities"
        )


        probabilities_np = (
            probabilities[
                0
            ]
            .cpu()
            .numpy()
        )


        probability_dict = {
            ORGAN_LABELS[i]:
                float(probabilities_np[i])
            for i in range(11)
        }


        st.bar_chart(
            probability_dict
        )


        st.caption(
            "This is an educational/research "
            "classification model, not a clinical "
            "diagnostic system."
        )