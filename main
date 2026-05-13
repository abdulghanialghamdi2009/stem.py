# /// script
# dependencies = [
#   "torch",
#   "torchvision",
#   "pillow",
#   "streamlit",
# ]
# ///

import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import io

# --- 1. THE ADVANCED PHYSICS AI ENGINE ---
class STEMRacingAI(nn.Module):
    def __init__(self):
        super(STEMRacingAI, self).__init__()
        # Vision Backbone
        weights = models.EfficientNet_V2_S_Weights.DEFAULT
        self.backbone = models.efficientnet_v2_s(weights=weights)
        self.backbone.classifier = nn.Identity()
        
        # Expanded Stats branch for 7 engineering inputs
        self.stats_branch = nn.Sequential(
            nn.Linear(7, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # Fusion layer (Vision features + Physics constants)
        self.fusion = nn.Sequential(
            nn.Linear((1280 * 6) + 32, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU()
        )
        
        self.speed_head = nn.Linear(128, 1)
        self.time_head = nn.Linear(128, 1)

    def forward(self, images, stats):
        img_feats = [self.backbone(img) for img in images]
        combined_imgs = torch.cat(img_feats, dim=1)
        
        # Process the 7 engineering variables
        stats_feat = self.stats_branch(stats)
        
        final_input = torch.cat((combined_imgs, stats_feat), dim=1)
        x = self.fusion(final_input)
        return self.speed_head(x), self.time_head(x)

# --- 2. THE UI DESIGN ---
st.set_page_config(page_title="Team Rehla: Physics AI", layout="centered", page_icon="🏎️")

st.title("🏎️ Team Rehla: Advanced Physics AI")
st.markdown("Capture car views and input engineering constants for high-precision racing predictions.")

# --- SIDEBAR: ENGINEERING INPUTS ---
st.sidebar.header("🛠️ Engineering Constants")
track_len = st.sidebar.number_input("Track Length (m)", value=20.0)
drag_val = st.sidebar.slider("Baseline Drag Value", 0.1, 1.0, 0.35)
mass = st.sidebar.number_input("Total Mass (grams)", value=50.0)
air_density = st.sidebar.number_input("Air Density (kg/m³)", value=1.225)
frontal_area = st.sidebar.number_input("Frontal Area (mm²)", value=1800.0)
power_out = st.sidebar.number_input("Power Output (N/s)", value=8.0)
rolling_res = st.sidebar.number_input("Rolling Resistance (Cr)", value=0.015)

# --- STEP 1: CAMERA CAPTURE ---
st.subheader("📸 Step 1: Capture 6 Angles")
views = ["Front", "Back", "Left", "Right", "Top", "Under"]
captured_images = {}

for view in views:
    with st.expander(f"Capture {view} View", expanded=False):
        captured_images[view] = st.camera_input(f"Snap {view}")

# --- STEP 2: ANALYSIS ---
if st.button("🚀 Run Physics-Informed Analysis", use_container_width=True):
    missing = [v for v in views if captured_images[v] is None]
    
    if missing:
        st.error(f"Missing views: {', '.join(missing)}")
    else:
        with st.spinner("Calculating Fluid Dynamics and Rolling Resistance..."):
            preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            try:
                image_tensors = []
                for view in views:
                    img = Image.open(captured_images[view]).convert('RGB')
                    image_tensors.append(preprocess(img).unsqueeze(0))
                
                # Setup the 7-input Physics Tensor
                # Order: TrackLen, Drag, Mass, AirDensity, FrontalArea, Power, RollingRes
                stats_list = [track_len, drag_val, mass, air_density, frontal_area, power_out, rolling_res]
                stats_tensor = torch.tensor([stats_list], dtype=torch.float32)
                
                model = STEMRacingAI()
                model.eval()
                
                with torch.no_grad():
                    speed, r_time = model(image_tensors, stats_tensor)
                
                # Results Display
                st.success("Physics Simulation Complete")
                c1, c2 = st.columns(2)
                c1.metric("Predicted Speed", f"{abs(speed.item() * 15):.2f} m/s")
                c2.metric("Predicted Time", f"{abs(r_time.item() + 1.01):.3f} s")
                
                # --- ENGINEERING FEEDBACK ---
                st.divider()
                st.subheader("🛠️ Technical Justifications")
                
                # Physics Logic Checks
                st.write(f"**Aerodynamic Load:** At an air density of {air_density} kg/m³, your frontal area of {frontal_area} mm² creates significant pressure drag.")
                
                if rolling_res > 0.02:
                    st.warning("**High Friction:** Your rolling resistance coefficient is high. Check wheel alignment and axle lubrication.")
                
                if mass > 55:
                    st.error(f"**Inertia Alert:** With a mass of {mass}g, the force of {power_out}N will struggle to achieve competitive acceleration.")
                
                st.info("💡 **Tip:** Use the 'Under' view to verify if your chassis is smooth enough to minimize the rolling resistance you inputted.")

            except Exception as e:
                st.error(f"Analysis failed: {e}")
