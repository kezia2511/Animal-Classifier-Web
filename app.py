import streamlit as st
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
from sklearn.preprocessing import LabelEncoder

# Page config
st.set_page_config(
    page_title="Animal Classifier",
    page_icon="🐾",
    layout="centered"
)

# Custom CSS
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .title { text-align: center; color: #2c3e50; }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Model
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pooling = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear((128*16*16), 256)
        self.dropout2 = nn.Dropout(0.4)
        self.output = nn.Linear(256, 3)

    def forward(self, x):
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.pooling(x)
        x = self.conv2(x); x = self.bn2(x); x = self.relu(x); x = self.pooling(x); x = self.dropout(x)
        x = self.conv3(x); x = self.bn3(x); x = self.relu(x); x = self.pooling(x)
        x = self.flatten(x); x = self.linear(x); x = self.relu(x); x = self.dropout2(x)
        return self.output(x)

# Load model
model = Net().to(device)
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.eval()

# Label encoder
label_encoder = LabelEncoder()
label_encoder.fit(['cat', 'dog', 'wild'])

# Transform
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float)
])

# Emoji per kelas
emoji_map = {'cat': '🐱', 'dog': '🐶', 'wild': '🐾'}
color_map = {'cat': '#ff6b6b', 'dog': '#4ecdc4', 'wild': '#45b7d1'}

# UI
st.markdown("<h1 class='title'>🐾 Animal Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center'>Upload gambar hewan untuk mengetahui jenisnya!</p>", unsafe_allow_html=True)
st.divider()

uploaded_file = st.file_uploader("📁 Pilih gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Gambar yang diupload", width=300)
    with col2:
        st.markdown("### Hasil Analisis")
        with st.spinner("Menganalisis gambar..."):
            img_tensor = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(img_tensor)
                predicted = torch.argmax(output, axis=1).item()
                result = label_encoder.inverse_transform([predicted])[0]
        
        emoji = emoji_map[result]
        color = color_map[result]
        
        st.markdown(f"""
            <div class='result-box' style='background-color:{color}; color:white;'>
                {emoji} {result.upper()}
            </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        st.markdown("**Kemungkinan kelas:**")
        probs = torch.softmax(output, dim=1)[0]
        classes = ['cat', 'dog', 'wild']
        for i, cls in enumerate(classes):
            st.progress(float(probs[i]), text=f"{emoji_map[cls]} {cls}: {probs[i]*100:.1f}%")