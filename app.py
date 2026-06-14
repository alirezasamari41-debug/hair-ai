import streamlit as st
import cv2
import numpy as np
from insightface.app import FaceAnalysis

st.set_page_config(page_title="AI Hair Advisor", layout="centered")

st.title("💇‍♂️ AI Hair Advisor")
st.write("عکس خودت رو آپلود کن 😎")

uploaded_file = st.file_uploader("Upload Image")

if uploaded_file is not None:

    img_array = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    st.image(img, channels="BGR")

    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=0)

    faces = app.get(img)

    if len(faces) > 0:
        face = faces[0]

        x1, y1, x2, y2 = face.bbox
        ratio = (y2 - y1) / (x2 - x1)

        if ratio < 1.25:
            shape = "round"
            shape_fa = "گرد"
        elif ratio < 1.6:
            shape = "oval"
            shape_fa = "بیضی"
        else:
            shape = "long"
            shape_fa = "کشیده"

        st.success(f"Face Shape: {shape} ({shape_fa})")

        st.subheader("Hair Recommendations")

        if shape == "round":
            st.write("Quiff (کوئیف)")
            st.write("Fade (فید)")
            st.write("Side Part (فرق کج)")

        elif shape == "oval":
            st.write("Textured Crop (کات بافت‌دار)")
            st.write("Undercut (زیرتراش)")
            st.write("Most styles fit (تقریباً همه مدل‌ها)")

        else:
            st.write("Fringe (چتری)")
            st.write("Short Sides (کناره کوتاه)")
            st.write("Volume Top (حجم بالا)")

    else:
        st.error("No face detected")