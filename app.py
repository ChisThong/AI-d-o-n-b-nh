#!/usr/bin/env python3
"""
Streamlit app for disease prediction
"""

import streamlit as st
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from model import DiseasePredictor

# Configure page
st.set_page_config(
    page_title="AI Dự đoán Bệnh",
    page_icon="🩺",
    layout="centered"
)

@st.cache_resource
def load_predictor():
    """Load the trained model"""
    predictor = DiseasePredictor()
    if predictor.load():
        return predictor
    return None

def main():
    st.title("🩺 AI Dự đoán Bệnh từ Triệu chứng")
    st.markdown("""
        *Lưu ý: Kết quả chỉ mang tính chất tham khảo, không thay thế chẩn đoán của bác sĩ chuyên khoa.*
    """)

    # Load model
    predictor = load_predictor()

    if predictor is None:
        st.error("❌ Chưa tìm thấy mô hình! Vui lòng chạy `python train.py` để huấn luyện AI trước.")
        return

    st.subheader("Nhập các triệu chứng bạn đang gặp phải:")
    st.markdown("Nhập mỗi triệu chứng cách nhau bởi dấu phẩy, ví dụ: `đau đầu, sốt, mệt mỏi`")

    with st.form(key="symptom_form"):
        symptom_input = st.text_area(
            "Triệu chứng",
            height=120,
            placeholder="đau đầu, sốt, mệt mỏi, buồn nôn..."
        )
        submit_button = st.form_submit_button("🔍 Dự đoán Bệnh")

    if submit_button:
        # Process input
        symptoms = [sym.strip().lower() for sym in symptom_input.split(',') if sym.strip()]

        if len(symptoms) == 0:
            st.warning("⚠️ Vui lòng nhập ít nhất 1 triệu chứng để AI có thể dự đoán.")
            return

        with st.spinner("🧠 AI đang phân tích dữ liệu..."):
            try:
                results = predictor.predict_top_k(symptoms, top_k=3)

                st.success("✅ Phân tích hoàn tất!")
                st.markdown("---")
                st.markdown("### 🛑 Các dự đoán có khả năng nhất")

                for index, item in enumerate(results, start=1):
                    percent = item['probability'] * 100
                    st.markdown(f"**{index}. {item['disease']}** — {percent:.1f}%")
                    if item['description']:
                        st.info(f"**📋 Mô tả:** {item['description']}")
                    else:
                        st.info("Không có mô tả chi tiết cho dự đoán này.")

                if len(results) == 0:
                    st.warning("Không có dự đoán. Vui lòng kiểm tra lại các triệu chứng đã nhập.")

                recognized = [sym for sym in symptoms if sym in [s.lower() for s in predictor.symptoms_list]]
                unrecognized = [sym for sym in symptoms if sym not in [s.lower() for s in predictor.symptoms_list]]

                if recognized:
                    st.markdown(f"**✅ Triệu chứng được nhận diện:** {', '.join(recognized)}")

                if unrecognized:
                    st.warning(f"⚠️ Triệu chứng chưa được hiểu: {', '.join(unrecognized)}. Vui lòng nhập chính xác hoặc dùng từ tương tự.")

            except Exception as e:
                st.error(f"❌ Lỗi khi dự đoán: {e}")

if __name__ == "__main__":
    main()
