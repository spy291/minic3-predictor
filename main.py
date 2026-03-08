import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="MINIC3 Predictor",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 MINIC3 智能预测系统")
st.markdown("基于机器学习的疗效与安全性双任务预测工具")

@st.cache_data
def load_data():
    np.random.seed(42)
    n = 200
    data = {
        '年龄': np.random.normal(58, 10, n).astype(int),
        '性别': np.random.choice(['男', '女'], n),
        '剂量': np.random.choice([0.3, 1.0, 3.0, 10.0], n),
        '肿瘤大小': np.random.uniform(10, 100, n),
        'ECOG': np.random.choice([0, 1, 2], n),
        '治疗线数': np.random.choice([1, 2, 3], n),
        'PDL1': np.random.choice(['阴性', '低表达', '高表达'], n),
    }
    df = pd.DataFrame(data)
    df['疗效'] = (df['剂量'] > 1) & (df['PDL1'] != '阴性')
    df['疗效'] = df['疗效'].astype(int)
    df['AE'] = ((df['剂量'] > 3) | (df['年龄'] > 70)).astype(int)
    return df

@st.cache_resource
def train_models():
    df = load_data()
    df_encoded = df.copy()
    df_encoded['性别码'] = df_encoded['性别'].map({'男':0, '女':1})
    df_encoded['PDL1码'] = df_encoded['PDL1'].map({'阴性':0, '低表达':1, '高表达':2})
    
    features = ['年龄', '性别码', '剂量', '肿瘤大小', 'ECOG', '治疗线数', 'PDL1码']
    X = df_encoded[features]
    
    X1, X2, y1, y2 = train_test_split(X, df_encoded['疗效'], test_size=0.2, random_state=42)
    model1 = RandomForestClassifier(n_estimators=100, random_state=42)
    model1.fit(X1, y1)
    acc1 = accuracy_score(y2, model1.predict(X2))
    
    X1, X2, y1, y2 = train_test_split(X, df_encoded['AE'], test_size=0.2, random_state=42)
    model2 = RandomForestClassifier(n_estimators=100, random_state=42)
    model2.fit(X1, y1)
    acc2 = accuracy_score(y2, model2.predict(X2))
    
    return model1, model2, acc1, acc2, features

model1, model2, acc1, acc2, features = train_models()

with st.sidebar:
    st.header("📋 输入患者信息")
    age = st.slider("年龄", 20, 90, 60)
    sex = st.selectbox("性别", ["男", "女"])
    dose = st.selectbox("剂量 (mg/kg)", [0.3, 1.0, 3.0, 10.0])
    tumor = st.slider("肿瘤大小 (mm)", 10, 100, 50)
    ecog = st.selectbox("ECOG", [0, 1, 2])
    lines = st.selectbox("治疗线数", [1, 2, 3])
    pdl1 = st.selectbox("PD-L1", ["阴性", "低表达", "高表达"])
    
    if st.button("🔮 开始预测", type="primary", use_container_width=True):
        input_df = pd.DataFrame([{
            '年龄': age,
            '性别码': 0 if sex == '男' else 1,
            '剂量': dose,
            '肿瘤大小': tumor,
            'ECOG': ecog,
            '治疗线数': lines,
            'PDL1码': 0 if pdl1 == '阴性' else 1 if pdl1 == '低表达' else 2
        }])
        
        prob1 = model1.predict_proba(input_df)[0][1]
        prob2 = model2.predict_proba(input_df)[0][1]
        
        st.session_state['prob1'] = prob1
        st.session_state['prob2'] = prob2
        st.session_state['predicted'] = True

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("模型准确率", f"疗效: {acc1:.0%} | AE: {acc2:.0%}")

with col2:
    if st.session_state.get('predicted'):
        st.metric("治疗有效概率", f"{st.session_state['prob1']*100:.1f}%")
        if st.session_state['prob1'] > 0.6:
            st.success("✅ 高概率有效")
        elif st.session_state['prob1'] > 0.3:
            st.warning("⚠️ 中等概率有效")
        else:
            st.error("❌ 低概率有效")

with col3:
    if st.session_state.get('predicted'):
        st.metric("不良事件风险", f"{st.session_state['prob2']*100:.1f}%")
        if st.session_state['prob2'] < 0.3:
            st.success("✅ 低风险")
        elif st.session_state['prob2'] < 0.6:
            st.warning("⚠️ 中等风险")
        else:
            st.error("❌ 高风险")

if st.session_state.get('predicted'):
    st.markdown("---")
    st.subheader("💡 治疗建议")
    if st.session_state['prob1'] > 0.5 and st.session_state['prob2'] < 0.4:
        st.success("✅ **推荐使用**：该患者适合MINIC3治疗")
    elif st.session_state['prob1'] > 0.3:
        st.warning("⚠️ **谨慎使用**：需密切监测")
    else:
        st.error("❌ **不推荐**：预期疗效不佳")
