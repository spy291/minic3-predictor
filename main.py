import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="MINIC3智能预测系统",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 MINIC3抗CTLA-4迷你抗体智能预测系统")
st.markdown("**基于机器学习的疗效与安全性双任务预测工具**")

@st.cache_data
def generate_enhanced_data():
    """生成模拟数据"""
    np.random.seed(42)
    n_patients = 200

    data = {
        '患者ID': [f'P{str(i).zfill(3)}' for i in range(1, n_patients + 1)],
        '剂量水平(mg/kg)': np.random.choice([0.3, 1.0, 3.0, 10.0], n_patients, p=[0.2, 0.3, 0.3, 0.2]),
        '年龄': np.random.normal(58, 10, n_patients).astype(int),
        '性别': np.random.choice(['男', '女'], n_patients, p=[0.55, 0.45]),
        '基线肿瘤大小(mm)': np.random.uniform(10, 100, n_patients),
        'ECOG评分': np.random.choice([0, 1, 2], n_patients, p=[0.3, 0.5, 0.2]),
        '既往治疗线数': np.random.choice([1, 2, 3], n_patients, p=[0.5, 0.3, 0.2]),
        'PD-L1表达': np.random.choice(['阴性', '低表达', '高表达'], n_patients, p=[0.4, 0.4, 0.2]),
        '肿瘤类型': np.random.choice(['肺癌', '乳腺癌', '结直肠癌', '胃癌', '肝癌'], n_patients),
    }

    df = pd.DataFrame(data)

    def calculate_response(row):
        base_prob = 0.3
        dose_effect = {0.3: -0.1, 1.0: 0.0, 3.0: 0.15, 10.0: 0.2}
        pdl1_effect = {'阴性': -0.1, '低表达': 0.05, '高表达': 0.2}
        tumor_effect = -0.002 * row['基线肿瘤大小(mm)']
        ecog_effect = -0.1 * row['ECOG评分']
        response_prob = base_prob + dose_effect[row['剂量水平(mg/kg)']] + pdl1_effect[row['PD-L1表达']] + tumor_effect + ecog_effect
        response_prob = max(0.05, min(0.8, response_prob))
        return np.random.binomial(1, response_prob)

    def calculate_ae(row):
        base_ae_prob = 0.4
        dose_ae_effect = {0.3: -0.2, 1.0: -0.1, 3.0: 0.1, 10.0: 0.3}
        age_effect = 0.005 * (row['年龄'] - 50)
        ae_prob = base_ae_prob + dose_ae_effect[row['剂量水平(mg/kg)']] + age_effect
        ae_prob = max(0.1, min(0.9, ae_prob))
        return np.random.binomial(1, ae_prob)

    df['是否缓解'] = df.apply(calculate_response, axis=1)
    df['是否发生AE'] = df.apply(calculate_ae, axis=1)
    df['肿瘤缓解状态'] = df['是否缓解'].map({1: np.random.choice(['完全缓解', '部分缓解'], p=[0.3, 0.7]),
                                             0: np.random.choice(['疾病稳定', '疾病进展'], p=[0.6, 0.4])})
    
    df['不良事件(AE)'] = df['是否发生AE'].map({1: '有不良事件', 0: '无'})

    return df.drop(['是否缓解', '是否发生AE'], axis=1)

class MINIC3PredictiveModel:
    def __init__(self):
        self.model_ae = None
        self.model_response = None
        self.feature_columns = None

    def prepare_features(self, df):
        feature_df = df.copy()
        feature_df['性别编码'] = feature_df['性别'].map({'男': 0, '女': 1})
        feature_df['PD-L1编码'] = feature_df['PD-L1表达'].map({'阴性': 0, '低表达': 1, '高表达': 2})
        self.feature_columns = ['剂量水平(mg/kg)', '年龄', '性别编码', '基线肿瘤大小(mm)',
                                'ECOG评分', '既往治疗线数', 'PD-L1编码']
        return feature_df[self.feature_columns]

    def prepare_targets(self, df):
        y_ae = (df['不良事件(AE)'] != '无').astype(int)
        y_response = df['肿瘤缓解状态'].isin(['完全缓解', '部分缓解']).astype(int)
        return y_ae, y_response

    def train(self, df):
        X = self.prepare_features(df)
        y_ae, y_response = self.prepare_targets(df)

        X_train, X_test, y_ae_train, y_ae_test = train_test_split(X, y_ae, test_size=0.2, random_state=42)
        _, _, y_response_train, y_response_test = train_test_split(X, y_response, test_size=0.2, random_state=42)

        self.model_ae = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model_ae.fit(X_train, y_ae_train)

        self.model_response = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model_response.fit(X_train, y_response_train)

        ae_acc = accuracy_score(y_ae_test, self.model_ae.predict(X_test))
        response_acc = accuracy_score(y_response_test, self.model_response.predict(X_test))
        
        return ae_acc, response_acc

    def predict_patient(self, patient_features):
        ae_prob = self.model_ae.predict_proba(patient_features)[0][1]
        response_prob = self.model_response.predict_proba(patient_features)[0][1]
        return ae_prob, response_prob

if 'model' not in st.session_state:
    st.session_state.model = MINIC3PredictiveModel()
    df = generate_enhanced_data()
    st.session_state.df = df
    with st.spinner('训练预测模型中...'):
        ae_acc, response_acc = st.session_state.model.train(df)
        st.success(f'模型训练完成！不良事件预测准确率：{ae_acc:.2f}，疗效预测准确率：{response_acc:.2f}')
else:
    df = st.session_state.df

st.sidebar.title("导航菜单")
page = st.sidebar.radio("选择功能", ["数据概览", "智能预测"])

if page == "数据概览":
    st.header("📊 数据集概览")
    st.write(f"数据集大小：{len(df)} 名患者")
    st.dataframe(df.head(10))
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("总患者数", len(df))
    with col2:
        orr = len(df[df['肿瘤缓解状态'].isin(['完全缓解', '部分缓解'])]) / len(df) * 100
        st.metric("总体有效率", f"{orr:.1f}%")
    with col3:
        ae_rate = len(df[df['不良事件(AE)'] != '无']) / len(df) * 100
        st.metric("总体AE率", f"{ae_rate:.1f}%")

elif page == "智能预测":
    st.header("🎯 智能预测系统")
    
    with st.expander("📝 输入患者信息", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            dose = st.selectbox("剂量水平 (mg/kg)", [0.3, 1.0, 3.0, 10.0])
            age = st.slider("年龄", 30, 80, 58)
            gender = st.selectbox("性别", ["男", "女"])
            tumor_size = st.slider("基线肿瘤大小 (mm)", 10, 100, 50)

        with col2:
            ecog = st.selectbox("ECOG评分", [0, 1, 2])
            prev_treatment = st.selectbox("既往治疗线数", [1, 2, 3])
            pdl1 = st.selectbox("PD-L1表达", ["阴性", "低表达", "高表达"])

    if st.button("开始预测", type="primary", use_container_width=True):
        input_data = pd.DataFrame([{
            '剂量水平(mg/kg)': dose, '年龄': age, '性别': gender,
            '基线肿瘤大小(mm)': tumor_size, 'ECOG评分': ecog,
            '既往治疗线数': prev_treatment, 'PD-L1表达': pdl1
        }])

        input_encoded = st.session_state.model.prepare_features(input_data)
        ae_prob, response_prob = st.session_state.model.predict_patient(input_encoded)

        st.session_state['prob1'] = response_prob
        st.session_state['prob2'] = ae_prob
        st.session_state['predicted'] = True

    if st.session_state.get('predicted'):
        st.markdown("---")
        st.subheader("📊 预测结果")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("治疗有效概率", f"{st.session_state['prob1']*100:.1f}%")
            if st.session_state['prob1'] > 0.6:
                st.success("✅ 高概率有效")
            elif st.session_state['prob1'] > 0.3:
                st.warning("⚠️ 中等概率有效")
            else:
                st.error("❌ 低概率有效")

        with col2:
            st.metric("不良事件风险", f"{st.session_state['prob2']*100:.1f}%")
            if st.session_state['prob2'] < 0.3:
                st.success("✅ 低风险")
            elif st.session_state['prob2'] < 0.6:
                st.warning("⚠️ 中等风险")
            else:
                st.error("❌ 高风险")

        st.subheader("💡 治疗建议")
        if st.session_state['prob1'] > 0.5 and st.session_state['prob2'] < 0.4:
            st.success("✅ **推荐使用**：该患者适合MINIC3治疗")
        elif st.session_state['prob1'] > 0.3:
            st.warning("⚠️ **谨慎使用**：需密切监测")
        else:
            st.error("❌ **不推荐**：预期疗效不佳")
