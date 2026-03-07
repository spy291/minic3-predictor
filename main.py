"""
MINIC3 Predictor - 完整版（带图表）
"""

# ===================== 强制安装依赖（黑科技） =====================
import sys
import subprocess
import pkg_resources

# 需要安装的包列表
required_packages = {
    'streamlit': '1.28.0',
    'pandas': '2.0.3',
    'numpy': '1.24.3',
    'scikit-learn': '1.3.0',
    'matplotlib': '3.7.2',
    'seaborn': '0.12.2',
    'joblib': '1.3.1',
    'pillow': '10.0.0'
}

# 检查并安装缺失的包
for package, version in required_packages.items():
    try:
        pkg_resources.get_distribution(f"{package}=={version}")
    except:
        subprocess.check_call([sys.executable, "-m", "pip", "install", f"{package}=={version}", "--quiet"])

# ===================== 导入依赖 =====================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免GUI错误
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix
import joblib
import warnings
import time
from datetime import datetime

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']  # 使用英文字体避免中文乱码
plt.rcParams['axes.unicode_minus'] = False

# ===================== 页面配置 =====================
st.set_page_config(
    page_title="MINIC3智能预测系统",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 MINIC3抗CTLA-4迷你抗体智能预测系统")
st.markdown("**基于机器学习的疗效与安全性双任务预测工具**")

# ===================== 生成模拟数据 =====================
@st.cache_data
def generate_enhanced_data():
    """生成真实的模拟数据"""
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
        '治疗周期': np.random.poisson(6, n_patients) + 1,
    }

    df = pd.DataFrame(data)

    # 基于特征生成真实的治疗结果
    def calculate_response(row):
        base_prob = 0.3
        dose_effect = {0.3: -0.1, 1.0: 0.0, 3.0: 0.15, 10.0: 0.2}
        pdl1_effect = {'阴性': -0.1, '低表达': 0.05, '高表达': 0.2}
        tumor_effect = -0.002 * row['基线肿瘤大小(mm)']
        ecog_effect = -0.1 * row['ECOG评分']

        response_prob = (base_prob + dose_effect[row['剂量水平(mg/kg)']] +
                         pdl1_effect[row['PD-L1表达']] + tumor_effect + ecog_effect)
        response_prob = max(0.05, min(0.8, response_prob))

        return np.random.binomial(1, response_prob)

    def calculate_ae(row):
        base_ae_prob = 0.4
        dose_ae_effect = {0.3: -0.2, 1.0: -0.1, 3.0: 0.1, 10.0: 0.3}
        age_effect = 0.005 * (row['年龄'] - 50)

        ae_prob = base_ae_prob + dose_ae_effect[row['剂量水平(mg/kg)']] + age_effect
        ae_prob = max(0.1, min(0.9, ae_prob))

        return np.random.binomial(1, ae_prob)

    # 生成治疗结果
    df['是否缓解'] = df.apply(calculate_response, axis=1)
    df['是否发生AE'] = df.apply(calculate_ae, axis=1)

    # 映射到临床结果
    df['肿瘤缓解状态'] = df['是否缓解'].map({1: np.random.choice(['完全缓解', '部分缓解'], p=[0.3, 0.7]),
                                             0: np.random.choice(['疾病稳定', '疾病进展'], p=[0.6, 0.4])})

    ae_severity = {1: np.random.choice(['1级腹泻', '1-2级皮疹', '2级转氨酶升高', '2级乏力'], p=[0.3, 0.3, 0.2, 0.2]),
                   0: '无'}
    df['不良事件(AE)'] = df['是否发生AE'].map(ae_severity)

    return df.drop(['是否缓解', '是否发生AE'], axis=1)

# ===================== 机器学习模型训练 =====================
class MINIC3PredictiveModel:
    def __init__(self):
        self.model_ae = None
        self.model_response = None
        self.feature_columns = None
        self.roc_curve_data = None

    def prepare_features(self, df):
        """准备特征数据"""
        feature_df = df.copy()

        # 编码分类变量
        feature_df['性别编码'] = feature_df['性别'].map({'男': 0, '女': 1})
        feature_df['PD-L1编码'] = feature_df['PD-L1表达'].map({'阴性': 0, '低表达': 1, '高表达': 2})
        feature_df['肿瘤类型编码'] = pd.Categorical(feature_df['肿瘤类型']).codes

        self.feature_columns = ['剂量水平(mg/kg)', '年龄', '性别编码', '基线肿瘤大小(mm)',
                                'ECOG评分', '既往治疗线数', 'PD-L1编码', '肿瘤类型编码']

        return feature_df[self.feature_columns]

    def prepare_targets(self, df):
        """准备目标变量"""
        y_ae = (df['不良事件(AE)'] != '无').astype(int)
        y_response = df['肿瘤缓解状态'].isin(['完全缓解', '部分缓解']).astype(int)
        return y_ae, y_response

    def train(self, df):
        """训练模型"""
        st.info("🚀 开始训练预测模型...")

        X = self.prepare_features(df)
        y_ae, y_response = self.prepare_targets(df)

        X_train, X_test, y_ae_train, y_ae_test = train_test_split(X, y_ae, test_size=0.2, random_state=42)
        _, _, y_response_train, y_response_test = train_test_split(X, y_response, test_size=0.2, random_state=42)

        # 训练不良事件预测模型
        self.model_ae = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model_ae.fit(X_train, y_ae_train)

        # 训练治疗反应预测模型
        self.model_response = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model_response.fit(X_train, y_response_train)

        # 评估模型
        ae_accuracy = accuracy_score(y_ae_test, self.model_ae.predict(X_test))
        response_accuracy = accuracy_score(y_response_test, self.model_response.predict(X_test))
        
        # 计算ROC数据
        y_ae_prob = self.model_ae.predict_proba(X_test)[:, 1]
        y_response_prob = self.model_response.predict_proba(X_test)[:, 1]
        
        ae_fpr, ae_tpr, _ = roc_curve(y_ae_test, y_ae_prob)
        response_fpr, response_tpr, _ = roc_curve(y_response_test, y_response_prob)
        
        ae_auc = roc_auc_score(y_ae_test, y_ae_prob)
        response_auc = roc_auc_score(y_response_test, y_response_prob)
        
        self.roc_curve_data = {
            'ae': {'fpr': ae_fpr, 'tpr': ae_tpr, 'auc': ae_auc},
            'response': {'fpr': response_fpr, 'tpr': response_tpr, 'auc': response_auc}
        }

        return ae_accuracy, response_accuracy

    def predict_patient(self, patient_features):
        """预测单个患者"""
        if self.model_ae is None or self.model_response is None:
            raise ValueError("模型尚未训练")

        ae_prob = self.model_ae.predict_proba(patient_features)[0][1]
        response_prob = self.model_response.predict_proba(patient_features)[0][1]

        return ae_prob, response_prob

    def get_feature_importance(self):
        """获取特征重要性"""
        if self.model_response is None:
            return None

        importance_df = pd.DataFrame({
            '特征': self.feature_columns,
            '重要性': self.model_response.feature_importances_
        }).sort_values('重要性', ascending=False)

        return importance_df

# ===================== 初始化 =====================
if 'model' not in st.session_state:
    st.session_state.model = MINIC3PredictiveModel()
    df = generate_enhanced_data()
    st.session_state.df = df
    with st.spinner('训练预测模型中...'):
        ae_acc, response_acc = st.session_state.model.train(df)
        st.success(f'模型训练完成！不良事件预测准确率：{ae_acc:.2f}，疗效预测准确率：{response_acc:.2f}')
else:
    df = st.session_state.df

# ===================== 侧边栏导航 =====================
st.sidebar.title("导航菜单")
page = st.sidebar.radio("选择功能", [
    "数据概览",
    "智能预测",
    "模型分析",
    "生存分析"
])

# ===================== 数据概览 =====================
if page == "数据概览":
    st.header("📊 数据集概览")
    st.write(f"数据集大小：{len(df)} 名患者")
    st.dataframe(df.head(10))

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("总患者数", len(df))
    with col2:
        orr = len(df[df['肿瘤缓解状态'].isin(['完全缓解', '部分缓解'])]) / len(df) * 100
        st.metric("总体ORR", f"{orr:.1f}%")
    with col3:
        ae_rate = len(df[df['不良事件(AE)'] != '无']) / len(df) * 100
        st.metric("总体AE率", f"{ae_rate:.1f}%")

# ===================== 智能预测 =====================
elif page == "智能预测":
    st.header("🎯 智能预测系统")
    
    tab1, tab2 = st.tabs(["单个患者预测", "批量预测"])

    with tab1:
        st.subheader("单个患者预后预测")

        col1, col2 = st.columns(2)

        with col1:
            dose = st.selectbox("剂量水平(mg/kg)", [0.3, 1.0, 3.0, 10.0])
            age = st.slider("年龄", 30, 80, 58)
            gender = st.selectbox("性别", ["男", "女"])
            tumor_size = st.slider("基线肿瘤大小(mm)", 10, 100, 50)

        with col2:
            ecog = st.selectbox("ECOG评分", [0, 1, 2], format_func=lambda x: f"{x}分")
            prev_treatment = st.selectbox("既往治疗线数", [1, 2, 3])
            pdl1 = st.selectbox("PD-L1表达", ["阴性", "低表达", "高表达"])
            cancer_type = st.selectbox("肿瘤类型", ["肺癌", "乳腺癌", "结直肠癌", "胃癌", "肝癌"])

        if st.button("开始预测", type="primary"):
            input_data = pd.DataFrame([{
                '剂量水平(mg/kg)': dose,
                '年龄': age,
                '性别': gender,
                '基线肿瘤大小(mm)': tumor_size,
                'ECOG评分': ecog,
                '既往治疗线数': prev_treatment,
                'PD-L1表达': pdl1,
                '肿瘤类型': cancer_type
            }])

            input_encoded = st.session_state.model.prepare_features(input_data)
            ae_prob, response_prob = st.session_state.model.predict_patient(input_encoded)

            col1, col2 = st.columns(2)

            with col1:
                st.metric("治疗有效概率", f"{response_prob * 100:.1f}%")
                if response_prob > 0.6:
                    st.success("✅ 高概率有效")
                elif response_prob > 0.3:
                    st.warning("⚠️ 中等概率有效")
                else:
                    st.error("❌ 低概率有效")

            with col2:
                st.metric("不良事件风险", f"{ae_prob * 100:.1f}%")
                if ae_prob < 0.3:
                    st.success("✅ 低风险")
                elif ae_prob < 0.6:
                    st.warning("⚠️ 中等风险")
                else:
                    st.error("❌ 高风险")

            st.subheader("💡 治疗建议")
            if response_prob > 0.5 and ae_prob < 0.4:
                st.success("**推荐治疗方案**：该患者适合使用MINIC3治疗，预期疗效好且安全性可控")
            elif response_prob > 0.3:
                st.warning("**谨慎使用**：疗效预期一般，需密切监测")
            else:
                st.error("**不推荐**：预期疗效不佳，建议考虑其他治疗方案")

    with tab2:
        st.subheader("批量患者预测")
        uploaded_file = st.file_uploader("上传CSV文件", type=['csv'])

        if uploaded_file is not None:
            batch_data = pd.read_csv(uploaded_file)
            st.write("上传数据预览：")
            st.dataframe(batch_data.head())

            if st.button("执行批量预测"):
                batch_encoded = st.session_state.model.prepare_features(batch_data)
                ae_probs = st.session_state.model.model_ae.predict_proba(batch_encoded)[:, 1]
                response_probs = st.session_state.model.model_response.predict_proba(batch_encoded)[:, 1]

                results_df = batch_data.copy()
                results_df['有效概率(%)'] = (response_probs * 100).round(1)
                results_df['AE风险(%)'] = (ae_probs * 100).round(1)
                results_df['推荐等级'] = np.where(
                    (response_probs > 0.5) & (ae_probs < 0.4), '推荐',
                    np.where(response_probs > 0.3, '谨慎', '不推荐')
                )

                st.success("✅ 批量预测完成")
                st.dataframe(results_df)

                csv = results_df.to_csv(index=False)
                st.download_button("下载预测结果", csv, "predictions.csv", "text/csv")

# ===================== 模型分析 =====================
elif page == "模型分析":
    st.header("🔬 模型性能分析")

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 ROC曲线")
        if st.session_state.model.roc_curve_data:
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # 疗效ROC
            response_data = st.session_state.model.roc_curve_data['response']
            ax.plot(response_data['fpr'], response_data['tpr'], 
                   label=f'疗效预测 (AUC = {response_data["auc"]:.2f})', linewidth=2)
            
            # AE ROC
            ae_data = st.session_state.model.roc_curve_data['ae']
            ax.plot(ae_data['fpr'], ae_data['tpr'], 
                   label=f'AE预测 (AUC = {ae_data["auc"]:.2f})', linewidth=2)
            
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
            ax.set_xlabel('假阳性率 (1-特异性)')
            ax.set_ylabel('真阳性率 (敏感性)')
            ax.set_title('ROC曲线')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            plt.close(fig)

    with col2:
        st.subheader("📊 特征重要性")
        importance_df = st.session_state.model.get_feature_importance()
        if importance_df is not None:
            fig, ax = plt.subplots(figsize=(8, 6))
            colors = plt.cm.viridis(np.linspace(0, 1, len(importance_df)))
            ax.barh(importance_df['特征'], importance_df['重要性'], color=colors)
            ax.set_xlabel('重要性')
            ax.set_title('特征重要性排名')
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3, axis='x')
            
            st.pyplot(fig)
            plt.close(fig)
    
    st.subheader("📋 关键发现")
    st.info("""
    - **剂量水平**和**PD-L1表达**是预测疗效的最重要因素
    - **基线肿瘤大小**和**ECOG评分**显著影响治疗结果
    - **年龄**和**性别**的影响相对较小但仍有参考价值
    """)

# ===================== 生存分析 =====================
elif page == "生存分析":
    st.header("📈 生存分析")
    
    # 模拟生存数据
    df_survival = df.copy()
    np.random.seed(42)
    df_survival['PFS_月'] = np.where(
        df_survival['肿瘤缓解状态'].isin(['完全缓解', '部分缓解']),
        np.random.normal(8, 2, len(df_survival)),
        np.random.normal(4, 1.5, len(df_survival))
    )
    df_survival['PFS_月'] = np.maximum(1, df_survival['PFS_月'])

    # 绘制生存曲线
    fig, ax = plt.subplots(figsize=(10, 6))

    for dose in sorted(df_survival['剂量水平(mg/kg)'].unique()):
        dose_data = df_survival[df_survival['剂量水平(mg/kg)'] == dose]
        time_points = np.sort(dose_data['PFS_月'].unique())
        survival_prob = []

        for t in time_points:
            at_risk = len(dose_data[dose_data['PFS_月'] >= t])
            events = len(dose_data[dose_data['PFS_月'] == t])
            if at_risk > 0:
                prob = (at_risk - events) / at_risk
                survival_prob.append(prob)
            else:
                survival_prob.append(0)

        cum_survival = np.cumprod(survival_prob)
        ax.step(time_points, cum_survival, where='post', label=f'{dose} mg/kg', linewidth=2)

    ax.set_xlabel('时间（月）')
    ax.set_ylabel('无进展生存率')
    ax.set_title('各剂量组无进展生存曲线')
    ax.legend()
    ax.grid(True, alpha=0.3)

    st.pyplot(fig)
    plt.close(fig)

    st.subheader("中位PFS")
    for dose in sorted(df_survival['剂量水平(mg/kg)'].unique()):
        median_pfs = df_survival[df_survival['剂量水平(mg/kg)'] == dose]['PFS_月'].median()
        st.write(f"- {dose} mg/kg组：中位PFS = {median_pfs:.1f} 月")
