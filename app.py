import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

try:
    import seaborn as sns
except ImportError:
    sns = None

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

st.set_page_config(page_title="CDSS - Heart Disease", page_icon="🫀", layout="wide")

st.markdown("""
    <style>
    .main {background-color: #fcfcfc;}
    div.stButton > button:first-child {
        background-color: #e63946;
        color: white;
        border-radius: 8px;
        font-weight: bold;
        width: 100%;
    }
    div.stButton > button:first-child:hover {
        background-color: #d90429;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Sistem Pendukung Keputusan Klinis (CDSS)")
st.markdown("**Deteksi Dini Risiko Penyakit Jantung Terintegrasi Machine Learning**")
st.divider()

@st.cache_data
def load_and_clean_data():
    try:
        df = pd.read_csv("heart.csv")
        df = df.drop(columns=['id', 'dataset'])
        df['target'] = df['num'].apply(lambda x: 1 if x > 0 else 0)
        df = df.drop(columns=['num'])
        df = df.rename(columns={'thalch': 'thalach'})
        
        mapping = {
            'sex': {'Male': 1, 'Female': 0},
            'cp': {'typical angina': 0, 'atypical angina': 1, 'non-anginal': 2, 'asymptomatic': 3},
            'fbs': {True: 1, False: 0},
            'restecg': {'normal': 0, 'st-t abnormality': 1, 'lv hypertrophy': 2},
            'exang': {True: 1, False: 0},
            'slope': {'upsloping': 0, 'flat': 1, 'downsloping': 2},
            'thal': {'normal': 1, 'fixed defect': 2, 'reversable defect': 3}
        }
        df = df.replace(mapping)
        df = df.fillna(df.median())
        return df
    except FileNotFoundError:
        st.error("File 'heart.csv' tidak ditemukan.")
        return None

df = load_and_clean_data()

if df is not None:
    st.sidebar.title("Navigasi CDSS")
    menu = st.sidebar.radio(
        "", 
        ["Exploratory Data (EDA)", "Komparasi Model", "Live Skrining", "Prediksi Massal (CSV)"]
    )
    st.sidebar.divider()

    X = df.drop(columns=['target'])
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    @st.cache_resource
    def train_models():
        rf = RandomForestClassifier(random_state=42)
        rf.fit(X_train_scaled, y_train)
        
        svm = SVC(probability=True, random_state=42)
        svm.fit(X_train_scaled, y_train)
        return rf, svm

    model_rf, model_svm = train_models()

    if menu == "Exploratory Data (EDA)":
        st.header("Exploratory Data Analysis")
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Distribusi Target")
            fig1, ax1 = plt.subplots(figsize=(5, 6))
            if sns is not None:
                sns.countplot(data=df, x='target', palette='Reds', ax=ax1)
                ax1.set_xticklabels(['Sehat (0)', 'Berisiko (1)'])
            else:
                counts = df['target'].value_counts().sort_index()
                labels = ['Sehat (0)', 'Berisiko (1)']
                values = [counts.get(0, 0), counts.get(1, 0)]
                ax1.bar(labels, values, color=['#f4a7a7', '#e63946'])
            ax1.set_xlabel('Status Risiko')
            ax1.set_ylabel('Jumlah')
            st.pyplot(fig1)
            
        with col2:
            st.subheader("Scatter Plot: Umur vs Detak Jantung Maks.")
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            if sns is not None:
                sns.scatterplot(
                    data=df,
                    x='age',
                    y='thalach',
                    hue='target',
                    palette=['#457b9d', '#e63946'],
                    alpha=0.7,
                    ax=ax2
                )
                handles, labels = ax2.get_legend_handles_labels()
                ax2.legend(handles=handles, labels=['Sehat (0)', 'Berisiko (1)'], title="Status Risiko")
            else:
                color_map = {0: '#457b9d', 1: '#e63946'}
                for status, color in color_map.items():
                    subset = df[df['target'] == status]
                    ax2.scatter(subset['age'], subset['thalach'], c=color, alpha=0.7, label='Sehat (0)' if status == 0 else 'Berisiko (1)')
                ax2.legend(title="Status Risiko")
            ax2.set_xlabel("Umur (Tahun)")
            ax2.set_ylabel("Detak Jantung Maksimal (bpm)")
            st.pyplot(fig2)
            
        st.divider()
        st.subheader("Dataset Bersih")
        st.dataframe(df.head(15), use_container_width=True)

    elif menu == "Komparasi Model":
        st.header("Evaluasi & Signifikansi Fitur")
        
        def evaluate_model(model, X_test, y_test):
            y_pred = model.predict(X_test)
            return {
                "Akurasi": accuracy_score(y_test, y_pred),
                "Presisi": precision_score(y_test, y_pred),
                "Recall": recall_score(y_test, y_pred),
                "F1-Score": f1_score(y_test, y_pred)
            }
            
        metrics_rf = evaluate_model(model_rf, X_test_scaled, y_test)
        metrics_svm = evaluate_model(model_svm, X_test_scaled, y_test)
        
        # Membagi layar: Kiri (1 bagian) untuk metrik, Kanan (2.5 bagian) untuk grafik
        col_kiri, col_kanan = st.columns([1, 2.5])
        
        with col_kiri:
            st.subheader("Skor Model")
            st.markdown("**Random Forest**")
            st.metric("F1-Score", f"{metrics_rf['F1-Score']:.4f}")
            st.metric("Akurasi", f"{metrics_rf['Akurasi']:.4f}")
            
            st.divider()
            
            st.markdown("**Support Vector Machine**")
            st.metric("F1-Score", f"{metrics_svm['F1-Score']:.4f}")
            st.metric("Akurasi", f"{metrics_svm['Akurasi']:.4f}")

        with col_kanan:
            st.subheader("Feature Importance (Random Forest)")
            
            kamus_fitur = {
                'age': 'Umur', 'sex': 'Jenis Kelamin', 'cp': 'Tipe Nyeri Dada',
                'trestbps': 'Tekanan Darah', 'chol': 'Kolesterol', 'fbs': 'Gula Darah Puasa',
                'restecg': 'Hasil EKG', 'thalach': 'Detak Jantung Maks.', 'exang': 'Nyeri Dada Olahraga',
                'oldpeak': 'Depresi ST', 'slope': 'Kemiringan Segmen ST', 'ca': 'Jml. Pembuluh Tersumbat',
                'thal': 'Kelainan Aliran Darah'
            }
            
            importance = model_rf.feature_importances_
            feat_df = pd.DataFrame({"Fitur": X.columns, "Bobot": importance})
            feat_df['Fitur Lengkap'] = feat_df['Fitur'].map(kamus_fitur)
            feat_df = feat_df.sort_values(by="Bobot", ascending=False)
            
            # Mengatur ukuran figure agar tingginya pas dengan tumpukan metrik di kiri
            fig_feat, ax_feat = plt.subplots(figsize=(8, 6.5))
            if sns is not None:
                sns.barplot(data=feat_df, x="Bobot", y="Fitur Lengkap", palette="Reds_r", ax=ax_feat)
            else:
                ax_feat.barh(feat_df['Fitur Lengkap'], feat_df['Bobot'], color='#e63946')
                ax_feat.invert_yaxis()
            ax_feat.set_ylabel("")
            st.pyplot(fig_feat)
        
        with st.expander("Lihat Penjelasan Detail Istilah Medis"):
            st.markdown("""
            * **Jml. Pembuluh Tersumbat (ca):** Jumlah arteri utama yang menyempit.
            * **Tipe Nyeri Dada (cp):** Klasifikasi nyeri.
            * **Detak Jantung Maks. (thalach):** Batas detak jantung saat aktivitas berat.
            * **Depresi ST (oldpeak):** Penurunan kurva EKG saat beraktivitas.
            * **Kelainan Aliran Darah (thal):** Seberapa merata darah mengaliri otot jantung.
            """)

    elif menu == "Live Skrining":
        st.header("Form Skrining Medis")
        
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                age = st.number_input("Umur", min_value=1, max_value=120, value=50)
                sex = st.selectbox("Jenis Kelamin", [1, 0], format_func=lambda x: "Pria" if x==1 else "Wanita")
                cp = st.selectbox("Tipe Nyeri Dada", [0, 1, 2, 3])
                trestbps = st.number_input("Tekanan Darah", min_value=50, max_value=250, value=120)
                chol = st.number_input("Kolesterol", min_value=100, max_value=600, value=200)
            with col2:
                fbs = st.selectbox("Gula Darah Puasa > 120", [1, 0], format_func=lambda x: "Ya" if x==1 else "Tidak")
                restecg = st.selectbox("Hasil ECG", [0, 1, 2])
                thalach = st.number_input("Detak Jantung Maksimal", min_value=50, max_value=250, value=150)
                exang = st.selectbox("Nyeri Dada saat Olahraga", [1, 0], format_func=lambda x: "Ya" if x==1 else "Tidak")
            with col3:
                oldpeak = st.number_input("Depresi ST", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
                slope = st.selectbox("Slope ST Segmen", [0, 1, 2])
                ca = st.selectbox("Jml. Pembuluh Darah (0-3)", [0, 1, 2, 3])
                thal = st.selectbox("Thalassemia", [1, 2, 3])
                
            submit = st.form_submit_button("Jalankan Prediksi AI")
            
        if submit:
            input_data = pd.DataFrame({
                'age': [age], 'sex': [sex], 'cp': [cp], 'trestbps': [trestbps], 'chol': [chol],
                'fbs': [fbs], 'restecg': [restecg], 'thalach': [thalach], 'exang': [exang],
                'oldpeak': [oldpeak], 'slope': [slope], 'ca': [ca], 'thal': [thal]
            })
            
            input_scaled = scaler.transform(input_data)
            prediction = model_rf.predict(input_scaled)
            probability = model_rf.predict_proba(input_scaled)[0][1] * 100
            
            st.divider()
            if prediction[0] == 1:
                st.error(f" **STATUS: PASIEN BERISIKO TINGGI** (Probabilitas: {probability:.1f}%)")
            else:
                st.success(f" **STATUS: PASIEN AMAN** (Probabilitas: {probability:.1f}%)")

    elif menu == "Prediksi Massal (CSV)":
        st.header("Prediksi Risiko Massal")
        st.info("Unggah file CSV berisi data rekam medis pasien. Pastikan format kolom sesuai dengan template.")
        
        template_df = pd.DataFrame(columns=X.columns)
        csv_template = template_df.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="Unduh Template CSV",
            data=csv_template,
            file_name='template_prediksi.csv',
            mime='text/csv',
        )
        
        uploaded_file = st.file_uploader("Upload Data Pasien (CSV)", type=["csv"])
        
        if uploaded_file is not None:
            try:
                data_pred = pd.read_csv(uploaded_file)
                st.write("Preview Data:")
                st.dataframe(data_pred.head())
                
                if st.button("Proses Prediksi"):
                    if list(data_pred.columns) != list(X.columns):
                        st.error("Error: Nama kolom pada file tidak sesuai dengan template!")
                    else:
                        data_scaled = scaler.transform(data_pred)
                        predictions = model_rf.predict(data_scaled)
                        
                        data_pred['Hasil Prediksi'] = predictions
                        data_pred['Status'] = data_pred['Hasil Prediksi'].apply(lambda x: "Berisiko" if x == 1 else "Aman")
                        
                        st.success("Prediksi berhasil diselesaikan!")
                        st.dataframe(data_pred[['age', 'sex', 'Hasil Prediksi', 'Status']].head(10))
                        
                        result_csv = data_pred.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Unduh Hasil Prediksi Lengkap",
                            data=result_csv,
                            file_name='hasil_prediksi_massal.csv',
                            mime='text/csv',
                        )
            except Exception as e:
                st.error(f"Terjadi kesalahan saat membaca file: {e}")
