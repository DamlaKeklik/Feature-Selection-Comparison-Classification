import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from helper import data_prep, grab_col_names
from my_fs import mutual_info_selector_auto, rf_selector_auto, spearman_selector_auto, auto_optimize_threshold_selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, auc

# Geniş ekran
st.set_page_config(layout="wide")

# Sayfa seçim menüsü
page = st.sidebar.selectbox("Sayfa Seç", ["EDA", "Model Karşılaştırma"])

# Sayfa 1: Keşifçi Veri Analizi
if page == "EDA":
    df = pd.read_csv("datasets\diabetes_binary_5050split_health_indicators_BRFSS2015.csv")

    # Değişken türlerini otomatik belirle
    cat_cols, num_cols, cat_but_car = grab_col_names(df)

    st.sidebar.title("EDA Araçları")
    eda_section = st.sidebar.radio("EDA Bölümünü Seçin", ["Genel Bilgiler", "Hedef Değişken", "Korelasyon Matrisi", "Dağılım Grafikleri", "Kategorik Dağılımlar"])

    st.title("📊 Keşifçi Veri Analizi (EDA)")

    if eda_section == "Genel Bilgiler":
        st.subheader("🔎 Veri Seti Genel Bilgileri")
        st.write(df.head())
        st.write("Veri Seti Boyutu:", df.shape)
        st.write("Eksik Değer Kontrolü:")
        st.dataframe(df.isnull().sum())

    elif eda_section == "Hedef Değişken":
        st.subheader("🎯 Hedef Değişken Dağılımı")
        target_counts = df["Diabetes_binary"].value_counts()
        st.bar_chart(target_counts)

    elif eda_section == "Korelasyon Matrisi":
        st.subheader("🔗 Korelasyon Matrisi (Spearman)")
        corr = df.corr(method="spearman")
        fig, ax = plt.subplots(figsize=(18, 12))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    elif eda_section == "Dağılım Grafikleri":
        st.subheader("📈 Sayısal Değişkenlerin Dağılımı")
        if num_cols:
            col_to_plot = st.selectbox("Değişken Seçin", num_cols)
            fig, ax = plt.subplots()
            sns.histplot(df[col_to_plot], kde=True, ax=ax)
            st.pyplot(fig)
        else:
            st.warning("Sayısal değişken bulunamadı.")

    elif eda_section == "Kategorik Dağılımlar":
        st.subheader("📊 Kategorik Değişkenlerin Dağılımı")
        if cat_cols:
            col_to_plot = st.selectbox("Kategorik Değişken Seçin", cat_cols)
            fig, ax = plt.subplots()
            df[col_to_plot].value_counts().plot(kind="bar", ax=ax)
            st.pyplot(fig)
        else:
            st.warning("Kategorik değişken bulunamadı.")

# Sayfa 2: Model Karşılaştırma
elif page == "Model Karşılaştırma":
    st.title("🚀 Feature Selection Karşılaştırma")
    st.markdown("## 🎯 Threshold Optimizasyonu ile FS Yöntemlerinin Karşılaştırılması")
    

    classifier_name = st.sidebar.selectbox(
        "🤖 Sınıflandırma Algoritması Seçin",
        ["Random Forest", "Logistic Regression"]
    )
    
    scoring_method = st.sidebar.selectbox(
        "📊 Değerlendirme Metriği",
        ["f1", "accuracy"]
    )

    run_clicked = st.sidebar.button("🚀 Optimizasyonu Başlat", type="primary")

    def get_model(name):
        if name == "Random Forest":
            return RandomForestClassifier(random_state=42, n_estimators=100)
        elif name == "Logistic Regression":
            return LogisticRegression(max_iter=1000, random_state=42)

    if run_clicked:
        with st.spinner("🔄 Otomatik threshold optimizasyonu çalışıyor..."):
            model_name = classifier_name
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()

            # --- 1. Veri Yükleme ve Hazırlık ---
            status_text.text("📂 Veri yükleniyor ve hazırlanıyor...")
            progress_bar.progress(10)
            
            df = pd.read_csv("datasets\diabetes_binary_5050split_health_indicators_BRFSS2015.csv")
            y = df["Diabetes_binary"]
            X = df.drop("Diabetes_binary", axis=1)
            X_prep, cat_cols, num_cols = data_prep(X, scale_method="standard", drop_first=True)

            results = {}
            roc_data = {}
            feature_counts = {}
            threshold_info = {}

            # --- 2. Spearman Optimizasyonu ---
            status_text.text("🔍 Spearman threshold optimizasyonu...")
            progress_bar.progress(25)
            
            spear_thresh, X_spear, spear_perf, spear_count, spear_results = auto_optimize_threshold_selector(
                spearman_selector_auto, X_prep, y, get_model(model_name), scoring=scoring_method
            )
            
            # Model performansını hesapla
            X_train, X_test, y_train, y_test = train_test_split(X_spear, y, test_size=0.3, random_state=42)
            clf = get_model(model_name)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            y_score = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(X_test)
            fpr, tpr, _ = roc_curve(y_test, y_score)
            
            results["Spearman"] = {
                "acc": accuracy_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred),
                "cm": confusion_matrix(y_test, y_pred)
            }
            roc_data["Spearman"] = (fpr, tpr, auc(fpr, tpr))
            feature_counts["Spearman"] = spear_count
            threshold_info["Spearman"] = {"threshold": spear_thresh, "total_evaluated": len(spear_results)}

            # --- 3. Random Forest Optimizasyonu ---
            status_text.text("🌳 Random Forest threshold optimizasyonu...")
            progress_bar.progress(50)
            
            rf_thresh, X_rf, rf_perf, rf_count, rf_results = auto_optimize_threshold_selector(
                rf_selector_auto, X_prep, y, get_model(model_name), scoring=scoring_method
            )
            
            X_train, X_test, y_train, y_test = train_test_split(X_rf, y, test_size=0.3, random_state=42)
            clf = get_model(model_name)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            y_score = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(X_test)
            fpr, tpr, _ = roc_curve(y_test, y_score)
            
            results["Random Forest"] = {
                "acc": accuracy_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred),
                "cm": confusion_matrix(y_test, y_pred)
            }
            roc_data["Random Forest"] = (fpr, tpr, auc(fpr, tpr))
            feature_counts["Random Forest"] = rf_count
            threshold_info["Random Forest"] = {"threshold": rf_thresh, "total_evaluated": len(rf_results)}

            # --- 4. Mutual Information Optimizasyonu ---
            status_text.text("🧮 Mutual Information threshold optimizasyonu...")
            progress_bar.progress(75)
            
            mi_thresh, X_mi, mi_perf, mi_count, mi_results = auto_optimize_threshold_selector(
                mutual_info_selector_auto, X_prep, y, get_model(model_name), scoring=scoring_method
            )
            
            X_train, X_test, y_train, y_test = train_test_split(X_mi, y, test_size=0.3, random_state=42)
            clf = get_model(model_name)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            y_score = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(X_test)
            fpr, tpr, _ = roc_curve(y_test, y_score)
            
            results["Mutual Info"] = {
                "acc": accuracy_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred),
                "cm": confusion_matrix(y_test, y_pred)
            }
            roc_data["Mutual Info"] = (fpr, tpr, auc(fpr, tpr))
            feature_counts["Mutual Info"] = mi_count
            threshold_info["Mutual Info"] = {"threshold": mi_thresh, "total_evaluated": len(mi_results)}

            progress_bar.progress(100)
            status_text.text("✅ Tüm optimizasyonlar tamamlandı!")

        st.success("🎉 Otomatik threshold optimizasyonu başarıyla tamamlandı!")
        
        # Optimal threshold ve feature count bilgilerini göster
        st.subheader("🎯Bulunan Optimal Threshold Değerleri")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "🔍 Spearman", 
                f"Threshold: {spear_thresh:.4f}", 
                f"{spear_count} özellik"
            )
            st.caption(f"📊 {threshold_info['Spearman']['total_evaluated']} threshold test edildi")
            
        with col2:
            st.metric(
                "🌳 Random Forest", 
                f"Threshold: {rf_thresh:.4f}", 
                f"{rf_count} özellik"
            )
            st.caption(f"📊 {threshold_info['Random Forest']['total_evaluated']} threshold test edildi")
            
        with col3:
            st.metric(
                "🧮 Mutual Info", 
                f"Threshold: {mi_thresh:.4f}", 
                f"{mi_count} özellik"
            )
            st.caption(f"📊 {threshold_info['Mutual Info']['total_evaluated']} threshold test edildi")

        # Feature count comparison chart
        st.subheader("📊 Otomatik Seçilen Özellik Sayıları")
        feature_df = pd.DataFrame.from_dict(feature_counts, orient='index', columns=['Feature Count'])
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(feature_df.index, feature_df['Feature Count'], 
                     color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.8)
        ax.set_ylabel('Özellik Sayısı')
        ax.set_title('Otomatik Optimizasyon ile Seçilen Özellik Sayıları')
        
        # Bar üzerinde değerleri göster
        for bar, count in zip(bars, feature_df['Feature Count']):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                   str(count), ha='center', va='bottom', fontweight='bold')
        
        st.pyplot(fig)

        # Performans karşılaştırma tablosu
        st.subheader("🏆 Performans Karşılaştırma Özeti")
        summary_data = []
        for method in results.keys():
            summary_data.append({
                'Yöntem': method,
                'Optimal Threshold': f"{threshold_info[method]['threshold']:.4f}",
                'Özellik Sayısı': feature_counts[method],
                'Accuracy': f"{results[method]['acc']:.4f}",
                'F1 Score': f"{results[method]['f1']:.4f}",
                'AUC': f"{roc_data[method][2]:.4f}",
                'Test Edilen Threshold': threshold_info[method]['total_evaluated']
            })
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)

        # ROC Curve Karşılaştırması
        st.subheader("📈 ROC Curve Karşılaştırması ")
        fig, ax = plt.subplots(figsize=(12, 8))
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        for i, (method, (fpr, tpr, roc_auc)) in enumerate(roc_data.items()):
            ax.plot(fpr, tpr, linewidth=3, color=colors[i], 
                   label=f"{method} (AUC={roc_auc:.3f}, {feature_counts[method]} features)", alpha=0.8)
        
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, linewidth=2, label="Random Classifier")
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title("ROC Curve Karşılaştırması ", fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        # Detaylı sonuçlar
        st.subheader("🔍 Detaylı Performans Analizi")
        
        # Her method için ayrı sekme
        tabs = st.tabs([f"🔍 {method}" for method in results.keys()])
        
        for i, (method, res) in enumerate(results.items()):
            with tabs[i]:
                st.markdown(f"### {method} FS Sonuçları")
                st.markdown(f"**Optimal Threshold:** `{threshold_info[method]['threshold']:.6f}`")
                st.markdown(f"**Seçilen Özellik Sayısı:** `{feature_counts[method]}/{X_prep.shape[1]}`")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("🎯 Accuracy", f"{res['acc']:.4f}")
                with col2:
                    st.metric("📊 F1 Score", f"{res['f1']:.4f}")
                
                st.markdown("**Confusion Matrix:**")
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(res["cm"], annot=True, fmt="d", cmap="Blues", ax=ax, 
                           cbar_kws={'label': 'Count'})
                ax.set_xlabel("Tahmin Edilen")
                ax.set_ylabel("Gerçek")
                ax.set_title(f"{method} - Confusion Matrix")
                st.pyplot(fig)

        # 🎯 Feature Importance Grafikleri
        st.subheader("🎯 Seçilen Özelliklerin Importance Skorları")

        importance_tabs = st.tabs([f"📊 {method}" for method in results.keys()])
        
        # 1. Spearman
        with importance_tabs[0]:
            st.markdown(f"### Spearman Feature Scores (Top {spear_count} seçildi)")
            spear_selected, spear_scores_dict = spearman_selector_auto(X_prep, y, threshold=spear_thresh)
            if spear_scores_dict:
                spear_scores_series = pd.Series(spear_scores_dict).sort_values(ascending=False)
                fig, ax = plt.subplots(figsize=(12, max(6, len(spear_scores_series) * 0.4)))
                bars = ax.barh(range(len(spear_scores_series)), spear_scores_series.values, 
                              color='#FF6B6B', alpha=0.7)
                ax.set_yticks(range(len(spear_scores_series)))
                ax.set_yticklabels(spear_scores_series.index)
                ax.set_xlabel("Mutlak Spearman Korelasyonu")
                ax.set_title(f"Spearman - Top {len(spear_scores_series)} Seçilen Özellik")
                ax.grid(True, alpha=0.3, axis='x')
                
                # Değerleri bar üzerinde göster
                for i, (bar, value) in enumerate(zip(bars, spear_scores_series.values)):
                    ax.text(value + 0.005, bar.get_y() + bar.get_height()/2, 
                           f'{value:.3f}', ha='left', va='center', fontsize=9)
                
                plt.tight_layout()
                st.pyplot(fig)

        # 2. Random Forest
        with importance_tabs[1]:
            st.markdown(f"### Random Forest Feature Scores (Top {rf_count} seçildi)")
            rf_selected, rf_scores_dict = rf_selector_auto(X_prep, y, threshold=rf_thresh)
            if rf_scores_dict:
                rf_scores_series = pd.Series(rf_scores_dict).sort_values(ascending=False)
                fig, ax = plt.subplots(figsize=(12, max(6, len(rf_scores_series) * 0.4)))
                bars = ax.barh(range(len(rf_scores_series)), rf_scores_series.values, 
                              color='#4ECDC4', alpha=0.7)
                ax.set_yticks(range(len(rf_scores_series)))
                ax.set_yticklabels(rf_scores_series.index)
                ax.set_xlabel("Özellik Önemi (Bilgi Kazancı)")
                ax.set_title(f"Random Forest - Top {len(rf_scores_series)} Seçilen Özellik")
                ax.grid(True, alpha=0.3, axis='x')
                
                # Değerleri bar üzerinde göster
                for i, (bar, value) in enumerate(zip(bars, rf_scores_series.values)):
                    ax.text(value + max(rf_scores_series.values) * 0.01, 
                           bar.get_y() + bar.get_height()/2, 
                           f'{value:.4f}', ha='left', va='center', fontsize=9)
                
                plt.tight_layout()
                st.pyplot(fig)

        # 3. Mutual Information
        with importance_tabs[2]:
            st.markdown(f"### Mutual Information Scores (Top {mi_count} seçildi)")
            mi_selected, mi_scores_dict = mutual_info_selector_auto(X_prep, y, threshold=mi_thresh)
            if mi_scores_dict:
                mi_scores_series = pd.Series(mi_scores_dict).sort_values(ascending=False)
                fig, ax = plt.subplots(figsize=(12, max(6, len(mi_scores_series) * 0.4)))
                bars = ax.barh(range(len(mi_scores_series)), mi_scores_series.values, 
                              color='#45B7D1', alpha=0.7)
                ax.set_yticks(range(len(mi_scores_series)))
                ax.set_yticklabels(mi_scores_series.index)
                ax.set_xlabel("Mutual Information Skoru")
                ax.set_title(f"Mutual Information - Top {len(mi_scores_series)} Seçilen Özellik")
                ax.grid(True, alpha=0.3, axis='x')
                
                # Değerleri bar üzerinde göster
                for i, (bar, value) in enumerate(zip(bars, mi_scores_series.values)):
                    ax.text(value + max(mi_scores_series.values) * 0.01, 
                           bar.get_y() + bar.get_height()/2, 
                           f'{value:.4f}', ha='left', va='center', fontsize=9)
                
                plt.tight_layout()
                st.pyplot(fig)

        # Optimizasyon süreci detayları
        with st.expander("🔬 Optimizasyon Süreci Detayları"):
            st.markdown("### Threshold Optimizasyon Sonuçları")
            
            # Her method için optimizasyon grafiği
            opt_tabs = st.tabs([f"📈 {method}" for method in results.keys()])
            
            all_results = [spear_results, rf_results, mi_results]
            method_names = list(results.keys())
            
            for i, (tab, method_results) in enumerate(zip(opt_tabs, all_results)):
                with tab:
                    if method_results:
                        opt_df = pd.DataFrame(method_results)
                        
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                        
                        # Performance vs Threshold
                        ax1.plot(opt_df['threshold'], opt_df['performance'], 'o-', 
                                color=colors[i], linewidth=2, markersize=6)
                        ax1.set_xlabel('Threshold')
                        ax1.set_ylabel('Performance')
                        ax1.set_title(f'{method_names[i]} - Performance vs Threshold')
                        ax1.grid(True, alpha=0.3)
                        
                        # Feature Count vs Threshold
                        ax2.plot(opt_df['threshold'], opt_df['feature_count'], 's-', 
                                color=colors[i], linewidth=2, markersize=6)
                        ax2.set_xlabel('Threshold')
                        ax2.set_ylabel('Feature Count')
                        ax2.set_title(f'{method_names[i]} - Feature Count vs Threshold')
                        ax2.grid(True, alpha=0.3)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # En iyi 5 sonucu göster
                        st.markdown("**En İyi 5 Threshold Sonucu:**")
                        top_results = opt_df.nlargest(5, 'total_score')[['threshold', 'performance', 'feature_count', 'total_score']]
                        st.dataframe(top_results.round(4))

    else:
        st.info("🚀 Threshold optimizasyonunu başlatmak için **'Başlat'** butonuna tıklayın.")
        
        st.markdown("""
        ### 🎯 Threshold Optimizasyonu Nasıl Çalışır?
        
        Bu geliştirilmiş sistem, **tamamen otomatik** olarak her feature selection yöntemi için optimal threshold değerini bulur:
        
        #### 🔍 **Multi-Objective Optimization**
        - **Performans (60% ağırlık)**: Yüksek accuracy/F1 score
        - **Efficiency (25% ağırlık)**: Az özellik sayısı kullanımı  
        - **Balance (15% ağırlık)**: Performance/Feature oranı optimizasyonu
        
        #### 🎛️ **Yöntem-Özel Threshold Aralıkları**
        - **Spearman**: 0.01 - 0.45 (30 farklı değer)
        - **Mutual Info**: 0.0001 - 0.04 (27 farklı değer)
        - **Random Forest**: 0.0001 - 0.04 (27 farklı değer)
        
        #### 🛡️ **Robust Error Handling**
        - Hata durumlarında otomatik fallback
        - Hiç özellik seçilmezse minimum threshold kullanımı
        - Tüm uç durumlar için güvenli çözümler
        
        #### 🏆 **Avantajlar**
        - ✅ Manuel parametre ayarı gerekmez
        - ✅ Her dataset için otomatik optimizasyon
        - ✅ Overfitting riskini minimize eder
        - ✅ Hesaplama verimliliği artışı
        - ✅ Yorumlanabilir model sonuçları
        """)
        
        st.markdown("---")
        st.markdown("💡 **İpucu:** Farklı sınıflandırıcılar ve değerlendirme metrikleri ile deneme yaparak en iyi kombinasyonu bulabilirsiniz!")