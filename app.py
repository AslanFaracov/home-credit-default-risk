import streamlit as st

import pandas as pd

import joblib

import shap
import matplotlib.pyplot as plt

import plotly.graph_objects as go

import plotly.express as px





# =====================================================

# PAGE CONFIG

# =====================================================



st.set_page_config(

    page_title="Home Credit Risk",

    page_icon="🏦",

    layout="wide",

    initial_sidebar_state="expanded"

)





# =====================================================

# CUSTOM CSS

# =====================================================



st.markdown("""

<style>



    /* ---------- GLOBAL ---------- */



    .main {

        background-color: #f7f9fc;

    }



    .block-container {

        padding-top: 2rem;

        padding-bottom: 3rem;

    }





    /* ---------- HEADER ---------- */



    .main-header {

        font-size: 2.4rem;

        font-weight: 800;

        color: #172033;

        margin-bottom: 0.1rem;

    }



    .sub-header {

        color: #697386;

        font-size: 1rem;

        margin-top: 0;

    }





    /* ---------- SECTION ---------- */



    .section-title {

        font-size: 1.35rem;

        font-weight: 700;

        color: #172033;

        margin-top: 1rem;

        margin-bottom: 0.7rem;

    }





    /* ---------- METRIC CARDS ---------- */



    div[data-testid="stMetric"] {

        background: white;

        border: 1px solid #e7ebf2;

        border-radius: 14px;

        padding: 18px;

        box-shadow: 0 2px 8px rgba(23, 32, 51, 0.04);

    }



    div[data-testid="stMetricLabel"] {

        color: #697386;

        font-weight: 600;

    }



    div[data-testid="stMetricValue"] {

        color: #172033;

        font-weight: 750;

    }





    /* ---------- RISK CARD ---------- */



    .risk-card {

        padding: 1.2rem 1.5rem;

        border-radius: 14px;

        margin: 1rem 0;

        font-size: 1.15rem;

        font-weight: 700;

    }



    .risk-low {

        background-color: #e8f7ee;

        color: #126b3a;

        border-left: 6px solid #20a05a;

    }



    .risk-medium {

        background-color: #fff7df;

        color: #8a6500;

        border-left: 6px solid #e5ad00;

    }



    .risk-high {

        background-color: #fdebec;

        color: #a32935;

        border-left: 6px solid #dc3545;

    }





    /* ---------- INFO CARD ---------- */



    .info-card {

        background: white;

        border: 1px solid #e7ebf2;

        border-radius: 14px;

        padding: 20px;

        margin-bottom: 15px;

        box-shadow: 0 2px 8px rgba(23, 32, 51, 0.04);

    }





    /* ---------- SIDEBAR ---------- */



    section[data-testid="stSidebar"] {

        background-color: #172033;

    }



    section[data-testid="stSidebar"] * {

        color: #f3f5f8;

    }





    /* ---------- BUTTON ---------- */



    .stButton > button {

        border-radius: 9px;

        font-weight: 650;

    }





    /* ---------- DATAFRAME ---------- */



    div[data-testid="stDataFrame"] {

        border-radius: 12px;

        overflow: hidden;

    }



</style>

""", unsafe_allow_html=True)





# =====================================================

# LOAD MODEL

# =====================================================



@st.cache_resource

def load_model():



    try:

        return joblib.load("final_model.pkl")



    except FileNotFoundError:

        st.error("❌ final_model.pkl faylı tapılmadı.")

        st.stop()



    except Exception as e:

        st.error(f"❌ Model yüklənərkən xəta baş verdi: {e}")

        st.stop()





# =====================================================

# =====================================================
# LOAD SHAP EXPLAINER
# =====================================================

# LOAD DATA

# =====================================================



@st.cache_data

def load_sample_data():



    try:

        return pd.read_csv("deployment_sample.csv")



    except FileNotFoundError:

        st.error("❌ deployment_sample.csv faylı tapılmadı.")

        st.stop()



    except Exception as e:

        st.error(f"❌ Dataset yüklənərkən xəta baş verdi: {e}")

        st.stop()





model = load_model()

sample_df = load_sample_data()

# SHAP explainer — direct XGBoost model, without Streamlit caching
try:
    xgb_model = model.named_steps["model"]
    explainer = shap.TreeExplainer(xgb_model)
except Exception as e:
    st.error(f"❌ SHAP explainer yaradılarkən xəta baş verdi: {e}")
    st.stop()





# =====================================================

# CLEAN DEPLOYMENT DATA

# =====================================================



# TARGET deployment zamanı modelə verilməməlidir

model_features_df = sample_df.drop(

    columns=["TARGET"],

    errors="ignore"

)





# =====================================================

# SESSION STATE

# =====================================================



if "result_df" not in st.session_state:

    st.session_state.result_df = None



if "batch_completed" not in st.session_state:

    st.session_state.batch_completed = False





# =====================================================

# HELPER FUNCTIONS

# =====================================================



def get_risk_category(proba):



    if proba < 0.20:

        return "Aşağı risk", "risk-low", "#20a05a"



    elif proba < 0.50:

        return "Orta risk", "risk-medium", "#e5ad00"



    else:

        return "Yüksək risk", "risk-high", "#dc3545"






def get_friendly_feature_name(feature_name):
    """Convert technical model feature names into simple business language."""
    name = str(feature_name)

    for prefix in ("numeric__", "ordinal__", "onehot__"):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break

    exact_map = {
        "min_refused_credit": "Rədd edilmiş kreditlər üzrə minimum məbləğ",
        "ext_source_mean": "Xarici kredit göstəricilərinin orta səviyyəsi",
        "credit_goods_ratio": "Kredit məbləğinin malın dəyərinə nisbəti",
        "amt_annuity": "Annuitet ödənişi",
        "ext_source_3": "Xarici kredit göstəricisi 3",
        "ext_source_2": "Xarici kredit göstəricisi 2",
        "ext_source_1": "Xarici kredit göstəricisi 1",
        "name_education_type": "Təhsil səviyyəsi",
        "region_population_relative": "Yaşayış bölgəsinin əhali göstəricisi",
        "total_payment_amount": "Ümumi ödəniş məbləği",
        "organization_type": "İş sahəsi",
        "income_total": "Ümumi gəlir",
        "credit_income_ratio": "Kredit məbləğinin gəlirə nisbəti",
        "annuity_income_ratio": "Annuitet ödənişinin gəlirə nisbəti",
        "days_birth": "Müştərinin yaşı",
        "days_employed": "İş stajı",
        "family_size": "Ailə üzvlərinin sayı",
        "children_count": "Uşaqların sayı",
    }

    if name in exact_map:
        return exact_map[name]

    category_prefixes = (
        ("organization_type_", "İş sahəsi"),
        ("name_education_type_", "Təhsil səviyyəsi"),
        ("gender_", "Cins"),
        ("contract_type_", "Müqavilə növü"),
        ("family_status_", "Ailə vəziyyəti"),
        ("housing_type_", "Yaşayış növü"),
        ("income_type_", "Gəlir mənbəyi"),
    )

    for prefix, label in category_prefixes:
        if name.startswith(prefix):
            category = name[len(prefix):].replace("_", " ")
            return f"{label}: {category}"

    readable = name.replace("_", " ").strip()
    return readable[:1].upper() + readable[1:]


def get_impact_level(shap_value):
    """Show an easy-to-read impact level instead of technical SHAP values."""
    magnitude = abs(float(shap_value))

    if magnitude >= 0.50:
        return "Çox yüksək təsir"
    elif magnitude >= 0.20:
        return "Yüksək təsir"
    elif magnitude >= 0.10:
        return "Orta təsir"
    return "Aşağı təsir"


def get_business_factors(shap_result, feature_names, max_items=4):
    """Return the main risk-increasing and risk-reducing factors."""
    values = shap_result.values[0] if getattr(shap_result.values, "ndim", 0) > 1 else shap_result.values

    if getattr(values, "ndim", 0) > 1:
        values = values[:, 0]

    rows = [
        {"feature": str(feature_names[i]), "value": float(value)}
        for i, value in enumerate(values)
    ]

    increasing = sorted(
        [r for r in rows if r["value"] > 0],
        key=lambda r: abs(r["value"]),
        reverse=True
    )[:max_items]

    decreasing = sorted(
        [r for r in rows if r["value"] < 0],
        key=lambda r: abs(r["value"]),
        reverse=True
    )[:max_items]

    def enrich(items):
        return [
            {
                "Faktor": get_friendly_feature_name(item["feature"]),
                "Təsir": get_impact_level(item["value"]),
            }
            for item in items
        ]

    return enrich(increasing), enrich(decreasing)

def create_gauge(proba, color):



    fig = go.Figure(

        go.Indicator(

            mode="gauge+number",

            value=proba * 100,



            number={

                "suffix": "%",

                "font": {

                    "size": 38,

                    "color": "#172033"

                }

            },



            title={

                "text": "Default ehtimalı",

                "font": {

                    "size": 18,

                    "color": "#697386"

                }

            },



            gauge={

                "axis": {

                    "range": [0, 100],

                    "ticksuffix": "%",

                    "tickfont": {

                        "color": "#697386"

                    }

                },



                "bar": {

                    "color": color,

                    "thickness": 0.65

                },



                "bgcolor": "#f1f3f7",



                "steps": [

                    {

                        "range": [0, 20],

                        "color": "#e8f7ee"

                    },

                    {

                        "range": [20, 50],

                        "color": "#fff7df"

                    },

                    {

                        "range": [50, 100],

                        "color": "#fdebec"

                    }

                ],



                "threshold": {

                    "line": {

                        "color": "#172033",

                        "width": 3

                    },

                    "thickness": 0.75,

                    "value": proba * 100

                }

            }

        )

    )



    fig.update_layout(

        height=310,

        margin=dict(

            l=25,

            r=25,

            t=55,

            b=10

        ),

        paper_bgcolor="white",

        font={

            "family": "Arial"

        }

    )



    return fig





# =====================================================

# SIDEBAR

# =====================================================



with st.sidebar:



    st.markdown("## 🏦 Home Credit")



    st.caption(

        "Credit Risk Assessment Platform"

    )



    st.divider()



    st.markdown("### 📌 Sistem")



    st.write(

        "Müştərilərin kredit defolt ehtimalını "

        "ML modeli vasitəsilə qiymətləndirir."

    )



    st.divider()



    st.markdown("### 🚦 Risk səviyyələri")



    st.markdown(

        "🟢 **Aşağı**  \n"

        "`0% – 20%`"

    )



    st.markdown(

        "🟡 **Orta**  \n"

        "`20% – 50%`"

    )



    st.markdown(

        "🔴 **Yüksək**  \n"

        "`50%+`"

    )



    st.divider()



    st.markdown("### 📊 Dataset")



    st.metric(

        "Müştəri sayı",

        f"{len(sample_df):,}"

    )



    st.metric(

        "Feature sayı",

        f"{model_features_df.shape[1]:,}"

    )



    st.divider()



    st.caption(

        "Home Credit Default Risk Project"

    )





# =====================================================

# HEADER

# =====================================================



st.markdown(

    '<div class="main-header">🏦 Home Credit Default Risk</div>',

    unsafe_allow_html=True

)



st.markdown(

    '<div class="sub-header">'

    'Machine Learning əsaslı kredit risk qiymətləndirmə sistemi'

    '</div>',

    unsafe_allow_html=True

)



st.divider()





# =====================================================

# TABS

# =====================================================



tab1, tab2, tab3 = st.tabs(

    [

        "🔍 Müştəri Analizi",

        "📊 Portfolio Analizi",

        "📋 Dataset"

    ]

)





# =====================================================

# TAB 1 — SINGLE CUSTOMER

# =====================================================



with tab1:



    st.markdown(

        '<div class="section-title">'

        'Müştəri Risk Qiymətləndirməsi'

        '</div>',

        unsafe_allow_html=True

    )



    col_select, col_preview = st.columns(

        [1, 2]

    )



    # -------------------------------------------------

    # CUSTOMER SELECTION

    # -------------------------------------------------



    with col_select:



        st.markdown("#### 🔎 Müştəri seç")



        row_index = st.selectbox(

            "Müştəri ID",

            sample_df.index,

            format_func=lambda x: f"Müştəri #{x}"

        )



        selected_row = model_features_df.loc[[row_index]]



        predict_clicked = st.button(

            "🔍 Risk qiymətləndir",

            use_container_width=True,

            type="primary"

        )



    # -------------------------------------------------

    # CUSTOMER PREVIEW

    # -------------------------------------------------



    with col_preview:



        st.markdown("#### 👤 Müştəri məlumatları")



        preview = selected_row.T.reset_index()



        preview.columns = [

            "Feature",

            "Value"

        ]



        st.dataframe(

            preview,

            use_container_width=True,

            height=210,

            hide_index=True

        )





    # =================================================

    # PREDICTION

    # =================================================



    if predict_clicked:



        with st.spinner("Model müştərini qiymətləndirir..."):



            try:

                proba = model.predict_proba(
                    selected_row
                )[:, 1][0]

                # SHAP üçün preprocessing
                X_transformed = model.named_steps[
                    "preprocess"
                ].transform(selected_row)

                # Sparse matrix-i yalnız SHAP üçün dense formata çeviririk
                if hasattr(X_transformed, "toarray"):
                    X_shap_input = X_transformed.toarray()
                else:
                    X_shap_input = X_transformed

                # SHAP explanation
                shap_result = explainer(X_shap_input)

                # Preprocessing sonrası feature adları
                feature_names = model.named_steps[
                    "preprocess"
                ].get_feature_names_out()

                shap_result.feature_names = feature_names



            except Exception as e:



                st.error(

                    f"❌ Prediction zamanı xəta baş verdi: {e}"

                )

                st.stop()





        risk, risk_class, color = get_risk_category(

            proba

        )



        st.divider()





        # -------------------------------------------------

        # RISK RESULT

        # -------------------------------------------------



        st.markdown(

            f"""

            <div class="risk-card {risk_class}">

                ⚠️ Risk qiymətləndirməsi:

                <strong>{risk}</strong>

                — Default ehtimalı: {proba:.2%}

            </div>

            """,

            unsafe_allow_html=True

        )





        # -------------------------------------------------

        # GAUGE + METRICS

        # -------------------------------------------------



        col_gauge, col_metrics = st.columns(

            [1.15, 1]

        )





        with col_gauge:



            fig = create_gauge(

                proba,

                color

            )



            st.plotly_chart(

                fig,

                use_container_width=True

            )





        with col_metrics:



            st.markdown("#### 📌 Risk göstəriciləri")



            m1, m2 = st.columns(2)



            m1.metric(

                "Default ehtimalı",

                f"{proba:.2%}"

            )



            m2.metric(

                "Risk səviyyəsi",

                risk

            )



            m3, m4 = st.columns(2)



            m3.metric(

                "Müştəri ID",

                str(row_index)

            )



            m4.metric(

                "Model statusu",

                "Active"

            )





            st.write("")





            if proba < 0.20:



                st.success("Bu müştərinin risk səviyyəsi aşağıdır.")



            elif proba < 0.50:



                st.warning("Bu müştərinin risk səviyyəsi ortadır.")



            else:



                st.error("Bu müştərinin risk səviyyəsi yüksəkdir.")





        # -------------------------------------------------

        # BUSINESS-FRIENDLY RISK EXPLANATION
        # -------------------------------------------------

        st.divider()

        st.markdown("#### 🔍 Risk qərarının izahı")

        st.write(
            "Aşağıdakı məlumatlar modelin bu müştəri üçün verdiyi "
            "risk qərarına ən çox təsir edən əsas amilləri sadə dildə göstərir."
        )

        try:
            feature_names = model.named_steps[
                "preprocess"
            ].get_feature_names_out()

            risk_increasing, risk_reducing = get_business_factors(
                shap_result,
                feature_names,
                max_items=4
            )

            col_up, col_down = st.columns(2)

            with col_up:
                st.markdown("##### 🔴 Riski artıran əsas amillər")

                if risk_increasing:
                    for item in risk_increasing:
                        st.markdown(
                            f"""
                            <div style="
                                background:#fff5f5;
                                border-left:5px solid #dc3545;
                                padding:12px 14px;
                                border-radius:8px;
                                margin-bottom:9px;
                            ">
                                <div style="font-weight:700; color:#252b36;">
                                    {item["Faktor"]}
                                </div>
                                <div style="color:#9b2c2c; font-size:0.9rem; margin-top:3px;">
                                    {item["Təsir"]}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                else:
                    st.success("Riski artıran nəzərəçarpacaq əsas amil yoxdur.")

            with col_down:
                st.markdown("##### 🟢 Riski azaldan əsas amillər")

                if risk_reducing:
                    for item in risk_reducing:
                        st.markdown(
                            f"""
                            <div style="
                                background:#f2fbf6;
                                border-left:5px solid #20a05a;
                                padding:12px 14px;
                                border-radius:8px;
                                margin-bottom:9px;
                            ">
                                <div style="font-weight:700; color:#252b36;">
                                    {item["Faktor"]}
                                </div>
                                <div style="color:#18794e; font-size:0.9rem; margin-top:3px;">
                                    {item["Təsir"]}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                else:
                    st.info("Riski azaldan nəzərəçarpacaq əsas amil yoxdur.")

            if risk_increasing and risk_reducing:
                top_up = risk_increasing[0]["Faktor"]
                top_down = risk_reducing[0]["Faktor"]

                st.markdown(
                    f"""
                    <div style="
                        background:#f7f9fc;
                        border:1px solid #e1e6ef;
                        padding:14px 16px;
                        border-radius:10px;
                        margin-top:10px;
                    ">
                        <strong>💡 Qısa izah:</strong>
                        Müştərinin riskinə ən çox təsir edən amil
                        <strong>{top_up}</strong> risk istiqamətindədir.
                        Eyni zamanda <strong>{top_down}</strong> riskin azalmasına
                        kömək edən əsas amillərdəndir.
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            elif risk_increasing:
                st.info(
                    "Modelin risk qərarında əsas təsir daha çox riski artıran amillərdən gəlir."
                )
            elif risk_reducing:
                st.success(
                    "Modelin risk qərarında əsas təsir daha çox riski azaldan amillərdən gəlir."
                )

        except Exception as e:
            st.warning(
                f"⚠️ Risk izahı hazırlana bilmədi: {e}"
            )

        # RISK SCALE

        # -------------------------------------------------



        st.markdown(

            "#### 🚦 Risk şkalası"

        )



        st.progress(

        float(min(proba, 1.0)),

        text=f"Default probability: {proba:.2%}"

        )



        # -------------------------------------------------

        # CUSTOMER FEATURES

        # -------------------------------------------------



        st.divider()



        st.markdown(

            "#### 📋 Müştərinin feature-ləri"

        )



        feature_df = selected_row.T.reset_index()



        feature_df.columns = [

            "Feature",

            "Value"

        ]



        st.dataframe(

            feature_df,

            use_container_width=True,

            height=350,

            hide_index=True

        )





# =====================================================

# TAB 2 — PORTFOLIO

# =====================================================



with tab2:



    st.markdown(

        '<div class="section-title">'

        '📊 Kredit Portfolio Analizi'

        '</div>',

        unsafe_allow_html=True

    )



    st.write(

        "Nümunə datasetindəki bütün müştərilər üçün "

        "default ehtimallarının ümumi görünüşü."

    )



    if st.button(

        "📊 Portfolio-nu analiz et",

        type="primary",

        use_container_width=True

    ):



        with st.spinner(

            "Bütün müştərilər üçün proqnoz hesablanır..."

        ):



            probs = model.predict_proba(

                model_features_df

            )[:, 1]



            result_df = sample_df.copy()



            result_df.insert(

                0,

                "default_probability",

                probs

            )



            result_df.insert(

                1,

                "risk_category",

                pd.cut(

                    probs,

                    bins=[

                        -0.01,

                        0.20,

                        0.50,

                        1.0

                    ],

                    labels=[

                        "Aşağı",

                        "Orta",

                        "Yüksək"

                    ]

                )

            )



            st.session_state.result_df = result_df

            st.session_state.batch_completed = True





    # =================================================

    # SHOW RESULTS

    # =================================================



    if st.session_state.batch_completed:



        result_df = st.session_state.result_df





        # -------------------------------------------------

        # KPI CARDS

        # -------------------------------------------------



        total_customers = len(result_df)



        low_count = (

            result_df["risk_category"]

            == "Aşağı"

        ).sum()



        medium_count = (

            result_df["risk_category"]

            == "Orta"

        ).sum()



        high_count = (

            result_df["risk_category"]

            == "Yüksək"

        ).sum()



        avg_probability = (

            result_df["default_probability"]

            .mean()

        )





        st.markdown(

            "### Portfolio göstəriciləri"

        )



        c1, c2, c3, c4 = st.columns(4)



        c1.metric(

            "👥 Müştəri sayı",

            f"{total_customers:,}"

        )



        c2.metric(

            "🟢 Aşağı risk",

            f"{low_count:,}"

        )



        c3.metric(

            "🟡 Orta risk",

            f"{medium_count:,}"

        )



        c4.metric(

            "🔴 Yüksək risk",

            f"{high_count:,}"

        )





        st.write("")





        c5, c6 = st.columns(2)



        c5.metric(

            "Orta default ehtimalı",

            f"{avg_probability:.2%}"

        )



        high_risk_pct = (

            high_count / total_customers

            if total_customers > 0

            else 0

        )



        c6.metric(

            "Yüksək risk payı",

            f"{high_risk_pct:.2%}"

        )





        st.divider()





        # -------------------------------------------------

        # CHARTS

        # -------------------------------------------------



        chart1, chart2 = st.columns(2)





        # Risk distribution



        with chart1:



            risk_counts = (

                result_df["risk_category"]

                .value_counts()

                .reindex(

                    [

                        "Aşağı",

                        "Orta",

                        "Yüksək"

                    ]

                )

                .fillna(0)

            )



            fig_risk = px.bar(

                x=risk_counts.index,

                y=risk_counts.values,

                labels={

                    "x": "Risk kateqoriyası",

                    "y": "Müştəri sayı"

                },

                title="Risk kateqoriyalarının paylanması"

            )



            fig_risk.update_layout(

                height=370,

                showlegend=False

            )



            st.plotly_chart(

                fig_risk,

                use_container_width=True

            )





        # Probability distribution



        with chart2:



            fig_hist = px.histogram(

                result_df,

                x="default_probability",

                nbins=30,

                title="Default ehtimalının paylanması",

                labels={

                    "default_probability":

                    "Default ehtimalı"

                }

            )



            fig_hist.update_xaxes(

                tickformat=".0%"

            )



            fig_hist.update_layout(

                height=370

            )



            st.plotly_chart(

                fig_hist,

                use_container_width=True

            )





        # -------------------------------------------------

        # HIGH RISK CUSTOMERS

        # -------------------------------------------------



        st.divider()



        st.markdown(

            "### 🔴 Ən yüksək riskli müştərilər"

        )



        top_risk = (

            result_df

            .sort_values(

                "default_probability",

                ascending=False

            )

            .head(10)

            .copy()

        )



        top_risk["default_probability"] = (

            top_risk["default_probability"]

            .map(lambda x: f"{x:.2%}")

        )



        st.dataframe(

            top_risk,

            use_container_width=True,

            hide_index=True

        )





        # -------------------------------------------------

        # DOWNLOAD

        # -------------------------------------------------



        st.divider()



        csv_data = result_df.to_csv(

            index=False

        ).encode("utf-8")



        st.download_button(

            label="⬇️ Prediction nəticələrini CSV kimi yüklə",

            data=csv_data,

            file_name="home_credit_predictions.csv",

            mime="text/csv",

            use_container_width=True

        )





# =====================================================

# TAB 3 — DATASET

# =====================================================



with tab3:



    st.markdown(

        '<div class="section-title">'

        '📋 Deployment Dataset'

        '</div>',

        unsafe_allow_html=True

    )



    st.write(

        "Model üçün istifadə olunan deployment datasetinin "

        "ümumi görünüşü."

    )





    # -------------------------------------------------

    # SEARCH

    # -------------------------------------------------



    search_value = st.text_input(

        "🔎 Müştəri index-i axtar",

        placeholder="Məsələn: 100"

    )





    display_df = sample_df.copy()





    if search_value:



        try:



            search_index = int(

                search_value

            )



            if search_index in display_df.index:



                display_df = display_df.loc[

                    [search_index]

                ]



            else:



                st.warning(

                    "Bu index datasetdə tapılmadı."

                )



        except ValueError:



            st.warning(

                "Zəhmət olmasa rəqəm daxil edin."

            )





    # -------------------------------------------------

    # DATASET METRICS

    # -------------------------------------------------



    d1, d2, d3 = st.columns(3)



    d1.metric(

        "Sətir sayı",

        f"{sample_df.shape[0]:,}"

    )



    d2.metric(

        "Feature sayı",

        f"{sample_df.shape[1]:,}"

    )



    d3.metric(

        "Missing value",

        f"{sample_df.isna().sum().sum():,}"

    )





    st.write("")





    st.dataframe(

        display_df,

        use_container_width=True,

        height=500

    )