import plotly.express as px
import pandas as pd
import streamlit as st
import plotly.graph_objs as go
import plotly.figure_factory as ff
import plotly.subplots as sp
from plotly.subplots import make_subplots
import joblib
from utils_app import (
    plot_continuos_histogram_matrix,
    plot_cat_histogram_matrix,
    show_corr,
)

# Load data
data = pd.read_csv("..\data\intermediate\complete_dataset.csv")
data = data.drop(columns=["Unnamed: 0"])

# Title
st.title("Adult Income Application")
# Text Description
st.markdown(
    "This is an app that take Adult Dataset for put in practice concepts \
    about data science and machine learning."
)
# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(
    ["Exploratory Data Analysis", "Preprocessing", "Modelling", "Test"]
)

# EDA
with tab1:
    st.markdown(
        "As a first step, the missing data are reviewed, obtaining \
                as a result:"
    )
    # Create some figures
    fig = ff.create_table(data.isnull().sum().to_frame(name="Count"), index=True)
    miss_values = data.isnull().sum().to_dict()
    fig.add_trace(
        go.Bar(
            x=list(miss_values.keys()),
            y=list(miss_values.values()),
            xaxis="x2",
            yaxis="y2",
            name="",
        )
    )
    fig.update_layout(
        title_text="<b>Missing Values</b>",
        height=500,
        width=1000,
        margin={"t": 75, "l": 70},
        xaxis={"domain": [0, 0.25]},
        xaxis2={"domain": [0.4, 0.9]},
        yaxis2={"domain": [0, 1], "anchor": "x2", "title": "Count"},
        template="plotly_white",
    )

    st.plotly_chart(fig, theme="streamlit", use_container_width=True)

    st.markdown(
        "Target proportion show that Adult Income, in fact, is an\
                unbalanced dataset:"
    )
    data["Income"] = data["Income"].replace(
    ["<=50K", ">50K", "<=50K.", ">50K."], [0, 1, 0, 1]
    )
    target_ratio = data["Income"].value_counts().to_dict()
    fig2 = px.pie(
        names=["<=50K", ">50K"],
        values=list(target_ratio.values()),
        title="<b>Target proportion</b>",
        width=500,
        height=500,
    )
    st.plotly_chart(fig2, theme="streamlit", use_container_width=True)

    st.markdown(
        "The histograms of the continuous characteristics show \
                some skewness in the distribution of the data as well as \
                some outliers."
    )

    fig3 = plot_continuos_histogram_matrix(data)

    st.plotly_chart(fig3, theme="streamlit", use_container_width=True)

    st.markdown(
        "On the other hand, categorical features show a lot of \
                categories:"
    )
    fig4 = plot_cat_histogram_matrix(data)
    st.plotly_chart(fig4, theme="streamlit", use_container_width=True)
    st.markdown(
        "As we can see, some of the variables have rare categories, for example \
                WorkClass and Native_Country. Now we realize a correlation analysis:"
    )
    corr_di = data.corr(method="pearson", numeric_only=True)
    fig5 = show_corr(correlation_matrix=corr_di)
    st.plotly_chart(fig5, theme="streamlit", use_container_width=True)

# Preprocessing
with tab2:
    markdown_list = """
        1. There are only missing values in the categorical variables, \
                the most frequent value will be used to fill them.
        2. For solve unbalanced data, a SMOTE technique was appliead trough imblearn\
            library
        3. Numerical column Edu_Num and categorical column Education represent the same \
            information, the difference is while Education refers to grade name Edu_Num refers\
            to a number. Education was removed because Edu_num keep relations among education \
            levels.
        4. Age has no change.
        5. Rare categories in WorkClass and Native_Country are grouped, Race is removed from data.
        6. The SMOTENC technique was applied to oversample the dataset. A new correlation \
           analysis was conducted, and the variables increased in their values except for \
           'fnlwgt' and 'Capital_Loss'.
        """
    st.markdown(markdown_list)

# Modelling
with tab3:
    st.markdown(
        "To illustrate the utilization of the Adult Income Dataset, \
        an experiment was conducted involving three different machine \
        learning models: Random Forest, XGBoost and \
        a small Neural Network"
    )
    st.markdown(
        "Since the dataset is unbalanced, all models were trained with \
        oversampled dataset"
    )

    st.subheader("Random Forest")
    st.markdown(
        "Due the previous steps, fnlwgt field was removed on the training \
        dataset. Then a small pipeline was building for do preprocessing \
        and use in join with the model."
    )
    st.markdown(
        "Next table show the Random Forest configuration through \
                Sklearn framework"
    )
    rf_config = pd.read_csv(
        r"..\reports\dataframe_results\random_forest\rf_config.csv",
        names=["param", "Value"],
        skiprows=[0],
    )
    rf_fig = ff.create_table(rf_config, index=False)
    rf_fig.update_layout(
        title_text="<b>Random Forest Configuration</b>",
        template="plotly_white",
        margin={"t": 75},
        width=320,
    )
    st.plotly_chart(rf_fig, theme="streamlit", use_container_width=False)

    st.markdown("Next tables show the classification report on train and test set:")

    cr_train=pd.read_csv(r"..\reports\dataframe_results\random_forest\cr_train.csv",
                         index_col=0)
    cr_train=cr_train.fillna(value='')

    cr_fig = ff.create_table(cr_train.T, index=True)
    cr_fig.update_layout(
        title_text="<b>Classification report (Train)</b>",
        margin = {'t':50, 'b':10,'l':10,'r':10},
        height=300,
        width=550,
    )
    st.plotly_chart(cr_fig, theme="streamlit", use_container_width=False)

    cr_test=pd.read_csv(r"..\reports\dataframe_results\random_forest\cr_test.csv",
                         index_col=0)
    cr_test=cr_test.fillna(value='')
    cr_fig2 = ff.create_table(cr_test.T, index=True)
    cr_fig2.update_layout(
        title_text="<b>Classification report (Test)</b>",
        margin = {'t':50, 'b':10,'l':10,'r':10},
        height=300,
        width=550,
    )
    st.plotly_chart(cr_fig2, theme="streamlit", use_container_width=False)
with tab4:
    st.markdown("In this section we can test the models!")
    def pass_cols(x):
        return x
    Pipeline=joblib.load(r'C:\Users\migue\Desktop\Portfolio\Clasificación de ingresos anuales\models\random_forest\rf_pipeline.joblib')
    with st.form("my_form"):
        header = st.columns([1,1,1])
        # header[0].subheader('Color')
        # header[1].subheader('Opacity')
        # header[2].subheader('Size')
        # * Numerical input
        age=header[0].number_input("Age",0,100)
        edu=header[0].selectbox("Level of Education",
                        ('Preschool','1st-4th','5th-6th','7th-8th','9th',
                        '10th','11th','12th','HS-grad','Some-college',
                        'Assoc-voc','Assoc-acdm','Bachelors','Masters',
                        'Prof-school','Doctorate'))
        cap_gain=header[0].number_input("Capital Gain",min_value=0)
        cap_loss=header[1].number_input("Capital Loss",min_value=0)
        hp_week=header[1].number_input("Hours per week",min_value=0)

        # * Categorical input
        workclass=header[1].selectbox("WorkClass",
                            ("Private","Self-emp-not-inc","Local-gov",
                                "State-gov","Self-emp-inc","Federal-gov",
                                "Other"))
        marital_status=header[2].selectbox("Marital Status",
                                    ("Married-civ-spouse","Never-married",
                                    "Divorced","Separated","Widowed",
                                    "Married-spouse-absent","Married-AF-spouse"))
        ocupation=header[2].selectbox("Occupation",
                            ('Prof-specialty', 'Craft-repair', 'Exec-managerial',
                                'Adm-clerical', 'Sales', 'Other-service', 
                                'Machine-op-inspct', 'Transport-moving', 
                                'Handlers-cleaners', 'Farming-fishing', 'Tech-support', 
                                'Protective-serv', 'Priv-house-serv', 'Armed-Forces'))
        relationship=header[2].selectbox("Relationship",
                                ('Husband', 'Not-in-family', 'Own-child', 'Unmarried', 
                                'Wife', 'Other-relative'))
        sex=st.selectbox("Sex",('Male','Female'))
        nat_country=st.selectbox("Native Country",
                                ("United-States",'OtherCountry'))
    
        cols=['Age', 'WorkClass', 'Edu_num', 'Marital_Status',
        'Occupation', 'Relationship', 'Sex', 'Capital_Gain',
        'Capital_Loss', 'hpweek', 'Native_Country']
        # Education pass to a num via 
        cats=('Preschool','1st-4th','5th-6th','7th-8th','9th',
                        '10th','11th','12th','HS-grad','Some-college',
                        'Assoc-voc','Assoc-acdm','Bachelors','Masters',
                        'Prof-school','Doctorate')
        nums=(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16)
        Edu_num=dict(zip(cats,nums))

        input_test=[age,
                    Edu_num[edu],
                    cap_gain,
                    cap_loss,
                    hp_week,
                    workclass,
                    marital_status,
                    ocupation,
                    relationship,
                    sex,
                    nat_country]
        data_test=dict(zip(
            ['Age', 'Edu_num', 'Capital_Gain', 'Capital_Loss', 'hpweek','WorkClass', 'Marital_Status', 'Occupation', 'Relationship', 'Sex', 'Native_Country'],
            input_test
        ))
        data_test=pd.DataFrame.from_dict(data_test,orient='index').T

        do_predict=st.form_submit_button('Give me a prediction')
    result={1:'>50K',0:'<=50K'}
    if do_predict:    
        res,res_proba=Pipeline.predict(data_test),Pipeline.predict_proba(data_test)
        st.write(f"Prediction results: Income {result[res[0]]}")
        st.write(f"Your probabilites for <=50K is {res_proba[0][0]} and for >50K {res_proba[0][1]}")