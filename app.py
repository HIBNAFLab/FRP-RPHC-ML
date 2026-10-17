"""Streamlit frontend for post-heating FRP-repaired concrete ML models."""
from io import BytesIO
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import joblib

st.set_page_config(page_title="FRP Post-Heating | ML Prediction Tool",page_icon="🏗️",layout="wide")
ROOT=Path(__file__).resolve().parent
ARTIFACTS={
    'regressor':'Best_RF_Regressor.joblib',
    'reg_poly':'PolyTransformer_fc.joblib',
    'reg_scaler':'Scaler_fc.joblib',
    'classifier':'Best_XGBoost_Classifier.joblib',
    'clf_poly':'PolyTransformer_Classification.joblib',
    'clf_scaler':'Scaler_Classification.joblib',
    'label_encoder':'LabelEncoder_Classification.joblib'
}
NUMERIC_RANGES={
    'd':(100,200),'h':(200,300),'fco':(28,56.97),'fco,T':(7.9,56.97),
    'ff':(1300,4900),'Ef':(65,234),'tf':(0.096,0.484),'εf':(1.7,2.5),
    'Length':(100,200),'Temp':(20,800),'Rate':(0,13),'Soaking':(0,180)
}
# CRITICAL: This retains the input sequence in the supplied working PySimpleGUI script.
FEATURE_COLUMNS=list(NUMERIC_RANGES)+['Cycle','Cooling']
CYCLE={'1: No':1,'2: Yes':2}
COOLING={'1: Air':1,'2: Water':2,'3: Furnace':3}
FAILURE_NAMES={1:'Rupture',2:'Debonding',3:'Mixed'}
LABELS={
    'd':'d (mm)','h':'h (mm)','fco':'fco (MPa)','fco,T':'fco,T (MPa)',
    'ff':'ff (MPa)','Ef':'Ef (GPa)','tf':'tf (mm)','εf':'εf (%)',
    'Length':'Overlap length (mm)','Cycle':'Dry cycle','Temp':'Target temperature (°C)',
    'Rate':'Heating rate (°C/min)','Soaking':'Soaking time (min)','Cooling':'Cooling method'
}
GROUPS=[
    ('Concrete Geometry & Parameters',['d','h','fco','fco,T']),
    ('FRP Parameters',['ff','Ef','tf','εf','Length']),
    ('Heating Parameters',['Cycle','Temp','Rate','Soaking']),
    ('Cooling Technique',['Cooling'])
]

@st.cache_resource(show_spinner='Loading trained models...')
def load_artifacts():
    directory=ROOT/'Results'/'Models'
    missing=[filename for filename in ARTIFACTS.values() if not (directory/filename).is_file()]
    if missing:
        raise FileNotFoundError('Missing files in Results/Models/: '+', '.join(missing))
    # Only load joblib files that you created or obtained from trusted sources.
    return {key:joblib.load(directory/filename) for key,filename in ARTIFACTS.items()}

def make_input(values):
    row=[]
    for feature in NUMERIC_RANGES:
        value=float(values[feature])
        low,high=NUMERIC_RANGES[feature]
        if not np.isfinite(value) or not low<=value<=high:
            raise ValueError(f'{LABELS[feature]} must be between {low} and {high}.')
        row.append(value)
    if values['Cycle'] not in CYCLE or values['Cooling'] not in COOLING:
        raise ValueError('Please select valid heating-cycle and cooling options.')
    row.extend([CYCLE[values['Cycle']],COOLING[values['Cooling']]])
    return np.asarray([row],dtype=np.float64)

def predict(values,models):
    X=make_input(values)
    xr=models['reg_scaler'].transform(models['reg_poly'].transform(X))
    xc=models['clf_scaler'].transform(models['clf_poly'].transform(X))
    strength=float(np.asarray(models['regressor'].predict(xr)).reshape(-1)[0])
    if not np.isfinite(strength):
        raise ValueError('Regression model returned a non-finite prediction.')
    classifier=models['classifier']
    confidence=None
    if hasattr(classifier,'predict_proba'):
        probabilities=np.asarray(classifier.predict_proba(xc))[0]
        choice=int(np.argmax(probabilities))
        encoded=classifier.classes_[choice]
        confidence=float(probabilities[choice])
    else:
        encoded=np.asarray(classifier.predict(xc)).reshape(-1)[0]
    try:
        decoded=models['label_encoder'].inverse_transform([encoded])[0]
    except (ValueError,TypeError,AttributeError):
        decoded=encoded
    try:
        mode=FAILURE_NAMES.get(int(decoded),str(decoded))
    except (ValueError,TypeError):
        mode=str(decoded)
    return {'fcc,T (MPa)':strength,'fcc,T/fco,T':strength/float(values['fco,T']),
            'Failure mode':mode,'Model class probability':confidence}

def excel_bytes(records):
    buffer=BytesIO()
    with pd.ExcelWriter(buffer,engine='openpyxl') as writer:
        pd.DataFrame(records).to_excel(writer,index=False,sheet_name='Predictions')
    return buffer.getvalue()

def clear_all():
    for feature in FEATURE_COLUMNS:
        st.session_state.pop('inp_'+feature,None)
    st.session_state['history']=[]
    st.session_state['last_result']=None

st.markdown('''<style>
.block-container{padding-top:1.6rem;max-width:1250px}
[data-testid="stMetricValue"]{font-size:1.55rem}
</style>''',unsafe_allow_html=True)
st.title('🏗️ ML-Aided Prediction of Post-Heated FRP-Repaired Concrete')
st.caption('Ultimate compressive strength and failure mode prediction | Research application')
st.info('Research-use tool. Predictions are data-driven estimates, not substitutes for experimental verification or engineering design checks.')

if 'history' not in st.session_state:
    st.session_state['history']=[]
if 'last_result' not in st.session_state:
    st.session_state['last_result']=None

try:
    models=load_artifacts()
except Exception as exc:
    st.error('The trained model files could not be loaded. Verify Results/Models/ and dependency versions.')
    st.code(str(exc))
    st.stop()

st.subheader('Specimen and exposure inputs')
st.caption('Enter values within the ranges from the original research GUI. All units are shown alongside their variables.')
inputs={}
left,right=st.columns(2,gap='large')
for index,(heading,features) in enumerate(GROUPS):
    with (left if index in (0,2) else right):
        with st.container(border=True):
            st.markdown(f'**{heading}**')
            for feature in features:
                if feature=='Cycle':
                    inputs[feature]=st.selectbox(LABELS[feature],list(CYCLE),key='inp_'+feature,index=None,placeholder='Select a cycle option')
                elif feature=='Cooling':
                    inputs[feature]=st.selectbox(LABELS[feature],list(COOLING),key='inp_'+feature,index=None,placeholder='Select a cooling method')
                else:
                    lo,hi=NUMERIC_RANGES[feature]
                    inputs[feature]=st.number_input(
                        LABELS[feature],min_value=float(lo),max_value=float(hi),
                        value=None,step=0.01 if feature in ('tf','εf') else 0.1,
                        format='%.3f' if feature=='tf' else '%.2f',
                        placeholder=f'{lo} to {hi}',key='inp_'+feature)

predict_col,clear_col=st.columns([1,1])
with predict_col:
    clicked=st.button('🔍 Predict',type='primary',use_container_width=True)
with clear_col:
    st.button('Clear inputs and batch',on_click=clear_all,use_container_width=True)

if clicked:
    absent=[LABELS[key] for key in FEATURE_COLUMNS if inputs.get(key) is None]
    if absent:
        st.error('Complete all inputs before prediction: '+', '.join(absent))
    else:
        try:
            result=predict(inputs,models)
            st.session_state['last_result']=result
            entry={feature:inputs[feature] for feature in FEATURE_COLUMNS}
            entry.update(result)
            st.session_state['history'].append(entry)
        except Exception as exc:
            st.error(f'Prediction failed: {exc}')

st.divider()
st.subheader('Prediction outputs')
result=st.session_state['last_result']
if result is None:
    st.caption('No prediction yet. Complete the specimen inputs and select Predict.')
else:
    a,b,c=st.columns(3)
    a.metric('Ultimate strength, fcc,T',f"{result['fcc,T (MPa)']:.3f} MPa")
    b.metric('Strength enhancement, fcc,T/fco,T',f"{result['fcc,T/fco,T']:.4f}")
    c.metric('Failure mode',result['Failure mode'])
    if result['Model class probability'] is not None:
        st.caption(f"Classifier probability for predicted class: {result['Model class probability']:.1%}. This is not a calibrated probability of structural failure.")

st.subheader(f"Batch predictions ({len(st.session_state['history'])})")
if st.session_state['history']:
    st.dataframe(pd.DataFrame(st.session_state['history']),hide_index=True,use_container_width=True)
    st.download_button('⬇️ Download predictions as Excel',data=excel_bytes(st.session_state['history']),
                       file_name='FRP_PostHeating_Predictions.xlsx',
                       mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                       use_container_width=True)
else:
    st.caption('Repeated predictions are collected here for Excel export. Batch records are held only in this browser session.')

st.divider()
st.caption('© 2026 I.A. Tijani, M. Elshorbagi, & C. Jiang. All rights reserved.')
st.caption('Data-Augmented Machine Learning Framework for Failure Modes Classification and Ultimate Strength Prediction of FRP-Repaired Post-Heated Concrete (under review).')
st.caption('Applicability is limited to the training-data domain; combinations within individual input ranges may still lie outside the multivariate training domain.')
