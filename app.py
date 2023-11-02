from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
from data import get_college_data
# GET MODELS
filename = r"C:\Users\ASUS\Downloads\nirf_ranking-master\rank_model.pkl"
filename1= r"C:\Users\ASUS\Downloads\nirf_ranking-master\polynomial_transform.pkl"
model = pickle.load(open(filename, 'rb'))
model1=pickle.load(open(filename1,'rb'))

class modify1:
    def find_range(self,num):
        if num<=0:
            return "2 (+/- 1)"
        elif num<=100:
            return str(num)+" (+/- 5)"
        elif num <=120:
            return str(num)+" (+/- 5)"
        elif num<=140:
            return str(num)+" (+/- 7) " 
        elif num<=170:
            return str(num)+" (+/- 12) " 
        elif num<=180:
            return str(num)+" (+/- 14) "
        else:
            return str(num)+" (+/- 13) " 

rank=[]
rpc=[]
tlr=[]
go=[]
oi=[]
ppn=[]

def engg_clear():
    rank.clear()
    rpc.clear()
    tlr.clear()
    go.clear()
    oi.clear()
    ppn.clear()    


app = Flask(__name__)
app.secret_key = 'nightowls'


@app.route('/')
def home():
	return render_template('codeut.html')

@app.route('/engg/')
def engg():
    return render_template('engg.html')

@app.route('/analysis/')
def analysis():
    return render_template('analysis.html', messages = [], clg_info=[], clg_total=[], clg_tlr=[], clg_rpc=[], clg_go=[], clg_oi=[], clg_pr=[],form_success=False)

@app.route('/index')
def index():
    engg_clear()
    return render_template('codeut.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        tlr1 = float(request.form['tlr'])
        rpc1 = float(request.form['rpc'])
        go1 = float(request.form['go'])
        oi1 =float(request.form['oi'])
        perception1 = float(request.form['ppn'])

        score=(tlr1*0.3)+(rpc1*0.3)+(go1*0.2)+(oi1*0.1)+(perception1*0.1)
        data = np.array([score]).reshape(1,-1)
        pre_prediction=model1.fit_transform(data)
        my_prediction = model.predict(pre_prediction)
        
        
        tlr.append(float(tlr1))
        rpc.append(float(rpc1))
        go.append(float(go1))
        oi.append(float(oi1))
        ppn.append(float(perception1))
        df_engg = pd.DataFrame({"TLR":tlr,"RPC":rpc,"GO":go,"OI":oi,"PPN":ppn})
        df_engg.index = np.arange(df_engg.shape[0])+1
        
        
        x=my_prediction[0]
        range1=modify1()
        ranges=range1.find_range(int(x))
        if x<=200:
            rank.append(ranges)
            df_engg['RANK']=rank
            return render_template('engg.html', score=score, prediction="The predicted rank might be in range : " + ranges,tables=[df_engg.to_html(classes='data')],titles=['.','TLR','RPC','GO','OI','PPN','RANK'])
        else:
            rank.append(">150")
            df_engg['RANK']=rank            
            return render_template('engg.html',prediction="The rank for this score is greater than 200",tables=[df_engg.to_html(classes='data')],titles=['.','TLR','RPC','GO','OI','PPN','RANK'])

@app.route('/college_data', methods=['POST'])
def college_data():
    if request.method == 'POST':
        college = request.form['college']
        print(college)
        messages = []
        clg_info = get_college_data(college)
        clg_total = [clg_info[4], clg_info[11], clg_info[18], clg_info[25], clg_info[32], clg_info[39]]
        clg_tlr = [clg_info[6], clg_info[13], clg_info[20], clg_info[27], clg_info[34], clg_info[41]]
        clg_rpc = [clg_info[7], clg_info[14], clg_info[21], clg_info[28], clg_info[35], clg_info[42]]
        clg_go = [clg_info[8], clg_info[15], clg_info[22], clg_info[29], clg_info[36], clg_info[43]]
        clg_oi = [clg_info[9], clg_info[16], clg_info[23], clg_info[30], clg_info[37], clg_info[44]]
        clg_pr = [clg_info[10], clg_info[17], clg_info[24], clg_info[31], clg_info[38], clg_info[45]]
        if clg_tlr[0]!=max(clg_tlr):
            messages.append("TLR")
        if clg_rpc[0]!=max(clg_rpc):
            messages.append("RPC")
        if clg_go[0]!=max(clg_go):
            messages.append("GO")
        if clg_oi[0]!=max(clg_oi):
            messages.append("OI")
        if clg_pr[0]!=max(clg_pr):
            messages.append("PR")
        return render_template('analysis.html', messages=messages,clg_info=clg_info,clg_total = clg_total, clg_tlr=clg_tlr, clg_rpc=clg_rpc, clg_go=clg_go, clg_oi=clg_oi, clg_pr=clg_pr, form_success=True)

if __name__ == '__main__':
	app.run(debug=True)
