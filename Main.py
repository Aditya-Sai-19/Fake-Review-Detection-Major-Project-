import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import os
from flask import Flask, send_file,render_template, request, redirect, Response
import pymysql
from datetime import date
import json
import os
from string import punctuation
from nltk.corpus import stopwords
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import pickle
from datetime import date
import random

from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
from keras.models import model_from_json
import pickle
from keras.layers import LSTM, Bidirectional
import keras

app = Flask(__name__)
app.secret_key = 'amazon'
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

global cnn_classifier, lstm_classifier
global vectorizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
reviews = []
sentiments = []
fake = []
accuracy = []
precision = []
recall = []
fscore = []

def cleanPost(doc):
    tokens = doc.split()
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tokens = ' '.join(tokens)
    return tokens

@app.route("/Train")
def Train():
    global reviews
    global sentiments
    global fake
    global accuracy
    global precision
    global recall
    global fscore
    global cnn_classifier, lstm_classifier
    global vectorizer
    
    reviews.clear()
    sentiments.clear()
    fake.clear()
    accuracy.clear()
    precision.clear()
    recall.clear()
    fscore.clear()
    
    if os.path.exists('model/reviews.txt.npy'):
        reviews = np.load("model/reviews.txt.npy")
        sentiments = np.load("model/rating.txt.npy")
        fake = np.load("model/fake.txt.npy")
    else:
        dataset = pd.read_csv("Dataset/Reviews.csv",nrows=10000)
        dataset = dataset.values
        for i in range(len(dataset)):
            numerator = dataset[i,4]
            denomerator = dataset[i,5]
            rating = dataset[i,6]
            text_review = dataset[i,9]
            text_review = text_review.strip('\n')
            text_review = text_review.strip()
            reviews.append(cleanPost(text_review.strip().lower()))
            sentiments.append(rating-1)
            if numerator >= denomerator:
                fake.append(0)
            else:
                fake.append(1)

        reviews = np.asarray(reviews)
        sentiments = np.asarray(sentiments)
        fake = np.asarray(fake)
        np.save("model/reviews.txt",reviews)
        np.save("model/rating.txt",sentiments)
        np.save("model/fake.txt",fake)

    vectorizer = TfidfVectorizer(stop_words=stop_words, use_idf=True, smooth_idf=False, norm=None, decode_error='replace', max_features=500)
    tfidf = vectorizer.fit_transform(reviews).toarray()        
    df = pd.DataFrame(tfidf, columns=vectorizer.get_feature_names())
    print(str(df))
    print(df.shape)
    df = df.values
    X = df[:, 0:500]
    X = X.reshape((X.shape[0],X.shape[1],1,1))
    X1 = X.reshape((X.shape[0],X.shape[1],1,1))
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    sentiments = sentiments[indices]
    sentiments = to_categorical(sentiments)

    X1 = X1[indices]
    fake = fake[indices]
    fake = to_categorical(fake)

    X_train, X_test, y_train, y_test = train_test_split(X, sentiments, test_size=0.2)
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, fake, test_size=0.2)
    
    if os.path.exists('model/cnn.json'):
        with open('model/cnn.json', "r") as json_file:
            loaded_model_json = json_file.read()
            cnn_classifier = model_from_json(loaded_model_json)
        json_file.close()    
        cnn_classifier.load_weights("model/cnn_weights.h5")
        cnn_classifier._make_predict_function()   
        
    else:
        cnn_classifier = Sequential()
        #defining convolution CNN layer with 32 filters and giving X as input details
        cnn_classifier.add(Convolution2D(32, 1, 1, input_shape = (X.shape[1], X.shape[2], X.shape[3]), activation = 'relu'))
        cnn_classifier.add(MaxPooling2D(pool_size = (1, 1)))
        #defining another CNN layer with 32 filters
        cnn_classifier.add(Convolution2D(32, 1, 1, activation = 'relu'))
        cnn_classifier.add(MaxPooling2D(pool_size = (1, 1)))
        #converting dataset into single dimensional from multi dimensional array
        cnn_classifier.add(Flatten())
        #defining output layer
        cnn_classifier.add(Dense(output_dim = 256, activation = 'relu'))
        #defining prediction output layet as sentiment values prediction
        cnn_classifier.add(Dense(output_dim = sentiments.shape[1], activation = 'softmax'))
        print(cnn_classifier.summary())
        #compiling CNN model
        cnn_classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        #now training CNN with X as dataset reviews and sentiments values as Y. Here X are reviews and sentiments are sentiment values
        hist = cnn_classifier.fit(X, sentiments, batch_size=16, epochs=10, shuffle=True, verbose=2)
        cnn_classifier.save_weights('model/cnn_weights.h5')            
        model_json = cnn_classifier.to_json()
        with open("model/cnn.json", "w") as json_file:
            json_file.write(model_json)
        json_file.close()

    if os.path.exists('model/lstm.json'):
        with open('model/lstm.json', "r") as json_file:
            loaded_model_json = json_file.read()
            lstm_classifier = model_from_json(loaded_model_json)
        json_file.close()
        lstm_classifier.load_weights("model/lstm_weights.h5")
        lstm_classifier._make_predict_function()   
        
    else:
        lstm_classifier = Sequential()
        #defining LSTM layer with 32 filetsr and X reviews as input
        lstm_classifier.add(LSTM(32, return_sequences=True, input_shape=(X1.shape[1],1)))
        lstm_classifier.add(Dropout(0.3))
        #defining another LSTM layer
        lstm_classifier.add(LSTM(16, return_sequences=True, input_shape=(X1.shape[1],1)))
        lstm_classifier.add(Dropout(0.3))
        #defining another LSTM layer to further filter dataset
        lstm_classifier.add(LSTM(8,input_shape=(X1.shape[1],1)))
        #Remove or drop irrelevant fetaures or data 
        lstm_classifier.add(Dropout(0.3))
        #input output values as FAKE
        lstm_classifier.add(Dense(fake.shape[1], activation='softmax'))
        #compile LSTM model
        lstm_classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        #now start training LSTM model with X reviews and fake class labels
        lstm_acc = lstm_classifier.fit(X1, fake, epochs=10, batch_size=256, validation_data=[X_test1, y_test1])
        lstm_classifier.save_weights('model/lstm_weights.h5')
        model_json = lstm_classifier.to_json()
        with open("model/lstm.json", "w") as json_file:
            json_file.write(model_json)
        json_file.close()        
    
    predict = cnn_classifier.predict(X_test)
    predict = np.argmax(predict, axis=1)
    y1_test1 = np.argmax(y_test, axis=1)
    output = '<table border=1 align=center>'
    output+='<tr><th><font size=3 color=black>Algorithm Name</font></th>'
    output+='<th><font size=3 color=black>Precision</font></th>'
    output+='<th><font size=3 color=black>Recall</font></th>'
    output+='<th><font size=3 color=black>FSCORE</font></th>'
    output+='<th><font size=3 color=black>Accuracy</font></th></tr>'     
    a = accuracy_score(y1_test1,predict)*100
    p = precision_score(y1_test1,predict,average='macro') * 100
    r = recall_score(y1_test1,predict,average='macro') * 100
    f = f1_score(y1_test1,predict,average='macro') * 100
    output+='<tr><td>CNN Algorithm</td><td>'+str(p)+'</td><td>'+str(r)+'</td><td>'+str(f)+'</td><td>'+str(a)+'</td></tr>'
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)

    predict = lstm_classifier.predict(X_test1)
    predict = np.argmax(predict, axis=1)
    y_test1 = np.argmax(y_test1, axis=1)
    a = accuracy_score(y_test1,predict)*100
    p = precision_score(y_test1,predict,average='macro') * 100
    r = recall_score(y_test1,predict,average='macro') * 100
    f = f1_score(y_test1,predict,average='macro') * 100
    output+='<tr><td>LSTM Algorithm</td><td>'+str(p)+'</td><td>'+str(r)+'</td><td>'+str(f)+'</td><td>'+str(a)+'</td></tr>'
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)

   
    df = pd.DataFrame([['CNN','Accuracy',accuracy[0]],['CNN','Precision',precision[0]],['CNN','Recall',recall[0]],['CNN','FScore',fscore[0]],
                       ['LSTM','Accuracy',accuracy[1]],['LSTM','Precision',precision[1]],['LSTM','Recall',recall[1]],['LSTM','FScore',fscore[1]],
                                             ],columns=['Parameters','Algorithms','Value'])
    df.pivot("Parameters", "Algorithms", "Value").plot(kind='bar')
    #plt.show()
    plt.savefig("static/result/cls.png")
    #plt.close()
    output+="</table>"
    output+="<br/>"
    output+='<img src="static/result/cls.png" height="600" width="600"/>'
    output+='<br/><br/><br/><br/><br/><br/>'
    return render_template("Train.html",output=output)
    

@app.route('/ViewClassification')
def ViewClassification():
    output = '<table border=1 align=center>'
    output+='<tr><th><font size=3 color=black>Reviewer Name</font></th>'
    output+='<th><font size=3 color=black>Review Text</font></th>'
    output+='<th><font size=3 color=black>Predicted Ratings</font></th>'
    output+='<th><font size=3 color=black>Predicted Sentiment</font></th>'
    output+='<th><font size=3 color=black>Review Date</font></th></tr>'
    con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'amazonreviews',charset='utf8')
    pos = 0
    neg = 0
    neu = 0
    myresult = "none"
    with con:
        cur = con.cursor()
        cur.execute("select * from submit_reviews")
        rows = cur.fetchall()
        for row in rows:
            if row[2] == 4 or row[2] == 5:
                pos = pos + 1
                myresult = "Positive"
            if row[2] == 3:
                neu = neu + 1
                myresult = "Neutral"
            if row[2] == 1 or row[2] == 2:
                neg = neg + 1
                myresult = "Negative"
            output+='<tr><td>'+row[0]+'</td><td>'+str(row[1])+'</td><td>'+str(row[2])+'</td><td>'+myresult+'</td><td>'+str(row[3])+'</td></tr>'
            
    height = [pos,neg,neu]
    bars = ('Positive', 'Negative','Neutral')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.title("Reviews Predicted Ratings Graph")
    plt.savefig("static/result/reviews.png")
    #plt.close()
    output+="</table>"
    output+="<br/>"
    output+='<img src="http://localhost:9999/static/result/reviews.png?cache="'+str(random.randint(10,10000))+' height="600" width="600"/>'                                                                                    
    output+='<br/><br/><br/><br/><br/><br/>'
    return render_template("ViewClassification.html",output=output)


@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

@app.route('/SubmitReviewAction', methods =['GET', 'POST'])
def SubmitReviewAction():
    if request.method == 'POST':
        global cnn_classifier, lstm_classifier
        global vectorizer

        if os.path.exists('model/cnn.json'):
            with open('model/cnn.json', "r") as json_file:
                loaded_model_json = json_file.read()
                cnn_classifier = model_from_json(loaded_model_json)
            json_file.close()    
        cnn_classifier.load_weights("model/cnn_weights.h5")
        cnn_classifier._make_predict_function()

        if os.path.exists('model/lstm.json'):
            with open('model/lstm.json', "r") as json_file:
                loaded_model_json = json_file.read()
                lstm_classifier = model_from_json(loaded_model_json)
            json_file.close()
        lstm_classifier.load_weights("model/lstm_weights.h5")
        lstm_classifier._make_predict_function()   
        
        name = request.form['t1']
        review = request.form['t2']
        result1 = 'none'
        result2 = 'none'
        data = cleanPost(review.strip().lower())
        temp = []
        temp.append(data)
        temp = np.asarray(temp)
        temp = vectorizer.transform(temp).toarray()
        temp = temp.reshape((temp.shape[0],temp.shape[1],1,1))

        predict1 = cnn_classifier.predict(temp)
        predict1 = np.argmax(predict1) + 1
        if predict1 == 1 or predict1 == 2:
            result1 = 'Negative';
        if predict1 == 3:
            result1 = 'Neutral';
        if predict1 == 4 or predict1 == 5:
            result1 = "Positive"

        predict2 = lstm_classifier.predict(temp)
        predict2 = np.argmax(predict2)
        if predict2 == 0:
            result2 = "Genuine";
        if predict2 == 1:
            result2 = "Fake";
        
        today = date.today()
        status = 'Error in submitting your review'
        db_connection = pymysql.connect(host='127.0.0.1',port = 3308,user = 'root', password = 'root', database = 'amazonreviews',charset='utf8')
        db_cursor = db_connection.cursor()
        query = "INSERT INTO submit_reviews(reviewer_name,review_text,ratings,review_date) VALUES('"+name+"','"+review+"','"+str(predict1)+"','"+str(today)+"')"
        db_cursor.execute(query)
        db_connection.commit()
        print(db_cursor.rowcount, "Record Inserted")
        if db_cursor.rowcount == 1:
            status = "Hybrid Fuzzy Prediction using CNN and LSTM<br/>Your review sentiments predicted as : "+result1+"<br/> Review predicted as : "+result2
        return render_template("SubmitReview.html",error=status)        


@app.route("/SubmitReview")
def SubmitReview():
    return render_template("SubmitReview.html")

@app.route("/Logout")
def Logout():
    return render_template("Logout.html")

@app.route("/index")
def index():
    return render_template("index.html")

@app.route("/Login")
def Login():
    return render_template("Login.html")

@app.route('/LoginAction', methods =['GET', 'POST'])
def LoginAction():
    if request.method == 'POST':
        global uid
        username = request.form['t1']
        password = request.form['t2']
        status = 'none'
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'amazonreviews',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select username FROM register where username='"+username+"' and password='"+password+"'")
            rows = cur.fetchall()
            for row in rows:
                if row[0] == username:
                    uid = username
                    status = 'success'
                    break
        if status == 'success':
            return render_template("RetailerScreen.html",error='Welcome '+username)
        else:
            return render_template("Login.html",error='Invalid Login')


 

@app.route("/Signup")
def Signup():
    return render_template("Signup.html")


@app.route('/SignupAction', methods =['GET', 'POST'])
def SignupAction():
    if request.method == 'POST':
        name = request.form['t1']
        gender = request.form['t2']
        contact = request.form['t3']
        address = request.form['t4']
        email = request.form['t5']
        username = request.form['t6']
        password = request.form['t7']

        status = 'none'
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'amazonreviews',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select * FROM register")
            rows = cur.fetchall()
            for row in rows:
                if row[6] == username:
                    status = 'Given username already exists'
                    break
        if status == 'none':
            db_connection = pymysql.connect(host='127.0.0.1',port = 3308,user = 'root', password = 'root', database = 'amazonreviews',charset='utf8')
            db_cursor = db_connection.cursor()
            query = "INSERT INTO register(retailer_name,gender,contact_no,address,email,username,password) VALUES('"+name+"','"+gender+"','"+contact+"','"+address+"','"+email+"','"+username+"','"+password+"')"
            db_cursor.execute(query)
            db_connection.commit()
            print(db_cursor.rowcount, "Record Inserted")
            if db_cursor.rowcount == 1:
                status = "New Retailer Signup process completed"
        return render_template("Signup.html",error=status)                
         

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=9999, debug=True)
