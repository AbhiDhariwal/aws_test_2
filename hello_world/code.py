import pickle
import json
from flask_lambda import FlaskLambda
from flask import request,render_template


app = FlaskLambda(__name__)


tfidf = None
model = None
class_dict = {1: 'World', 2: 'Sports', 3: 'Business', 4: 'Science'}
try:
    with open('tfidf.pickle', 'rb') as f:
        tfidf = pickle.load(f)
        f.close()
except:
    tfidf = None

try:
    with open('save_nb_classifier.pkl', 'rb') as f1:
        model = pickle.load(f1)
        f1.close()
except:
    model = None


def tokenization(text):
    text = text.lower()
    lst = text.split()
    return lst


def remove_punctuations(lst):
    new_lst = []
    for i in lst:
        for j in '!\"#$%&\'()*+, -./:;<=>?@[\]^_`{|}~':
            i = i.replace(j, '')
        new_lst.append(i)
    return new_lst


def remove_numbers(lst):
    nodig_lst = []
    new_lst = []

    for i in lst:
        for j in '0123456789':
            i = i.replace(j, '')
        nodig_lst.append(i)
    for i in nodig_lst:
        if i != '':
            new_lst.append(i)
    return new_lst


def load_text(text):
    token = tokenization(text)
    token = remove_punctuations(token)
    token = remove_numbers(token)
    str_text = " ".join(token)
    return str_text


def predict(msg):
    pred = None
    updated_msg = load_text(msg)

    if tfidf is None:
        print("error in loading tokenizer")
        return "token_error " + updated_msg
    else:
        array_val = tfidf.transform([updated_msg]).toarray()

    if model is None:
        print("Trained model not available")
        return "model_error " +updated_msg
    else:
        pred = model.predict(array_val)


    if pred is None:
        return "pred_error" +updated_msg

    return class_dict[pred[0]]



@app.route('/hello', methods=['GET'])
def index_():
    if request.method == 'GET':
        return render_template('predict.html')
    else:
        return json_response({"invalid request"})



@app.route('/students', methods=['GET', 'POST'])
def put_list_students():
    if request.method == 'GET':
        return json_response({"body": "Welcome this is API call using aws lambda and I have deployed basic ml model which classify news headline into 4 categories :- sport, world, business, science",
                              "dataset_link": "https://www.kaggle.com/amananandrai/ag-news-classification-dataset"
                              })
    else:
        data = request.form.to_dict()
        response_data = {}

        for key in data.keys():
            prediction = predict(str(data[key]))
            response_data[data[key]] = prediction

        return json_response({"Model output": response_data})


@app.route('/students/<id>', methods=['GET'])
def get_patch_delete_student(id):
    if type(id) is str:
        input_data = id
    else:
        print("converting to string")
        input_data = str(id)

    prediction = predict(input_data)
    if request.method == 'GET':
        return json_response({"Model Output": prediction})


def json_response(data, response_code=200):
    return json.dumps(data), response_code, {'Content-Type': 'application/json'}

# if __name__ == '__main__':
#     app.run()