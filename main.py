import json
import joblib
import numpy as np 
import pandas as pd
import time

from utils import *
from flask import jsonify
from flask import Flask, request

app = Flask(__name__)

###### INIT CONSTANT

with open("data/PRODUCT_LIST_DICT.json", "r") as f:
    PRODUCT_LIST_DICT = json.load(f)

# load model
model = joblib.load("./data/model_random_forest.joblib")

dt_train_preprocessed_unique = pd.read_csv("data/train_preprocessed_unique.csv", encoding="ISO-8859-1")

############################### 

@app.route('/')
def hello():
    return 'Hello, World!'

@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':   
        record = json.loads(request.data)   
        SEARCH_TERM = record['search_term']

        # SEARCH_TERM from request
        # SEARCH_TERM = "small angle bracket" 
        # SEARCH_TERM = "angle bracket"
        # SEARCH_TERM = "20 ton bottle jack"
        # SEARCH_TERM = "6 ton bottle jack" # good
        # SEARCH_TERM = "hydrant"
        # SEARCH_TERM = "Fan control" # good

        START_TIME = time.time()

        SEARCH_TERM_PREPROCESSED = get_root_form(normalizer_search_term(SEARCH_TERM))

        # Khởi tạo tập test giả lập
        dt_test = dt_train_preprocessed_unique.copy()
        dt_test["search_term"] = [SEARCH_TERM_PREPROCESSED] * len(dt_test)

        # tiền xử lý tập test và rút trích features
        dt_test['len_of_querry'] = [len(_.split()) for _ in dt_test['search_term'].values]
        dt_test['shared_words_whole_st_pt'] = [get_shared_words_whole(_) for _ in dt_test[['search_term', 'product_title']].values]
        dt_test['shared_words_whole_st_pdat'] = [get_shared_words_whole(_) for _ in dt_test[['search_term', 'product_description_attributes']].values]
        dt_test['shared_words_part_st_pt'] = [get_shared_words_part(_) for _ in dt_test[['search_term', 'product_title']].values]
        dt_test['similarity_st_pt'] = [Levenshtein.ratio(_[0], _[1]) for _ in dt_test[['search_term', 'product_title']].values]

        # ở tập test cũng gỡ hết các trường khác chỉ để lại các trường mang features
        X_test = dt_test.drop(['product_title','search_term','product_description_attributes'],axis=1).values

        # dự đoán với mô hình da train
        y_pred_tmp = model.predict(X_test)

        k_top = 20
        product_uid_test = dt_test["product_uid"]
        score_relevance_sorted, product_uid_sorted = zip(*sorted(zip(y_pred_tmp, product_uid_test), reverse=True))

        ### TRẢ KẾT QUẢ VỀ
        result = []
        for i in range(k_top):
            res = { 
                "product_uid": product_uid_sorted[i],
                "product_title": PRODUCT_LIST_DICT[str(product_uid_sorted[i])],
                "score_relevance": score_relevance_sorted[i]
            }
            result.append(res)

        END_TIME = time.time()

        elapsed_time = END_TIME - START_TIME
        print("execution time of a query: {} seconds".format(elapsed_time))
        print(result)
        
        return jsonify({
                "elapsed_time": elapsed_time,
                "top_result": result
            })

if __name__=="__main__":
    app.run("0.0.0.0")