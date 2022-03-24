import re
import nltk
import numpy as np 
import pandas as pd
import seaborn as sns # vừa đếm sô lượng vừa vẽ lên biểu đồ
import Levenshtein # Đo khoảng cách giữa 2 string (độ tương đồng)
from matplotlib import pyplot as plt # vẽ biểu đồ
from matplotlib_venn import venn2 # vẽ biểu đồ nhưng là dạng tròn
from nltk.corpus import stopwords # các stopwords (the, or, and, ...)
from nltk.stem.snowball import SnowballStemmer # kỹ thuật stemmer đưa về từ gốc
from nltk.stem import WordNetLemmatizer # kỹ thuật Lemmatizer đưa về từ gốc
stemmer = SnowballStemmer('english') # stemer với ngôn ngữ là tiếng anh
nltk.download('wordnet') # dowload bộ từ điển cho Lemmatizer
nltk.download('stopwords') # dowload bộ các stopwords
stop_words = set(stopwords.words('english')) # các stopwords là tiếng anh 



def normalizer_sentences(string):
    string = string.lower()

    # gỡ các ký hiệu sản phẩm không liên quan
    string = string.replace('-', '')
    string = string.replace('in.', 'inch')
    string = string.replace('ft.', 'foot')
    string = string.replace('-oz. ', ' ')
    string = string.replace('oz.', ' ')
    string = string.replace('sq.', ' ')
    string = string.replace('Gal.', ' ')
    string = string.replace('lb.', ' ')
    string = string.replace('cu.', ' ')
    string = string.replace('O.D.', ' ')
    string = string.replace('sq.', ' ')
    string = string.replace('st.', ' ')
    string = string.replace('lb.', ' ')
    string = string.replace('Dia.', ' ')
    string = string.replace('dia.', ' ')

    # Thay thế tất cả các ký tự đặc biệc bằng 1 space
    string = re.sub(r'[^a-zA-Z0-9]+', ' ', string)

    # Thay thế các ký tự đứng một mình bằng 1 space
    string = re.sub(r'\b[a-zA-Z]\b', ' ', string)

    # Gộp các space đứng liền nhau thành 1 space
    string = re.sub(r'\s+', ' ', string)

    return string

def normalizer_search_term(string):
    string = string.lower()

    # gỡ các ký hiệu sản phẩm không liên quan
    string = string.replace('in.', 'inch')
    string = string.replace('ft.', 'foot')
    string = string.replace('ft', 'foot')

    # key word bị viết riêng lẻ
    string = string.replace('r ', 'r')

    # Thay thế tất cả các ký tự đặc biệc bằng 1 space
    string = re.sub(r'[^a-zA-Z0-9]+', ' ', string)

    # Gộp các space đứng liền nhau thành 1 space
    string = re.sub(r'\s+', ' ', string)

    return string

# Đưa các từ về từ gốc của nó.
# ta sẽ thực áp dụng hàm này cho cả 3 trường title, search_term và des_attr
def get_root_form(string):
    lemmatizer = WordNetLemmatizer()
    # return ' '.join(map(lambda x: stemmer.stem(x), string.split())) # kỹ thuật stemmer
    # return ' '.join(map(lambda x: lemmatizer.lemmatize(x), string.split())) # kỹ thuật lemmatizer
    return ' '.join(map(lambda x: lemmatizer.lemmatize(x), list(map(lambda x: stemmer.stem(x), string.split()))))  # sử dụng cả 2 kỹ thuật

# Tìm các tokens chung 
# tách tokens của 2 câu bằng hàm split()
# rồi lần lượt tính sự xuất hiện của tokens ở câu 1 trong câu 2
def str_common_tokens(sentence_1, sentence_2):
    # return sum(1 for word in str(sentence_2).split() if word in set(str(sentence_1).split()))
    return len(set(sentence_1.split()).intersection(sentence_2.split()))

# Các từ chung một phần nào đấy
# dể lấy các từ chung một phần thì ta không tách tokens (ko dùng hàm split() ở đây)
# lần lượt tính sự xuất hiện của các từ trong câu 1 ở câu 2
def str_common_word(sentence_1, sentence_2):
    return sum(1 for word in str(sentence_2) if word in set(sentence_1))

# Tính tổng tất cả các Tokens xuất hiện "toàn phần"
def get_shared_words_whole(row_data):
    return str_common_tokens(row_data[0], row_data[1])

# Tính tổng tất cả các Từ xuất hiện "một phần"
def get_shared_words_part(row_data):
    return str_common_word(row_data[0], row_data[1])

