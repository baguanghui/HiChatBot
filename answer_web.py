# -*- coding: utf-8 -*-
from bottle import route,run,static_file
from bottle import template,view,request
import json
import difflib
from text_search import TextIndex
from answer import Answer
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

@route('/chat')
@route('/')
@view('chat_web')
def index(): 
    '''
     pc聊天页面
    '''
    return None

@route('/chat/h5')
@route('/h5')
@view('chat_web_h5')
def index_h5():
    '''
     h5,移动设备聊天页面
    '''
    return None

text_index = TextIndex() #whoosh问答索引
answer = Answer()  #模型

@route('/getAnswer', method = "GET")
def get_answer():
    question = str(request.query.question)
    print(question)
    response_json = get_answer_response(question) # 获取问答
    return response_json

@route('/getAnswer', method = "POST")
def post_answer():   
    request.POST.decode('utf-8') #解决乱码的问题
    question = request.forms.question   
    response_json = get_answer_response(question) # 获取问答
    return response_json 



def get_answer_response(question):
    '''
        获取用户回答
        return json格式数据
    '''
    if question.strip() == '':
        return '不知道你想说什么'
    retAnswer = searchAnswer(question) #检索最匹配的回答
    if not retAnswer: #如果没有检索到，让交给模型预测回答
        retAnswer = answer.predict_answer(question)#//模型预测 
        print('模型预测:',retAnswer)
    else:
        print('问答检索:',retAnswer)
    feed_data = {'question':question,'answer':str(retAnswer)} #字典格式问答对    
    return json.dumps(feed_data,ensure_ascii=False) #json格式数据，返回url请求接口

def searchAnswer(question):
    '''
        在索引库中检索回答对，并对检索回答句子相似度匹配
    '''
    results = text_index.search_answer(question) #对答检索
    if(len(results) > 0):
        sent_ratios = [difflib.SequenceMatcher(None,question,sent['question']).ratio() for sent in results]
        max_ratio = max(sent_ratios)
        if max_ratio >= 0.66:
            max_ratio_index = sent_ratios.index(max_ratio)
            return results[max_ratio_index]["answer"]
        else:
            return None        
    else:
        return None


assets_path = './assets'

@route(r'/assets/<filename:re:.*\.css|.*\.js|.*\.png|.*\.jpg|.*\.gif>')
def server_static(filename):
    """定义/assets/下的静态(css,js,图片)资源路径"""
    return static_file(filename, root=assets_path)

if __name__ == '__main__':    
    run(host='localhost', port=8080, debug=True, reloader=True) #开启服务