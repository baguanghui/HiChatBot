from whoosh.index import create_in,open_dir
from whoosh.fields import *
from whoosh.qparser import QueryParser
import os
import jieba
import pandas as pd
from jieba.analyse import ChineseAnalyzer

class TextIndex:
    def __init__(self):
        self.index_dir = 'index'
        self.data_file_path = './dialog/all_question_answer.csv'
        jieba_analyzer = ChineseAnalyzer()
        self.schema = Schema(question=TEXT(stored=True,analyzer=jieba_analyzer), answer=TEXT(stored=True))
        self.ix = self._create_index()
        self.ix_search = self.ix.searcher()

    def _create_index(self):
        jieba_analyzer = ChineseAnalyzer()
        if not os.path.exists(self.index_dir):
            os.mkdir(self.index_dir)
            ix = create_in(self.index_dir,self.schema,'question_index')
        else:
            # files_under_dir = os.listdir(self.index_dir)
            # # if('')
            try:
                ix = open_dir(self.index_dir,'question_index')
            except:
                print('except')
                ix = create_in(self.index_dir, self.schema, 'question_index')
        return ix

    def recreate_index(self):
        '''
        重新创建索引
        :return:
        '''
        confirm = input("你确定删除索引再重新创建索引（y/n）:")
        if confirm not in "yY":
            return
        if os.path.exists(self.index_dir):
            files_under_dir = os.listdir(self.index_dir)
            for file in files_under_dir:
                try:
                    os.remove(os.path.join(self.index_dir, file))
                except:
                    continue
            # os.rmdir(self.index_dir)
        self.ix = self._create_index()
        self.ix_search = self.ix.searcher()

    def search_answer(self,question):
        '''
        调用索引引擎，查询匹配问答，返回评分高的前10条问答对
        '''
        parser = QueryParser("question", self.ix.schema)
        words = jieba.lcut_for_search(question)
        sents_word = ' '.join(words)
        query = parser.parse(sents_word)
        results = self.ix_search.search(query)
        return results

    def load_document(self):
        '''
        向索引添加问答对
        '''
        ix_writer = self.ix.writer()
        # 把数据写入索引
        corpus_df = pd.read_csv(self.data_file_path, names=['question', 'answer'])
        for dialog in corpus_df.values[1:]:
            ix_writer.add_document(question=str(dialog[0]), answer=str(dialog[1]))
            print(str(dialog[0]), str(dialog[1]))
        ix_writer.commit()

    


    def __del__(self):
        self.ix_search.close()

if __name__ == '__main__':
    index = TextIndex()
    # index.recreate_index()
    # index.load_document()
    while True:
        user_input = input("me->")
        if len(user_input) == 0:
            print('请重新输入：')
            continue;

        if user_input.strip() == 'exit':
            break

        results = index.search_answer(user_input)
        print(type(results),type(list(results)))
        if (len(results) > 0):
            # print(list(results))
            print("bot->", results[0]['answer']) 

