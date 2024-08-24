from functools import lru_cache
from pymorphy3 import MorphAnalyzer #type: ignore
from stop_words import get_stop_words # type: ignore
import re
import pandas

class DataFilter():
    def __init__(self):
       self.morph_analyzer = MorphAnalyzer()
       self.stop_words = set(get_stop_words("en"))
       self.word_regex = re.compile(r'\w+')

    def transform(self, df) -> pandas.DataFrame:
        df = df.drop_duplicates()
        df['text'] = df['Review'].apply(self.lemmatize) 
        if 'Sentiment' in df.columns:
            df['Label']=df['Sentiment'].map({'Positive':1,'Negative':0})
        return df
    
    @lru_cache
    def get_normal_form(self, word: str) -> str:
       return self.morph_analyzer.parse(word)[0].normal_form

    def lemmatize(self, line):
        return ' '.join(
            filter (
            lambda word: word not in self.stop_words,
            map(
                lambda x: self.get_normal_form(str(x)),
                self.word_regex.findall(line)
            )
            )
        )