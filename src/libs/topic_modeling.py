import pandas as pd
import gensim
from gensim import corpora, models
from ast import literal_eval

import pyLDAvis
from pyLDAvis.gensim_models import prepare

from nlp_utils import generate_N_grams, process_text
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")


def create_topics(data: pd.DataFrame, text_column: str, method: str = "bow", num_topics: int = 5, n_gram = 1, stopwords: str = [],
                  filename: str = "topic_modeling_anexos.html"):
    """
    """
    text = data[text_column].apply(lambda x: generate_N_grams(
            process_text( x ,additional_stopwords=stopwords, keep_as_list=True), n_gram
        )
    )
    dictionary_ = gensim.corpora.Dictionary(text)
    
    bow_corpus = [dictionary_.doc2bow(doc) for doc in text]
    
    if method != "bow":
        tfidf = models.TfidfModel(bow_corpus)
        corpus_tfidf = tfidf[bow_corpus]
        lda_model = gensim.models.LdaMulticore(
            corpus_tfidf, num_topics=num_topics, id2word=dictionary_, passes=50, workers=4
        )
    else:
        lda_model = gensim.models.LdaMulticore(
            bow_corpus, num_topics=num_topics, id2word=dictionary_, passes=50, workers=2
        )
        
    for idx, topic in lda_model.print_topics(-1):
        print('Topic: {} Word: {}'.format(idx+1, topic))
        
    LDAvis_prepared = prepare(lda_, bow_corpus, dictionary_)
    pyLDAvis.save_html(LDAvis_prepared, filename )
        
    return lda_model


def make_lda_plot(text: str, lda_model, n_gram):
    print(f"Texto base: {text}", "\n", f"N-grams: {n_gram}")
    text = generate_N_grams(
            process_text( text, keep_as_list=True), n_gram
        )
    dictionary_ = gensim.corpora.Dictionary([text])
    document = dictionary_.doc2bow(text)
    
    #plt.bar(lda_model[document])
    df = pd.DataFrame(lda_model[document], columns = ["Topico", "Similaridad"])
    sns.barplot(data=df, y="Similaridad", x ="Topico")
    plt.title(f"Modelamiento de topicos")