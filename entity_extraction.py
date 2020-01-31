
import pandas as pd
from nltk import word_tokenize
from nltk import pos_tag    #  parts of speech tagging.
from nltk import ne_chunk   # Named Entity Recognition.
from nltk import Tree       # To handle tree object returned by ne_chunk.
from nltk.sem.relextract import NE_CLASSES  # To filter classes
from nltk import help


path = 'enterprises.xlsx' # set reviews file path.

data = pd.read_excel(path)
data.head(n=2)
EnterpriseName = list(data["EnterpriseName"])

extracted_entities = []
for enterprise in EnterpriseName:
    temp_list = [word for (word, pos) in pos_tag(word_tokenize(str(enterprise))) if pos in ['NN', 'NS', 'VB', 'ADV']]
    extracted_entities.append(temp_list)


"""
    for word, pos in pos_tag(word_tokenize(str(enterprise))):
        if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS'):
            temp_list.append(word)
"""
len(EnterpriseName), len(extracted_entities)
extracted_entities[0:50]
#postagged = pos_tag(word_tokenize(Text))
#posdf = pd.DataFrame(postagged, columns=['Word', 'Tag'])
#print(posdf[0:5])

#entities_tagged = ne_chunk(pos_tag(word_tokenize(Text)))    # ne_chunk needs words to be pos-tagged.
#print(entities_tagged[20:30])

extracted_name = [' '.join(words) for words in extracted_entities]
data['ExtractedName'] = extracted_name
data.to_csv('newdata(NN_NS_VB_ADV).csv',index=False)


