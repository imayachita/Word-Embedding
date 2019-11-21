import numpy as np
from nltk import sent_tokenize
import PyPDF2
import pandas as pd
import pdftotext
import re
import os
from tqdm import tqdm

def is_ascii(s):
    return all(ord(c) < 128 for c in s)

def read_pdf(folder,output_file):
    #read pdf files in a folder ONLY UP TO TABLE OF CONTENT (wellname tagging) with pdftotext
    text_data=[]
    results=[]

    # files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    # print('list of files: ', files)


    file_list=[]
    for root, dirs, files in os.walk(folder):
        for file in files:
            file_list.append(os.path.join(root,file))

    # print('File list: ', file_list)
    print('Number of files: ', len(file_list))
    files = list(filter(lambda f: f.endswith(('.pdf','.PDF')), file_list))

    txt_id=0
    num_corrupted_files = 0

    for i,file in enumerate(tqdm(files)):
        # pdfFileObj = open(os.path.join(folder,file), 'rb')
        pdfFileObj = open(file, 'rb')
        try:
            pdfReader = pdftotext.PDF(pdfFileObj)
            for page in range(len(pdfReader)):
                #sentence tokenize the pdf file
                text_doc=[]
                tokenized = sent_tokenize(pdfReader[page])
                t = []
                for token in tokenized:
                    token = token.replace('\n',' ')
                    token = re.sub(' +', ' ', token)
                    t.append(token)

                text_doc.append(t)

                #convert the text to dictionary -- equivalent to bible.json
                '''
                Doc_ID = document number
                Page_ID = the page number
                Text = the corresponding sentence
                Sent_ID = sentence number in that particular page
                ID = sentence number in the whole corpus
                File = file name
                '''

                for index, doc in enumerate(text_doc[0]):
                    if len(doc)>10 and is_ascii(doc):
                        #create dictionary to map all of the texts with doc_id, page_id, and sent_idx
                        data = dict(
                                    Doc_ID=i,
                                    Page_ID=page,
                                    Sent_ID=index,
                                    # ID = txt_id,
                                    Text=doc.replace(';',','),
                                    File = file,
                                    Abs_path = os.path.abspath(file)
                                   )
                        text_data.append(data)
                        txt_id+=1

        except:
            print('cant read this file: ', file)
            num_corrupted_files+=1

    text_df = pd.DataFrame(text_data)
    print('num_corrupted_files: ', num_corrupted_files)
    text_df.to_csv(output_file, index=False)
    return text_df

#main program
folder="./data"
output_file='facilities_text.csv'
read_pdf(folder,output_file)
print('Done')
