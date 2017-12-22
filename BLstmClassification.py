# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 23:45:19 2017
use BiLSTMs to classify a question into a category

@author: linkw
"""

class BiLstmTextClassification(object):

    def __init__(self):
        pass

    #read the file into dataframe
    def read_file(self,inp_path,encoding_type='utf-8',delimiter=','):
        import pandas as pd
        dataframe=pd.read_csv(inp_path,header=None,encoding = encoding_type,delimiter=delimiter)
        Y=dataframe.loc[:,1].apply(lambda x : x.strip())
        X=dataframe.loc[:,0]
        num_classes=len(Y.unique())
        return X,Y,num_classes
    
    #do nlp processing and save cleaned output to be used by fasttext
    def pre_processing(self,X):
        import spacy
        nlp = spacy.load('en_core_web_sm')
        import re
        r=re.compile(r'[\W]', re.U)
        X=X.apply(lambda x : re.sub('[\\s]+',' ',r.sub(' ',' '.join((str(token.tag_)+'parsed').lower() if str(token.tag_) in ('NN','VB') else str(token.lemma_).lower() for token in nlp(x)))))
        return X
    
    #use fasttext to learn word embeddings or load pretrained vectors
    def learn_embeddings(self,inp_path,out_path,emb_epoch=40,emb_lr=0.01,emb_dim=100,encoding_type='utf-8'):
        import fasttext
        fasttext.skipgram(inp_path,out_path,epoch=emb_epoch,lr=emb_lr,dim=emb_dim)
        from gensim.models.wrappers import FastText
        return FastText.load_fasttext_format(out_path+'.bin',encoding=encoding_type)
    
    #prepare all requirements of the neural net training
    def prepare(self,X,Y,emb_model,seq_length=20,stratify='n',test_split=0.2,emb_dim=100):
        #prepare data for use in NN
        #Convert text to sequences of word vectors
        X_seq=X.apply(lambda x : get_vecs(x,emb_model,seq_length))
        import numpy as np
        X_seq=np.concatenate(X_seq.values).astype(None)
        
        #encode labels in 1h vector
        from sklearn.preprocessing import LabelBinarizer
        label_encoder=LabelBinarizer()
        Y_coded=label_encoder.fit_transform(Y)
        
        #create test and train split
        from sklearn.model_selection import train_test_split
        if stratify=='y':
            x_train,x_test,y_train,y_test=train_test_split(X_seq,Y_coded,test_size=test_split,random_state=141289,stratify=Y_coded)
        else:
            x_train,x_test,y_train,y_test=train_test_split(X_seq,Y_coded,test_size=test_split,random_state=141289)
        return x_train,x_test,y_train,y_test,label_encoder   
        
    #training method which trains the nueral network
    def train(self,x_train,x_test,y_train,y_test,num_classes,seq_length=20,emb_dim=100,dropouts=(0.4,0.4),cells=100,bch_siz=50,epoch=45):
        #setup and train the nueral net
        from tensorflow.python.keras.models import Model
        from tensorflow.python.keras.layers import Bidirectional, Dense, Dropout, Input, LSTM
        inp=Input(shape=(seq_length,emb_dim))
        out=Bidirectional(LSTM(cells,return_sequences=True))(inp)
        out=Dropout(dropouts[0])(out)
        out=Bidirectional(LSTM(cells,return_sequences=False))(out)
        out=Dropout(dropouts[1])(out)
        out=Dense(num_classes, activation='softmax')(out)
        model=Model(inp, out)
        model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])
        #print (model.summary)
        model.fit(x_train, y_train, batch_size=bch_siz,
                  epochs=epoch, verbose=2, validation_data=(x_test, y_test))
        return model

    def train_with_embeddings(self,inp_path,encoding_type='utf-8',delimiter=',',out_path='',emb_epoch=40,emb_lr=0.01,emb_dim=100,seq_length=20,stratify='n',test_split=0.2,dropouts=(0.4,0.4),cells=100,bch_siz=50,epoch=45):
        import time
        import pickle
        import os
        #note start time
        start=time.time()
        
        #check if input file exists
        print('Reading data',flush=True)
        if os.path.isfile(inp_path):
            try:
                #try to create output directory
                if out_path is not '':
                    out_path=out_path
                else:    
                    out_path=os.path.join(os.path.dirname(inp_path),'models','BLSTM',os.path.basename(inp_path))    
                os.makedirs(out_path,exist_ok=True)
                
            except Exception:
                print(Exception)
            if 1:
                #read file and get labels and descriptions
                X,Y,num_classes=self.read_file(inp_path,encoding_type,delimiter=delimiter)
                
                #clean descriptions for use
                print('Cleaning data',flush=True)
                X=self.pre_processing(X)
                
                #create cleaned text file
                refined_text=os.path.join(out_path,'text.txt')
                X.to_csv(refined_text,index=False)
                
                #create embedding model output
                emb_model_output=os.path.join(out_path,'ftmodel')
                print('Learning embeddings',flush=True)
                emb_model=self.learn_embeddings(refined_text,emb_model_output,emb_epoch,emb_lr,emb_dim,encoding_type)
                
                #create and save train/test split, embedding matrix etc
                print('Preparing data for use in NN',flush=True)
                x_train,x_test,y_train,y_test,label_encoder=self.prepare(X,Y,emb_model,seq_length,stratify,test_split,emb_dim)
                with open(os.path.join(out_path,'label_enc.pkl'), 'wb') as f:
                    pickle.dump([label_encoder,seq_length,encoding_type], f,protocol=pickle.HIGHEST_PROTOCOL)
                
                #train the neural net
                print('Starting training',flush=True)
                model=self.train(x_train,x_test,y_train,y_test,num_classes,seq_length,emb_dim,dropouts,cells,bch_siz,epoch)
                model.save(os.path.join(out_path,'blstm.h5'))
                
                #report total time
                print('Training finished. Total time = %s seconds' % (time.time() - start))
            else:
                pass
        else:
            print("Invalid input path")

def get_vecs(rec,model,seq_length):
    #generate vector sequences
    import numpy as np
    wv=[]
    for word in rec.split():
        try:
            wv.append(model[word])
        except KeyError:
            continue 
        #pad to max length
    from tensorflow.python.keras.preprocessing import sequence
    return sequence.pad_sequences([wv],maxlen=seq_length,dtype=np.float64)
        
#predict functionality       
def predict(out_path,txt,top=1):
    import pickle
    import os
    if os.path.isfile(os.path.join(out_path,'label_enc.pkl')):      
        with open(os.path.join(out_path,'label_enc.pkl'), 'rb') as f:
            label_decoder,seq_len,encoding_type=pickle.load(f)
        
        #do preprocessing bit
        import spacy
        nlp = spacy.load('en_core_web_sm')
        import re
        r=re.compile(r'[\W]', re.U)
        txt=[re.sub('[\\s]+',' ',r.sub(' ',' '.join((str(token.tag_)+'parsed').lower() if str(token.tag_) in ('NN','VB') else str(token.lemma_).lower() for token in nlp(txt))))]
        
        #convert text to sequence 
        from gensim.models.wrappers import FastText
        emb_model=FastText.load_fasttext_format(os.path.join(out_path,'ftmodel.bin'),encoding=encoding_type)
        txt_seq=[get_vecs(txt[0],emb_model,20)]
        
        #load NN model and predict
        from tensorflow.python.keras.models import load_model
        model=load_model(os.path.join(out_path,'blstm.h5'))
        output=model.predict(txt_seq)
        
        #create binary sequences for top x predictions
        sorted_idx=(-output).argsort()
        import numpy as np
        label=np.zeros((top,len(output[0])))
        for i in range(0,top):
            label[i][sorted_idx[0][i]]=1
        
        #convert to txt labels
        return label_decoder.inverse_transform(label)
    else:
        return "Invalid output path!"        

net=BiLstmTextClassification()
net.train_with_embeddings(inp_path="D:\\Projects\\Github\\question_classification\\LabelledData.txt",out_path='D:\\ML_Models\\niki',encoding_type='ISO-8859-1',delimiter=',,,',emb_dim=100,seq_length=20,stratify='y',test_split=0.2,bch_siz=50,epoch=72)
print(predict(out_path='D:\\ML_Models\\niki',txt="What time does the train leave ?"))