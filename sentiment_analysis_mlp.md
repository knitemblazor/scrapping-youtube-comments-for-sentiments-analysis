

```python
from google.colab import drive
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
from gensim.models import Word2Vec
import tensorflow as tf



```

*Labeling  the statements as  neutral(0), positive(1) and negative(2) *

---



---




```python
drive.mount('/content/gdrive')

file = open('/content/gdrive/My Drive/latest/comments.txt',"r") #comments of  delhi video tours  especially food related
comments= file.read().splitlines()  
print(len(comments),"of comments in the given dataset")
file.close()

 filename = '/content/gdrive/My Drive/latest/negative.txt'
file = open(filename,"r" )
negative = file.read().splitlines()
file.close()   

filename = '/content/gdrive/My Drive/latest/positive.txt'
file = open(filename,"r" )
positive = file.read().splitlines()
file.close()

lab=[]
for i in range(len(comments)):
    lab.append(0)
def label(comments,negative,positive):
    for i in range(len(comments)):
        words=comments[i].split()
        for j in range(len(words)):
            for k in range(len(positive)):
                if(words[j]==positive[k]):
                    lab[i]=1
                
            for l in range(len(negative)): 
                if(words[j]==negative[l]):
                    lab[i]=2
                
           
label(comments,negative,positive)
np.save('/content/gdrive/My Drive/latest/label.npy', lab)


```

    Drive already mounted at /content/gdrive; to attempt to forcibly remount, call drive.mount("/content/gdrive", force_remount=True).
    30080 of comments in the given dataset


*Vocabulary building section *

---



---




```python


filename = '/content/gdrive/My Drive/latest/stopwords.txt'
file = open(filename,"r" )
stopwords = file.read().splitlines()
file.close() 

words=[]
sentences =[]
k = []
for i in range(len(comments)):
    sentences.append(0)
for i in range(0,len(comments)):
    sentences[i]=comments[i].split()
    for j in sentences[i]:   
        if(len(j)<9):         #part of filtering
            words.append(j)
wordsf= []
words = [word.lower() for word in words]
words = [word for word in words if word.isalpha()]# setting the vocabulary
wordsf=(set(words).difference(stopwords))      
 
print(len(words))
words=list(wordsf)
print(len(sentences))




```

    338342
    30080


*Removing larger sentences and retaining only ones with the length specified or lesser ones*

---



---




```python
j=[]
sentencesf=[]         #calculating the length of all sentences ie the no. of words
labelf=[]
lab=np.load('/content/gdrive/My Drive/latest/label.npy')
for i in range(len(sentences)):
  j.append(len(sentences[i]))


for i in range(len(j)):
  if(j[i]<=68):
    sentencesf.append(sentences[i])
    labelf.append(lab[i])
```


*Word2Vec*

---



---




```python
model = Word2Vec(sentencesf,size=68, min_count=1)#word vector dim=28
# summarize the loaded model
print(model)
# summarize vocabulary
words = list(model.wv.vocab)
# print(words)
# access vector for one word
# print(model[ 'was' ])
# save model


```

    Word2Vec(vocab=44612, size=68, alpha=0.025)


*Padding for MLP*

---



---




```python
zerovector=[]
for i in range(68):
  zerovector.append(0)
# print(label)
def sentencevector(sent):
  vector=[]
  for word in sent:
       vector.append(model[word])
  if(len(sent)<68):
    for i in range(len(sent),68):
      vector.append(zerovector)
  return(vector)


```

*Splitting into training and test data*

---



---




```python
Xtotal=[]
Ytotal=labelf
for i in range(len(sentencesf)):
  a=np.array(sentencevector(sentencesf[i]))
  Xtotal.append(a)

Xtrain=np.array(Xtotal[:20000])
Xtest=np.array(Xtotal[20000:27263])
Y_train=np.array(Ytotal[:20000])
Y_test=np.array(Ytotal[20000:27263])
X_train = Xtrain.reshape(Xtrain.shape[0], 68, 68 , 1).astype('float32')
X_test=Xtest.reshape(Xtest.shape[0], 68, 68 , 1).astype('float32')



```

    /usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:8: DeprecationWarning: Call to deprecated `__getitem__` (Method will be removed in 4.0.0, use self.wv.__getitem__() instead).
      


*Model construction*

---



---




```python
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(70, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.3),
  tf.keras.layers.Dense(10, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(3, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=10)
model.evaluate(X_test, Y_test)

```

    Epoch 1/10
    20000/20000 [==============================] - 6s 316us/step - loss: 0.8909 - acc: 0.5937
    Epoch 2/10
    20000/20000 [==============================] - 5s 259us/step - loss: 0.8039 - acc: 0.6329
    Epoch 3/10
    20000/20000 [==============================] - 5s 238us/step - loss: 0.7806 - acc: 0.6428
    Epoch 4/10
    20000/20000 [==============================] - 5s 244us/step - loss: 0.7657 - acc: 0.6489
    Epoch 5/10
    20000/20000 [==============================] - 5s 239us/step - loss: 0.7546 - acc: 0.6559
    Epoch 6/10
    20000/20000 [==============================] - 5s 247us/step - loss: 0.7401 - acc: 0.6643
    Epoch 7/10
    20000/20000 [==============================] - 5s 243us/step - loss: 0.7304 - acc: 0.6711
    Epoch 8/10
    20000/20000 [==============================] - 5s 240us/step - loss: 0.7207 - acc: 0.6782
    Epoch 9/10
    20000/20000 [==============================] - 5s 244us/step - loss: 0.7064 - acc: 0.6884
    Epoch 10/10
    20000/20000 [==============================] - 5s 244us/step - loss: 0.6954 - acc: 0.6905
    7263/7263 [==============================] - 1s 150us/step





    [0.8011589164200785, 0.6504199365832237]


