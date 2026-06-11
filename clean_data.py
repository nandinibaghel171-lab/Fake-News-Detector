import pandas as pd 
import re
# Data load karo
df =pd.read_csv('data/fake.csv')
print("original shape:",df.shape)
# sirf kaam ki rows and columns rakho
df=df[['title','text']]
#missing values hatao
df=df.dropna()
print("cleared shape:",df.shape)
#text ko clean kare
def clean_text(text):
    text=text.lower() #tlower caset
    text=re.sub(r'[^\d+]', '',text) #punctuator  hatado 
    text=re.sub(r'[^\w\s]','',text) #punctuator  hatado  text
    text=re.sub(r'[^\s+]', '',text) #extra spaces hatado
    return text.strip()

df['title']=df['title'].apply(clean_text)
df['text']=df['text'].apply(clean_text)
print("cleared data:",df.shape)
df['label']=1
df.to_csv('data/cleaned_fake.csv',index=False)
print("Data cleaned and saved ")
print(df.head())