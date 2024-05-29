import streamlit as st
import pickle
import re
import nltk

nltk.download('stopwords')
nltk.download('punkt')

clf = pickle.load(open('clf.pkl','rb'))
tfidf = pickle.load(open('tfidf.pkl','rb'))

def clean_resume(resume_text):
    clean_text = re.sub('http\S+\s*', ' ', resume_text)
    clean_text = re.sub('RT|cc', ' ', clean_text)
    clean_text = re.sub('#\S+', '', clean_text)
    clean_text = re.sub('@\S+', '  ', clean_text)
    clean_text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', clean_text)
    clean_text = re.sub(r'[^\x00-\x7f]', r' ', clean_text)
    clean_text = re.sub('\s+', ' ', clean_text)
    return clean_text

def main():
    st.title("Resume Screening App")
    uploaded_file = st.file_uploader('Upload Resume',type=['txt','pdf'])

    if uploaded_file is not None:
        try:
            resume_bytes = uploaded_file.read()
            resume_text = resume_bytes.decode('utf-8')
        except UnicodeDecodeError:
            resume_text = resume_bytes.decode('latin-1')
        
        cleaned_resume = clean_resume(resume_text=resume_text)
        input_data = tfidf.transform([cleaned_resume])
        predict_id = clf.predict(input_data)[0] 

        cat_map = {0: 'Advocate',
                    1: 'Arts',
                    2: 'Automation Testing',
                    3: 'Blockchain',
                    4: 'Business Analyst',
                    5: 'Civil Engineer',
                    6: 'Data Science',
                    7: 'Database',
                    8: 'DevOps Engineer',
                    9: 'DotNet Developer',
                    10: 'ETL Developer',
                    11: 'Electrical Engineering',
                    12: 'HR',
                    13: 'Hadoop',
                    14: 'Health and fitness',
                    15: 'Java Developer',
                    16: 'Mechanical Engineer',
                    17: 'Network Security Engineer',
                    18: 'Operations Manager',
                    19: 'PMO',
                    20: 'Python Developer',
                    21: 'SAP Developer',
                    22: 'Sales',
                    23: 'Testing',
                    24: 'Web Designing'}
        
        st.write(cat_map[predict_id])
    
if __name__ == "__main__":
    main()