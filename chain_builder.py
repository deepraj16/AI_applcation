from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

def create_qa_chain(llm, retriever):
    """Create QA chain with Marathi prompt template"""
    marathi_prompt = PromptTemplate(
        template="""तुम्ही एक अनुभवी महाविद्यालयीन प्रवेश सल्लागार आहात. खालील संदर्भाच्या आधारे विद्यार्थ्याच्या प्रश्नाचे अचूक आणि उपयुक्त उत्तर मराठीत द्या.

महत्वाचे सूचना:
- फक्त दिलेल्या संदर्भातील माहिती वापरा
- जर संदर्भात माहिती नसेल तर "या प्रश्नाची माहिती सध्या उपलब्ध दस्तऐवजात नाही" असे सांगा
- स्पष्ट, सोप्या आणि विनम्र मराठी भाषेत उत्तर द्या
- आवश्यक तपशील आणि चरण दिल्यास उत्तम
- जर तारीख, फी, किंवा संख्या आहेत तर त्या अचूकपणे द्या

संदर्भ माहिती:
{context}

विद्यार्थ्याचा प्रश्न: {question}

उत्तर (मराठीत):""",
        input_variables=["context", "question"]
    )
    
    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": marathi_prompt},
        return_source_documents=False,  # cleaner output
        verbose=False
    )
    
    return qa_chain
