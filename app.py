from langchain_groq import ChatGroq
import os
from data import extract_text_from_pdf, clean_text
from nltk.tokenize import sent_tokenize
from embeddings import *
from model import *
from dotenv import load_dotenv


load_dotenv()

pdf_path= 'dataset_new.pdf'
text = extract_text_from_pdf(pdf_path)

text = clean_text(text)

sentences = sent_tokenize(text)

chunks = chunk_text_with_overlap(sentences)

embeddings = generate_embeddings(chunks)

save_to_faiss(embeddings)

index = load_faiss_index("index.faiss")



def create_context(query, relevant_chunks):
    context = "\n".join([f"Section {i+1}: {chunk}" for i, chunk in enumerate(relevant_chunks)])
    prompt = f"""
    You are an assistant that provides detailed and relevant answers to the query based on the context provided. 
    Please ensure that the answer is informative, clear, and precise.
    
    Instructions:
    Please provide the most accurate response based on the question.
    Answer should begin with "Answer is : ".
    And If the answer of any question is not there in content then 
    please reply with the line " We don't have answer to this question, kindly contact the business".
    keep a formal tone of the response.
    Make sure the answer directly addresses the query.
    Do not include any irrelevant information.
    
    Context:
    {context}

    Question: {query}

    Answer:
    """
    
    return prompt



groq_api_key=os.getenv('GROQ_API_KEY')

llm=ChatGroq(groq_api_key=groq_api_key,
             model_name="Llama3-8b-8192")


def generate_answer(query, relevant_chunks):
    
    prompt = create_context(query, relevant_chunks)

    response = llm.invoke(prompt)
    
    answer = response.content.strip()  

    return answer


def main():
    while True:
        query = input("Please enter your query (or type 'exit' to stop): \n")
        if query.lower() == 'exit':
            print("Exiting the query system. Goodbye!")
            break

        relevant_chunks, distances = perform_semantic_search(query, index, chunks)

        answer = generate_answer(query, relevant_chunks)
        print(answer)



if __name__ == "__main__":
    main()