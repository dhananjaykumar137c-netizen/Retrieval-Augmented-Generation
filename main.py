from pypdf import PdfReader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

file_name = "dbms_miniproject.pdf"

# loader = PdfReader(file_name)
loader = PyPDFLoader(file_name)
pages = loader.load()
for i in pages: 
    print(i)
    print("\n----------------------------------------------------------------------------------------------------------\n")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    length_function=len,
)

texts = text_splitter.split_documents(pages)
print(f"Created {len(texts)} documents")
print("this is python")
