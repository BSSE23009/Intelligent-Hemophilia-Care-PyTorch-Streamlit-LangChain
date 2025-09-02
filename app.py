import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import io
from typing import List
from langchain.schema import Document
from dotenv import load_dotenv
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()



@st.cache_resource
def load_model():
    resnet = models.resnet18(pretrained=True)
    for param in resnet.parameters():
        param.requires_grad = False
    num_features = resnet.fc.in_features
    resnet.fc = torch.nn.Linear(num_features, 9)
    resnet.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
    resnet.eval()
    return resnet

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Lambda(lambda pic: torch.tensor(
        pic.convert("RGB").getdata()
    ).view(224, 224, 3).permute(2, 0, 1) / 255.0),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

my_data: List[Document] = []
def filter_docs(docs: List[Document]) -> List[Document]:
    return [Document(page_content=doc.page_content, metadata={"source": doc.metadata.get("source")}) for doc in docs]

medical_new_docs = filter_docs(my_data)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
my_chunks = text_splitter.split_documents(medical_new_docs)

pc_api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=pc_api_key)
index_name = "hemophilia-care-ai"
if not pc.has_index(index_name):
    pc.create_index(name=index_name, dimension=1536, metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1"))
index = pc.Index(index_name)

embedding = OpenAIEmbeddings(model="text-embedding-3-small")

small_chunks = text_splitter.split_documents(my_chunks)
batch_size = 50
for i in range(0, len(small_chunks), batch_size):
    batch = small_chunks[i:i+batch_size]
    vectors = [
        {"id": f"doc-{i+j}", "values": embedding.embed_query(doc.page_content),
         "metadata": {"text": doc.page_content}}
        for j, doc in enumerate(batch)
    ]
    index.upsert(vectors=vectors)

docsearch_client = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embedding)
retriever = docsearch_client.as_retriever(search_type="similarity", search_kwargs={"k": 3})

chatModel = ChatOpenAI(model="gpt-4o")
system_prompt = (
    "You are a knowledgeable and cautious medical assistant. "
    "Answer questions using ONLY the retrieved context below. "
    "If not found, say you don’t know. Keep it concise, factual, and medically accurate."
    "\n\nContext:\n{context}"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

st.set_page_config(page_title="Hemophilia Care AI", layout="wide")
st.title("🩸 Hemophilia Care AI")

tab1, tab2 = st.tabs(["Bruise Detection", "💬 Medical Q&A"])

with tab1:
    st.subheader("Upload an Image")
    uploaded_file = st.file_uploader(" ", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(io.BytesIO(uploaded_file.read()))
        st.image(image, caption="Uploaded Image", use_column_width=True)
        with st.spinner("Analyzing image..."):
            img_tensor = transform(image).unsqueeze(0)
            with torch.no_grad():
                outputs = model(img_tensor)
                _, predicted = torch.max(outputs, 1)
        st.success(f"**Predicted Class:** {predicted.item()}")

with tab2:
    st.subheader("Ask a Medical Question")
    user_q = st.text_input("Type your question here")
    if st.button("Get Answer"):
        with st.spinner("Thinking..."):
            question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
            rag_chain = create_retrieval_chain(retriever, question_answer_chain)
            response = rag_chain.invoke({"input": user_q})
        st.info(response["answer"])
