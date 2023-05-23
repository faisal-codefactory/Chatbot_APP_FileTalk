from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, DataFrameLoader, CSVLoader, GitLoader
import nbformat
from nbconvert import PythonExporter
import pandas as pd
from pathlib import Path
import zipfile
from io import BytesIO


from langchain.text_splitter import RecursiveCharacterTextSplitter


class IncomingFileProcessor():
    def __init__(self, chunk_size=750) -> None:
        self.chunk_size = chunk_size
        # pass

    def get_text_splits(self, text_file: str):
        with open(text_file, 'r') as txt:
            data = txt.read()
        textsplit = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=15, length_function=len)

        doc_list = textsplit.split_text(data)
        return doc_list

    def get_pdf_splits(self, pdf_file: str):
        loader = PyPDFLoader(pdf_file)
        pages = loader.load_and_split()
        textsplit = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=15, length_function=len)

        doc_list = []
        for pg in pages:
            pg_splits = textsplit.split_text(pg.page_content)
            doc_list.extend(pg_splits)
        return doc_list

    def get_docx_splits(self, docx_file: str):
        loader = Docx2txtLoader(str(docx_file))
        txt = loader.load()
        textsplit = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=15, length_function=len)

        doc_list = textsplit.split_text(txt[0].page_content)
        return doc_list

    # def get_zip_pdf_splits(self, zip_pdf_file: str):
    def get_zip_pdf_splits(self, zip_stream: BytesIO):
        pdf_paths = []
        with zipfile.ZipFile(zip_stream, 'r') as zip_file:
            doc_list = []
            for name in zip_file.namelist():
                if name.lower().endswith('.pdf'):
                    pdf_paths.append(zip_file.extract(name))
        doc_list = []
        for pdf_file in pdf_paths:
            tmp = self.get_pdf_splits(pdf_file)
            doc_list.extend(tmp)
        return doc_list

    def get_zip_docx_splits(self, zip_docx_file: str):
        file_paths = []
        with zipfile.ZipFile(zip_docx_file, 'r') as zip_file:
            doc_list = []
            for name in zip_file.namelist():
                if name.lower().endswith('.docx'):
                    file_paths.append(zip_file.extract(name))
        doc_list = []
        for extracted_file in file_paths:
            tmp = self.get_docx_splits(extracted_file)
            doc_list.extend(tmp)
        return doc_list

    def get_zip_splits(self, zip_stream: BytesIO):
        doc_list = []
        with zipfile.ZipFile(zip_stream, 'r') as zip_file:
            for name in zip_file.namelist():
                if name.lower().endswith('.pdf'):
                    tmp_name = zip_file.extract(name)
                    tmp_docs = self.get_pdf_splits(tmp_name)
                    doc_list.extend(tmp_docs)
                elif name.lower().endswith('.docx'):
                    tmp_name = zip_file.extract(name)
                    tmp_docs = self.get_docx_splits(tmp_name)
                    doc_list.extend(tmp_docs)
                else:
                    pass
        return doc_list

    def get_excel_splits(self, excel_file, target_col, sheet_name):

        TrialDF = pd.read_excel(
            io=excel_file, engine='openpyxl', sheet_name=sheet_name)
        df_loader = DataFrameLoader(TrialDF, page_content_column=target_col)
        excel_docs = df_loader.load()
        return excel_docs

    def get_csv_splits(self, csv_file):
        csvloader = CSVLoader(csv_file)
        csvdocs = csvloader.load()
        return csvdocs

    def get_ipynb_splits(self, notebook):
        """Function takes the notebook file,reads the file 
        data as python script, then splits script data directly"""

        with open(notebook) as fh:
            nb = nbformat.reads(fh.read(), nbformat.NO_CONVERT)

        exporter = PythonExporter()
        source, meta = exporter.from_notebook_node(nb)

        # Python file data is in the source variable

        textSplit = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size,
                                                   chunk_overlap=15,
                                                   length_function=len)
        doc_list = textSplit.split_text(source)
        return doc_list

    def get_git_files(self, repo_link, folder_path, file_ext):
        # eg. loading only python files
        git_loader = GitLoader(clone_url=repo_link,
                               repo_path=folder_path,
                               file_filter=lambda file_path: file_path.endswith(file_ext))
        # Will take each file individual document
        git_docs = git_loader.load()

        textSplit = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size,
                                                   chunk_overlap=15,
                                                   length_function=len)
        doc_list = []
        # Pages will be list of pages, so need to modify the loop
        for code in git_docs:
            code_splits = textSplit.split_text(code.page_content)
            doc_list.extend(code_splits)

        return doc_list

    def embed_existing_vectorstore(self, doc_list, embed_fn, index_store):
        '''
        Takes existing vectorstore, new doc_list and a properly initialized embed_fn.
        if no vector_index is given a new is created, otherwise new_embedding is merged with exisitng index.
        '''
        try:
            faiss_db = FAISS.from_documents(doc_list, embed_fn)
        except Exception as ex:
            faiss_db = FAISS.from_texts(doc_list, embed_fn)

        index_store_path = Path(index_store)
        if index_store_path.exists():
            local_db = FAISS.load_local(str(index_store_path), embed_fn)
            local_db.merge_from(faiss_db)
            print("Merge completed")
            local_db.save_local(index_store)
        else:
            faiss_db.save_local(folder_path=index_store)

    def create_new_vectorstore(self, doc_list, embed_fn):
        try:
            faiss_db = FAISS.from_documents(doc_list, embed_fn)
        except Exception as ex:
            faiss_db = FAISS.from_texts(doc_list, embed_fn)
        return faiss_db

    def get_docs_length(self, index_path, embed_fn):
        test_index = FAISS.load_local(index_path,
                                      embeddings=embed_fn)
        test_dict = test_index.docstore._dict
        return len(test_dict.values())
