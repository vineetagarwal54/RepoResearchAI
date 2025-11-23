import os
import tempfile
from langchain_community.document_loaders import GitLoader, DirectoryLoader
# from langchain.docstore.document import Document
from models import Document
from langchain_community.document_loaders import UnstructuredFileLoader
import zipfile
import nbformat
from pathlib import Path

class RepoLoader:
    @staticmethod
    def load_zip(zip_file_path: str):
        temp_dir = tempfile.mkdtemp()
        with zipfile.ZipFile(zip_file_path, "r") as z:
            z.extractall(temp_dir)
        return temp_dir
    
    @staticmethod
    def load_github(url: str) -> str:
        """
        Clones GitHub repo using LangChain GitLoader
        Tries 'main' branch first, falls back to 'master' if that fails
        """
        temp_dir = tempfile.mkdtemp()
        
        # Try main branch first
        try:
            loader = GitLoader(repo_path=temp_dir, clone_url=url, branch="main")
            loader.load()  # Actually clones the repo
            return temp_dir
        except Exception as e:
            print(f"Failed to clone 'main' branch: {e}")
            # Try master branch as fallback
            try:
                import shutil
                shutil.rmtree(temp_dir)  # Clean up failed clone
                temp_dir = tempfile.mkdtemp()
                loader = GitLoader(repo_path=temp_dir, clone_url=url, branch="master")
                loader.load()
                return temp_dir
            except Exception as e2:
                print(f"Failed to clone 'master' branch: {e2}")
                # Last resort: try without specifying branch
                try:
                    import shutil
                    shutil.rmtree(temp_dir)
                    temp_dir = tempfile.mkdtemp()
                    loader = GitLoader(repo_path=temp_dir, clone_url=url)
                    loader.load()
                    return temp_dir
                except Exception as e3:
                    raise ValueError(f"Failed to clone repository: {e3}")
    
    @staticmethod
    def load_repo(input_value: str) -> str:
        if input_value.endswith(".zip") and os.path.exists(input_value):
            return RepoLoader.load_zip(input_value)
        elif input_value.startswith("http"):
            return RepoLoader.load_github(input_value)
        else:
            raise ValueError("Invalid input type or file path does not exist")
        
    @staticmethod
    def load_documents(repo_path: str) -> list[Document]:
        def is_valid_file(file_path):
            parts = file_path.split(os.sep)
            for part in parts:
                if part.startswith(".") or part in ["node_modules", "__pycache__"]:
                    return False
            filename = os.path.basename(file_path)
            if filename.startswith(".env"):
                return False
            return os.path.isfile(file_path)
        
        def notebook_to_code(file_path: str):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    nb = nbformat.read(f, as_version=4)
                code_cells = [cell.source for cell in nb.cells if cell.cell_type == "code"]
                return "\n\n".join(code_cells)
            except Exception as e:
                print(f"Failed to read notebook {file_path}: {e}")
                return ""
        
        documents = []
        for root, dirs, files in os.walk(repo_path):
            for file in files:
                file_path = os.path.join(root, file)
                ext = Path(file_path).suffix.lower()

                if not is_valid_file(file_path):
                    continue

                if ext == ".ipynb":
                    content = notebook_to_code(file_path)
                    if content.strip():
                        documents.append(Document(page_content=content, metadata={"source":file_path}))
                    else:
                        print("No code in the notebook")
                else:
                    loader = DirectoryLoader(
                        repo_path,
                        glob="**/*",
                        show_progress=True,
                        loader_cls=UnstructuredFileLoader,
                        # show_progress=True,
                        silent_errors=True,
                    )

                    # documents = []
                    for doc in loader.load():
                        file_path = doc.metadata.get("source", "")
                        if is_valid_file(file_path):
                            documents.append(Document(page_content=doc.page_content, metadata=doc.metadata))
        return documents
    

