import setuptools
from typing import List

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()


__version__ = "0.0.3"

REPO_NAME = "DL-ALGORITHMS"
AUTHOR_USER_NAME = "adityaiiitmk"
SRC_REPO = "DL-ALGORITHMS"
AUTHOR_EMAIL = "aditya.s@duk.ac.in"


HYPEN_E_DOT='-e .'

def get_requirements(file_path:str)->List[str]:
    '''
    this function will return the list of requirements
    '''
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    
    return requirements


setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="A Deep Learning tutorial Repo.",
    long_description=long_description,
    long_description_content="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        REPO_NAME: f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/",
    },
    packages=setuptools.find_packages(),
    install_requires=get_requirements('requirements.txt')

)