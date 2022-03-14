import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="COVID_XAI", 
    version="0.0.1",
    author="Iván Sevillano Garcia",
    author_email="isevillano@ugr.es",
    description="COVIDGR",
    long_description=long_description,
    classifiers=[
        "Programming Language :: Python :: 3"
    ],
    packages=["COVID_XAI"],
)
