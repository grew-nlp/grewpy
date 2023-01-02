import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='grewpy',
    version='0.1.1',
    packages=['grewpy','examples'],
    license='LICENSE/Licence_CeCILL_V2-en.txt',
    description="A binding to the Grew software",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://grew.fr",
    author="bguil",
    author_email="Bruno.Guillaume@loria.fr"
)
