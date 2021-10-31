import setuptools

setuptools.setup(
    name="PDNN",
    zip_safe=False,
    version="1.0.0",
    author="DAVID ISTRATI",
    author_email="istrati.david@gmail.com",
    description="neural networks that operate based on probability distribution functions rather than numbers",
    url="https://github.com/DavidIstrati/PDNN",
    package_dir={"": "src"},
    packages=setuptools.find_packages("src"),
    # package_data={"": ["stopwords/*/*"]},
    python_requires=">=3.7",
)
