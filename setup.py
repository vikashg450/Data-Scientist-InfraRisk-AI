from setuptools import find_packages, setup

setup(
    name="infrarisk_ai",
    version="0.1.0",
    description="InfraRisk AI - Infrastructure Finance and Risk Modelling & Credit Assessment Platform",
    author="Data Scientist",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        # Dependencies are managed via requirements.txt
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
)
