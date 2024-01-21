from setuptools import setup, find_packages


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="qutritium",
    version="0.0.1",
    author="Son Pham, Tien Nguyen, Bao Bach",
    author_email="reachphamhson@gmail.com",
    description="Qutritium package provides basic calibration and features for qutrit system using Qiskit and other "
                "quantum packages",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='MIT License',
    url="https://github.com/spham1611/qutritium",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Utilities",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.6"
)

