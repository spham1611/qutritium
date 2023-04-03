import setuptools


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="qutritium",
    version="0.0.1",
    author="Son Pham, Tien Nguyen",
    author_email="reachphamhson@gmail.com, nguyendinhtien_t65@hus.edu.vn",
    description="Qutritium package provides basic calibration and features for qutrit system using Qiskit and other "
                "quantum packages",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/spham1611/qutritium",
    project_urls={
        'Bug Tracker: https://github.com/spham1611/qutritium/~/issues',
        'repository: https://github.com/spham1611/qutritium'
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6"
)
