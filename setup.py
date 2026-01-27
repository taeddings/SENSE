from setuptools import setup, find_packages

setup(
    name="sense",
    version="4.0.0",  # Updated to v4
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=[
        "openai",
        "pydantic",
        "aiohttp",
        "termcolor",
        "python-dotenv",
        "numpy",
        "requests",
        "rich" 
    ],
    entry_points={
        'console_scripts': [
            'sense=sense.main:main',
        ],
    },
)