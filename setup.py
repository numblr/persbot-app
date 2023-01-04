from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("VERSION", "r") as fh:
    version = fh.read().strip()

setup(
    name='cltl.eliza-app',
    version=version,
    package_dir={'': 'py-app'},
    packages=find_packages(include=['*'], where='py-app'),
    data_files=[('VERSION', ['VERSION'])],
    url="https://github.com/leolani/cltl-eliza-app",
    license='MIT License',
    author='CLTL',
    author_email='t.baier@vu.nl',
    description='VAD for Leolani',
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires='>=3.8',
    install_requires=[
        "cltl.backend[impl,host,service]",
        "cltl.asr[google,service]",
        "cltl.vad[impl,service]",
        "cltl.chat-ui",
        "persbot",
        "flask",
        "werkzeug"
    ],
    entry_points={
        'eliza': [ 'eliza = app:main']
    }
)
