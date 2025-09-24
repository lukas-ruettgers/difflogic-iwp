from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name='difflogic_iwp',
    version='0.1.0',
    author='Anonymous',
    author_email='anonymous',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/anonymous/difflogic-iwp',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    package_dir={'difflogic': 'difflogic'},
    packages=['difflogic'],
    install_requires=[
        "torch",
        "numpy",
    ],

    ext_modules=[CUDAExtension('difflogic_cuda_iwp', [
        'difflogic/cuda/bindings_iwp.cpp',
        'difflogic/cuda/difflogic_iwp_forward_train.cu',
        'difflogic/cuda/difflogic_iwp_forward_eval.cu',
        'difflogic/cuda/difflogic_iwp_backward_w.cu',
        'difflogic/cuda/difflogic_iwp_backward_x.cu',
    ], extra_compile_args={
        'nvcc': [
            '-lineinfo',
        ]
    })],
    # ], extra_compile_args={'nvcc': ['-maxrregcount', '128']})],


    cmdclass={'build_ext': BuildExtension},
)
